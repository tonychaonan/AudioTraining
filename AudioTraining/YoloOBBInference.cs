using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using AudioTraining.Services;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace AudioTraining
{
    public class YoloOBBPrediction
    {
        public int ClassId { get; set; }
        public string Label { get; set; }
        public float Confidence { get; set; }
        public PointF[] RotatedBox { get; set; } // 4 corner points
        public float Angle { get; set; } // Rotation angle in degrees (optional)
    }

    public class YoloOBBInference : IDisposable
    {
        private InferenceSession _inferenceSession;
        private readonly int _targetSize = 640;
        private string[] _labels;

        public bool IsModelLoaded => _inferenceSession != null;

        public void LoadModel(string modelPath, string[] labels = null)
        {
            if (_inferenceSession != null)
            {
                _inferenceSession.Dispose();
                _inferenceSession = null;
            }

            try
            {
                var options = new SessionOptions();
                _inferenceSession = new InferenceSession(modelPath, options);
                _labels = labels;
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to load YOLOv8-OBB model from {modelPath}: {ex.Message}", ex);
            }
        }

        public float TopConfidence { get; private set; }

        public List<YoloOBBPrediction> Predict(Bitmap image, float confThreshold = 0.25f, float iouThreshold = 0.45f)
        {
            if (_inferenceSession == null)
            {
                throw new InvalidOperationException("Model not loaded. Call LoadModel first.");
            }

            // 转换Bitmap到OpenCvSharp Mat
            Mat mat = BitmapToMat(image);

            // 1. Letterbox预处理（与Python Ultralytics一致）
            Mat inputMat = Letterbox(mat, new OpenCvSharp.Size(_targetSize, _targetSize), out float ratio, out float dw, out float dh);

            // 2. Mat转Tensor
            var inputTensor = MatToTensor(inputMat);

            // 3. 推理
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inferenceSession.InputMetadata.Keys.First(), inputTensor)
            };

            List<YoloOBBPrediction> predictions;
            using (var results = _inferenceSession.Run(inputs))
            {
                var output = results.First().AsTensor<float>();
                // 4. 后处理（使用OpenCvSharp的NMSBoxesRotated）
                predictions = PostprocessWithOpenCV(output, ratio, dw, dh, confThreshold, iouThreshold);
            }

            mat.Dispose();
            inputMat.Dispose();

            return predictions;
        }

        // 转换Bitmap到OpenCvSharp Mat
        private Mat BitmapToMat(Bitmap bitmap)
        {
            // 使用OpenCvSharp的扩展方法
            return OpenCvSharp.Extensions.BitmapConverter.ToMat(bitmap);
        }

        // Letterbox预处理（与Python Ultralytics一致）
        private Mat Letterbox(Mat img, OpenCvSharp.Size newShape, out float ratio, out float dw, out float dh)
        {
            var shape = img.Size();
            float r = Math.Min((float)newShape.Width / shape.Width, (float)newShape.Height / shape.Height);

            // 不进行upscale
            if (r > 1f) r = 1f;

            int newUnpadW = (int)Math.Round(shape.Width * r);
            int newUnpadH = (int)Math.Round(shape.Height * r);

            dw = (newShape.Width - newUnpadW) / 2f;
            dh = (newShape.Height - newUnpadH) / 2f;

            ratio = r;

            Mat resized = new Mat();
            Cv2.Resize(img, resized, new OpenCvSharp.Size(newUnpadW, newUnpadH));

            Mat bordered = new Mat();
            // 填充灰色边框 (114, 114, 114)
            Cv2.CopyMakeBorder(resized, bordered, 
                (int)Math.Floor(dh), (int)Math.Ceiling(dh), 
                (int)Math.Floor(dw), (int)Math.Ceiling(dw), 
                BorderTypes.Constant, new Scalar(114, 114, 114));

            resized.Dispose();
            return bordered;
        }

        // Mat转Tensor
        private DenseTensor<float> MatToTensor(Mat mat)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, mat.Height, mat.Width });

            // 转换颜色空间 BGR -> RGB
            Mat rgb = new Mat();
            Cv2.CvtColor(mat, rgb, ColorConversionCodes.BGR2RGB);

            // 归一化并填充Tensor (NCHW格式)
            Mat[] channels = Cv2.Split(rgb);
            
            for (int c = 0; c < 3; c++)
            {
                var indexer = channels[c].GetGenericIndexer<byte>();
                for (int y = 0; y < mat.Height; y++)
                {
                    for (int x = 0; x < mat.Width; x++)
                    {
                        tensor[0, c, y, x] = indexer[y, x] / 255.0f;
                    }
                }
                channels[c].Dispose();
            }

            rgb.Dispose();
            return tensor;
        }

        // 使用OpenCvSharp的NMSBoxesRotated进行后处理
        private List<YoloOBBPrediction> PostprocessWithOpenCV(Tensor<float> output, float ratio, float dw, float dh, float confThreshold, float iouThreshold)
        {
            var results = new List<YoloOBBPrediction>();

            // YOLOv8-OBB输出: [1, 5+classes, 8400]
            // Channels: 0-3=xywh, 4=angle, 5+=class_scores
            int dimensions = output.Dimensions[1];
            int rows = output.Dimensions[2]; // 8400 anchors
            int numClasses = dimensions - 5;

            LoggerService.Debug($"[OBB后处理OpenCV] 输出形状: [{output.Dimensions[0]}, {dimensions}, {rows}], 类别数: {numClasses}");

            var rects = new List<RotatedRect>();
            var scores = new List<float>();
            var classIds = new List<int>();

            for (int i = 0; i < rows; i++)
            {
                // 找出最高置信度的类别
                float maxScore = 0;
                int maxClassId = -1;

                for (int c = 0; c < numClasses; c++)
                {
                    float rawScore = output[0, 5 + c, i];
                    // 应用Sigmoid
                    float score = Sigmoid(rawScore);

                    if (score > maxScore)
                    {
                        maxScore = score;
                        maxClassId = c;
                    }
                }

                if (maxScore > confThreshold)
                {
                    float x = output[0, 0, i];
                    float y = output[0, 1, i];
                    float w = output[0, 2, i];
                    float h = output[0, 3, i];
                    float angle = output[0, 4, i]; // 弧度

                    // 还原坐标（去掉padding，除以缩放比）
                    x = (x - dw) / ratio;
                    y = (y - dh) / ratio;
                    w /= ratio;
                    h /= ratio;

                    // 转换为度数（OpenCV RotatedRect需要度数）
                    float angleDegree = angle * 180f / (float)Math.PI;

                    rects.Add(new RotatedRect(new Point2f(x, y), new Size2f(w, h), angleDegree));
                    scores.Add(maxScore);
                    classIds.Add(maxClassId);
                }
            }

            LoggerService.Debug($"[OBB后处理OpenCV] 通过置信度阈值: {rects.Count}/{rows}");

            // 使用自定义的旋转框NMS（OpenCvSharp的NMSBoxes不支持RotatedRect）
            if (rects.Count > 0)
            {
                //CvDnn.NMSBoxes(rects, scores, confThreshold, iouThreshold, out OpenCvSharp.Rect[] boxes, out float[] confidences, out int[] indices);
                //Cv2.nm(rects, scores, confThreshold, iouThreshold, out int[] indicesArray);
                var indices = NMSRotated(rects, scores, iouThreshold);

                LoggerService.Debug($"[OBB后处理OpenCV] NMS后: {indices.Count}");

                foreach (var idx in indices)
                {
                    var rect = rects[idx];
                    // 计算4个角点
                    Point2f[] corners = rect.Points();
                    PointF[] rotatedBox = new PointF[4];
                    for (int i = 0; i < 4; i++)
                    {
                        rotatedBox[i] = new PointF(corners[i].X, corners[i].Y);
                    }

                    results.Add(new YoloOBBPrediction
                    {
                        ClassId = classIds[idx],
                        Label = _labels != null && classIds[idx] < _labels.Length ? _labels[classIds[idx]] : classIds[idx].ToString(),
                        Confidence = scores[idx],
                        RotatedBox = rotatedBox,
                        Angle = rect.Angle
                    });
                }

                TopConfidence = results.Count > 0 ? results.Max(r => r.Confidence) : 0;
            }

            return results;
        }

        // 自定义旋转框NMS
        private List<int> NMSRotated(List<RotatedRect> rects, List<float> scores, float iouThreshold)
        {
            var result = new List<int>();
            var indices = Enumerable.Range(0, rects.Count).OrderByDescending(i => scores[i]).ToList();

            while (indices.Count > 0)
            {
                int current = indices[0];
                result.Add(current);
                indices.RemoveAt(0);

                for (int i = indices.Count - 1; i >= 0; i--)
                {
                    float iou = CalculateRotatedRectIoU(rects[current], rects[indices[i]]);
                    if (iou > iouThreshold)
                    {
                        indices.RemoveAt(i);
                    }
                }
            }

            return result;
        }

        // 计算两个旋转矩形的IoU
        private float CalculateRotatedRectIoU(RotatedRect rect1, RotatedRect rect2)
        {
            // 使用OpenCV的rotatedRectangleIntersection计算交集
            Point2f[] intersectionPoints;
            var intersectionType = Cv2.RotatedRectangleIntersection(rect1, rect2, out intersectionPoints);

            if (intersectionType == RectanglesIntersectTypes.None)
            {
                return 0f;
            }

            // 计算交集面积
            float intersectionArea = 0f;
            if (intersectionPoints != null && intersectionPoints.Length > 2)
            {
                // 使用contourArea计算多边形面积
                intersectionArea = (float)Math.Abs(Cv2.ContourArea(intersectionPoints));
            }

            // 计算两个矩形的面积
            float area1 = rect1.Size.Width * rect1.Size.Height;
            float area2 = rect2.Size.Width * rect2.Size.Height;

            // 计算IoU
            float unionArea = area1 + area2 - intersectionArea;
            if (unionArea <= 0) return 0f;

            return intersectionArea / unionArea;
        }


        private float Sigmoid(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }

        public void Dispose()
        {
            _inferenceSession?.Dispose();
        }
    }
}
