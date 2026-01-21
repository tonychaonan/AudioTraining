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
        public PointF[] RotatedBox { get; set; } // 旋转模型的4个角点
        public float Angle { get; set; } 
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

            LoggerService.Debug($"[OBB推理] 原始图像尺寸: {image.Width}x{image.Height}");

            // 转换Bitmap到OpenCvSharp Mat
            Mat mat = BitmapToMat(image);

            // 1. 简单缩放预处理
            Mat inputMat = ResizeImage(mat, _targetSize, out float ratio, out int padLeft, out int padTop);

            LoggerService.Debug($"[OBB推理] 缩放比例: {ratio}, 输入尺寸: {inputMat.Width}x{inputMat.Height}, Padding: left={padLeft}, top={padTop}");

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
                // 4. 后处理
                predictions = PostprocessSimple(output, image.Width, image.Height, ratio, padLeft, padTop, confThreshold, iouThreshold);
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

        // 简单缩放预处理
        private Mat ResizeImage(Mat img, int targetSize, out float ratio, out int padLeft, out int padTop)
        {
            var shape = img.Size();
            // 计算缩放比例（保持宽高比）
            ratio = Math.Min((float)targetSize / shape.Width, (float)targetSize / shape.Height);

            int newW = (int)Math.Round(shape.Width * ratio);
            int newH = (int)Math.Round(shape.Height * ratio);

            Mat resized = new Mat();
            Cv2.Resize(img, resized, new OpenCvSharp.Size(newW, newH));

            // 如果不是正方形，填充到正方形
            if (newW != targetSize || newH != targetSize)
            {
                Mat padded = new Mat();
                padTop = (targetSize - newH) / 2;
                int bottom = targetSize - newH - padTop;
                padLeft = (targetSize - newW) / 2;
                int right = targetSize - newW - padLeft;
                
                Cv2.CopyMakeBorder(resized, padded, padTop, bottom, padLeft, right, 
                    BorderTypes.Constant, new Scalar(114, 114, 114));
                resized.Dispose();
                return padded;
            }

            padLeft = 0;
            padTop = 0;
            return resized;
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

        // 简化后处理
        private List<YoloOBBPrediction> PostprocessSimple(Tensor<float> output, int originalW, int originalH, float ratio, int padLeft, int padTop, float confThreshold, float iouThreshold)
        {
            var results = new List<YoloOBBPrediction>();

            // YOLOv8-OBB输出: [1, num_channels, 8400]
            int dimensions = output.Dimensions[1];
            int rows = output.Dimensions[2]; // 8400 anchor
            int numClasses = dimensions - 5; // 前4个是xywh，最后1个是angle，中间是类别

            LoggerService.Debug($"[OBB简化后处理] 输出形状: [{output.Dimensions[0]}, {dimensions}, {rows}], 类别数: {numClasses}");

            // 第一步：置信度过滤
            var filteredData = new List<(float x, float y, float w, float h, float conf, int classId, float angle)>();

            for (int i = 0; i < rows; i++)
            {
                // 找出最高置信度的类别（从channel 4开始）
                float maxScore = 0;
                int maxClassId = -1;

                for (int c = 0; c < numClasses; c++)
                {
                    float score = output[0, 4 + c, i];

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
                    float angle = output[0, dimensions - 1, i]; // 角度在最后一个channel

                    filteredData.Add((x, y, w, h, maxScore, maxClassId, angle));
                }
            }

            LoggerService.Debug($"[OBB简化后处理] 通过置信度阈值: {filteredData.Count}/{rows}");

            if (filteredData.Count == 0)
            {
                return results;
            }

            // 第二步：NMS，使用轴对齐IoU
            var nmsResults = NMSSimple(filteredData, iouThreshold);

            LoggerService.Debug($"[OBB简化后处理] NMS后: {nmsResults.Count}");

            // 第三步：坐标还原和构建结果
            foreach (var data in nmsResults)
            {
                LoggerService.Debug($"[坐标还原] 模型输出: cx={data.x:F2}, cy={data.y:F2}, w={data.w:F2}, h={data.h:F2}, angle={data.angle:F4}");
                
                // 还原到原始图像坐标：先减去padding，再除以缩放比
                float x = (data.x - padLeft) / ratio;
                float y = (data.y - padTop) / ratio;
                float w = data.w / ratio;
                float h = data.h / ratio;
                float angle = data.angle; // 角度不缩放

                LoggerService.Debug($"[坐标还原] 还原后: cx={x:F2}, cy={y:F2}, w={w:F2}, h={h:F2}, angle={angle:F4} (ratio={ratio:F4}, padLeft={padLeft}, padTop={padTop})");

                // 计算旋转矩形的4个角点
                PointF[] corners = CalculateRotatedCorners(x, y, w, h, angle);

                LoggerService.Debug($"[坐标还原] 角点: P1=({corners[0].X:F2},{corners[0].Y:F2}), P2=({corners[1].X:F2},{corners[1].Y:F2}), P3=({corners[2].X:F2},{corners[2].Y:F2}), P4=({corners[3].X:F2},{corners[3].Y:F2})");

                results.Add(new YoloOBBPrediction
                {
                    ClassId = data.classId,
                    Label = _labels != null && data.classId < _labels.Length ? _labels[data.classId] : data.classId.ToString(),
                    Confidence = data.conf,
                    RotatedBox = corners,
                    Angle = angle * 180f / (float)Math.PI // 转换为度数
                });
            }

            TopConfidence = results.Count > 0 ? results.Max(r => r.Confidence) : 0;

            return results;
        }

        // 简化NMS，使用轴对齐IoU
        private List<(float x, float y, float w, float h, float conf, int classId, float angle)> NMSSimple(
            List<(float x, float y, float w, float h, float conf, int classId, float angle)> data, float iouThreshold)
        {
            // 按置信度降序排序
            var sorted = data.OrderByDescending(d => d.conf).ToList();
            var result = new List<(float x, float y, float w, float h, float conf, int classId, float angle)>();

            while (sorted.Count > 0)
            {
                var current = sorted[0];
                result.Add(current);
                sorted.RemoveAt(0);

                // 移除与当前框IoU过高的框
                for (int i = sorted.Count - 1; i >= 0; i--)
                {
                    float iou = CalculateAxisAlignedIoU(current, sorted[i]);
                    if (iou > iouThreshold)
                    {
                        sorted.RemoveAt(i);
                    }
                }
            }

            return result;
        }

        // 计算轴对齐矩形IoU
        private float CalculateAxisAlignedIoU(
            (float x, float y, float w, float h, float conf, int classId, float angle) box1,
            (float x, float y, float w, float h, float conf, int classId, float angle) box2)
        {
            // 转换为左上角和右下角坐标
            float x1_min = box1.x - box1.w / 2;
            float y1_min = box1.y - box1.h / 2;
            float x1_max = box1.x + box1.w / 2;
            float y1_max = box1.y + box1.h / 2;

            float x2_min = box2.x - box2.w / 2;
            float y2_min = box2.y - box2.h / 2;
            float x2_max = box2.x + box2.w / 2;
            float y2_max = box2.y + box2.h / 2;

            // 计算交集
            float inter_x_min = Math.Max(x1_min, x2_min);
            float inter_y_min = Math.Max(y1_min, y2_min);
            float inter_x_max = Math.Min(x1_max, x2_max);
            float inter_y_max = Math.Min(y1_max, y2_max);

            float intersectionArea = 0f;
            if (inter_x_min < inter_x_max && inter_y_min < inter_y_max)
            {
                intersectionArea = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min);
            }

            // 计算并集
            float area1 = box1.w * box1.h;
            float area2 = box2.w * box2.h;
            float unionArea = area1 + area2 - intersectionArea;

            if (unionArea <= 0) return 0f;

            return intersectionArea / unionArea;
        }

        // 计算旋转矩形的4个角点
        private PointF[] CalculateRotatedCorners(float cx, float cy, float w, float h, float angle)
        {
            float cos_value = (float)Math.Cos(angle);
            float sin_value = (float)Math.Sin(angle);

            float[] vec1 = { w / 2 * cos_value, w / 2 * sin_value };
            float[] vec2 = { -h / 2 * sin_value, h / 2 * cos_value };

            return new PointF[]
            {
                new PointF(cx + vec1[0] + vec2[0], cy + vec1[1] + vec2[1]),
                new PointF(cx + vec1[0] - vec2[0], cy + vec1[1] - vec2[1]),
                new PointF(cx - vec1[0] - vec2[0], cy - vec1[1] - vec2[1]),
                new PointF(cx - vec1[0] + vec2[0], cy - vec1[1] + vec2[1])
            };
        }



        public void Dispose()
        {
            _inferenceSession?.Dispose();
        }
    }
}
