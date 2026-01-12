using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using AudioTraining.Services;

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

        public List<YoloOBBPrediction> Predict(Bitmap image, float confidenceThreshold = 0.5f)
        {
            TopConfidence = 0f;
            if (_inferenceSession == null) throw new InvalidOperationException("Model not loaded.");

            var (tensor, scale, xPadding, yPadding) = Preprocess(image);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", tensor)
            };

            using (var results = _inferenceSession.Run(inputs))
            {
                var output = results.First().AsTensor<float>();
                return Postprocess(output, image.Width, image.Height, scale, xPadding, yPadding, confidenceThreshold);
            }
        }

        private (DenseTensor<float> Tensor, float Scale, float XPadding, float YPadding) Preprocess(Bitmap image)
        {
            int w = image.Width;
            int h = image.Height;

            float scale = Math.Min((float)_targetSize / w, (float)_targetSize / h);
            int newWidth = (int)(w * scale);
            int newHeight = (int)(h * scale);

            int xPadding = (_targetSize - newWidth) / 2;
            int yPadding = (_targetSize - newHeight) / 2;

            var tensor = new DenseTensor<float>(new[] { 1, 3, _targetSize, _targetSize });

            using (var resized = new Bitmap(_targetSize, _targetSize, PixelFormat.Format24bppRgb))
            using (var g = Graphics.FromImage(resized))
            {
                g.Clear(Color.FromArgb(114, 114, 114));
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Bicubic;
                g.DrawImage(image, xPadding, yPadding, newWidth, newHeight);

                var data = resized.LockBits(new Rectangle(0, 0, _targetSize, _targetSize), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);

                try
                {
                    unsafe
                    {
                        byte* ptr = (byte*)data.Scan0;
                        int stride = data.Stride;

                        for (int y = 0; y < _targetSize; y++)
                        {
                            for (int x = 0; x < _targetSize; x++)
                            {
                                int offset = y * stride + x * 3;
                                tensor[0, 0, y, x] = ptr[offset + 2] / 255.0f; // R
                                tensor[0, 1, y, x] = ptr[offset + 1] / 255.0f; // G
                                tensor[0, 2, y, x] = ptr[offset + 0] / 255.0f; // B
                            }
                        }
                    }
                }
                finally
                {
                    resized.UnlockBits(data);
                }
            }

            return (tensor, scale, xPadding, yPadding);
        }

        private List<YoloOBBPrediction> Postprocess(Tensor<float> output, int originalW, int originalH, float scale, float xPad, float yPad, float confThreshold)
        {
            var predictions = new List<YoloOBBPrediction>();

            // YOLOv8-OBB output shape: [1, 5+classes, 8400]
            // Channels 0-3: cx, cy, w, h (box coordinates)
            // Channel 4: angle (rotation angle in radians)
            // Channels 5+: class confidence scores

            int dim1 = output.Dimensions[1];
            int dim2 = output.Dimensions[2];

            LoggerService.LogOBBInference(
                new int[] { output.Dimensions[0], dim1, dim2 },
                dim1 - 5,
                dim2,
                confThreshold
            );
            
            // 诊断：检查输出张量的值范围
            float minVal = float.MaxValue;
            float maxVal = float.MinValue;
            float sumVal = 0;
            int sampleCount = 0;
            for (int i = 0; i < Math.Min(100, dim2); i++)
            {
                for (int c = 5; c < dim1; c++)
                {
                    float val = output[0, c, i];
                    if (val < minVal) minVal = val;
                    if (val > maxVal) maxVal = val;
                    sumVal += val;
                    sampleCount++;
                }
            }
            float avgVal = sumVal / sampleCount;
            LoggerService.Debug($"【模型输出诊断】前100个锚点的类别分数: 最小={minVal:F4}, 最大={maxVal:F4}, 平均={avgVal:F4}");

            // YOLOv8-OBB format is always [1, 5+classes, anchors]
            int numClasses = dim1 - 5;  // Total channels minus 5 (4 coords + 1 angle)
            int numAnchors = dim2;

            if (numClasses < 1) 
            {
                LoggerService.Warn($"OBB推理失败: 类别数无效 (numClasses={numClasses})");
                return predictions;
            }

            float globalMaxScore = 0f;
            int validCount = 0;

            for (int i = 0; i < numAnchors; i++)
            {
                // Find best class from channels 5 onwards
                float maxClassScore = 0;
                int maxClassId = -1;

                for (int c = 0; c < numClasses; c++)
                {
                    float rawScore = output[0, 5 + c, i];  // Channels 5+ are class scores (raw logits)
                    float score = Sigmoid(rawScore);  // Apply sigmoid to convert to probability

                    if (score > maxClassScore)
                    {
                        maxClassScore = score;
                        maxClassId = c;
                    }
                }

                if (maxClassScore > globalMaxScore) globalMaxScore = maxClassScore;

                // Log first 10 detections with raw values
                if (i < 10)
                {
                    float[] rawScores = new float[numClasses];
                    float[] sigmoidScores = new float[numClasses];
                    for (int c = 0; c < numClasses; c++)
                    {
                        rawScores[c] = output[0, 5 + c, i];
                        sigmoidScores[c] = Sigmoid(rawScores[c]);
                    }
                    
                    // Log raw box coordinates too
                    float rawCx = output[0, 0, i];
                    float rawCy = output[0, 1, i];
                    float rawW = output[0, 2, i];
                    float rawH = output[0, 3, i];
                    float rawAngle = output[0, 4, i];
                    
                    LoggerService.Debug($"【锚点{i}详细】原始分数=[{string.Join(", ", Array.ConvertAll(rawScores, s => s.ToString("F4")))}], Sigmoid后=[{string.Join(", ", Array.ConvertAll(sigmoidScores, s => s.ToString("F4")))}]");
                    LoggerService.Debug($"【锚点{i}坐标】cx={rawCx:F2}, cy={rawCy:F2}, w={rawW:F2}, h={rawH:F2}, angle={rawAngle:F4}");
                }

                if (maxClassScore < confThreshold) continue;
                
                validCount++;

                // Extract box parameters from channels 0-4
                float cx = output[0, 0, i];      // Channel 0: center x
                float cy = output[0, 1, i];      // Channel 1: center y
                float w = output[0, 2, i];       // Channel 2: width
                float h = output[0, 3, i];       // Channel 3: height
                float angle = output[0, 4, i];   // Channel 4: angle in radians

                // Convert to original image coordinates
                cx = (cx - xPad) / scale;
                cy = (cy - yPad) / scale;
                w = w / scale;
                h = h / scale;

                // CRITICAL FIX: Validate box dimensions to filter out invalid detections
                if (w <= 0 || h <= 0 || w > originalW * 2 || h > originalH * 2)
                    continue;

                // Additional validation: check if box is within reasonable bounds
                if (cx < -originalW || cx > originalW * 2 || cy < -originalH || cy > originalH * 2)
                    continue;

                // Calculate 4 corner points of rotated rectangle
                PointF[] corners = CalculateRotatedCorners(cx, cy, w, h, angle);

                predictions.Add(new YoloOBBPrediction
                {
                    ClassId = maxClassId,
                    Label = _labels != null && maxClassId < _labels.Length ? _labels[maxClassId] : maxClassId.ToString(),
                    Confidence = maxClassScore,
                    RotatedBox = corners,
                    Angle = (float)(angle * 180.0 / Math.PI) // Convert to degrees
                });
            }

            TopConfidence = globalMaxScore;
            
            // 统计分析：检查模型输出的分布
            int countAbove08 = 0;
            int countAbove06 = 0;
            int countAbove05 = 0;
            for (int i = 0; i < Math.Min(1000, numAnchors); i++)
            {
                for (int c = 0; c < numClasses; c++)
                {
                    float score = Sigmoid(output[0, 5 + c, i]);
                    if (score > 0.8f) countAbove08++;
                    if (score > 0.6f) countAbove06++;
                    if (score > 0.5f) countAbove05++;
                }
            }
            
            LoggerService.Info($"OBB推理完成: 通过阈值={validCount}, 验证后={predictions.Count}, 最高置信度={globalMaxScore:F4}");
            LoggerService.Info($"【统计分析】前1000个锚点中: >0.8={countAbove08}, >0.6={countAbove06}, >0.5={countAbove05}");

            return NMS(predictions);
        }

        private PointF[] CalculateRotatedCorners(float cx, float cy, float w, float h, float angle)
        {
            // Calculate 4 corners of rotated rectangle
            float cos = (float)Math.Cos(angle);
            float sin = (float)Math.Sin(angle);

            float halfW = w / 2;
            float halfH = h / 2;

            PointF[] corners = new PointF[4];

            // Top-left
            corners[0] = new PointF(
                cx + (-halfW * cos - (-halfH) * sin),
                cy + (-halfW * sin + (-halfH) * cos)
            );

            // Top-right
            corners[1] = new PointF(
                cx + (halfW * cos - (-halfH) * sin),
                cy + (halfW * sin + (-halfH) * cos)
            );

            // Bottom-right
            corners[2] = new PointF(
                cx + (halfW * cos - halfH * sin),
                cy + (halfW * sin + halfH * cos)
            );

            // Bottom-left
            corners[3] = new PointF(
                cx + (-halfW * cos - halfH * sin),
                cy + (-halfW * sin + halfH * cos)
            );

            return corners;
        }

        private List<YoloOBBPrediction> NMS(List<YoloOBBPrediction> predictions, float iouThreshold = 0.45f)
        {
            var result = new List<YoloOBBPrediction>();
            var sorted = predictions.OrderByDescending(p => p.Confidence).ToList();

            while (sorted.Count > 0)
            {
                var current = sorted[0];
                result.Add(current);
                sorted.RemoveAt(0);

                for (int i = sorted.Count - 1; i >= 0; i--)
                {
                    if (CalculateRotatedIoU(current.RotatedBox, sorted[i].RotatedBox) > iouThreshold)
                    {
                        sorted.RemoveAt(i);
                    }
                }
            }

            return result;
        }

        private float CalculateRotatedIoU(PointF[] box1, PointF[] box2)
        {
            // Simplified IoU calculation for rotated boxes
            // For accurate OBB IoU, you would need polygon intersection algorithms
            // Here we use bounding box approximation for simplicity

            float minX1 = box1.Min(p => p.X);
            float maxX1 = box1.Max(p => p.X);
            float minY1 = box1.Min(p => p.Y);
            float maxY1 = box1.Max(p => p.Y);

            float minX2 = box2.Min(p => p.X);
            float maxX2 = box2.Max(p => p.X);
            float minY2 = box2.Min(p => p.Y);
            float maxY2 = box2.Max(p => p.Y);

            float x1 = Math.Max(minX1, minX2);
            float y1 = Math.Max(minY1, minY2);
            float x2 = Math.Min(maxX1, maxX2);
            float y2 = Math.Min(maxY1, maxY2);

            float intersectionW = Math.Max(0, x2 - x1);
            float intersectionH = Math.Max(0, y2 - y1);
            float intersectionArea = intersectionW * intersectionH;

            float area1 = (maxX1 - minX1) * (maxY1 - minY1);
            float area2 = (maxX2 - minX2) * (maxY2 - minY2);

            if (area1 + area2 - intersectionArea == 0) return 0;

            return intersectionArea / (area1 + area2 - intersectionArea);
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
