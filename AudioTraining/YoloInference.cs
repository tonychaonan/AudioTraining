using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace AudioTraining
{
    public class YoloPrediction
    {
        public int ClassId { get; set; }
        public string Label { get; set; }
        public float Confidence { get; set; }
        public RectangleF Rectangle { get; set; } // x, y, w, h
    }

    public class YoloInference : IDisposable
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
                throw new Exception($"Failed to load YOLOv8 model from {modelPath}: {ex.Message}", ex);
            }
        }

        public List<YoloPrediction> Predict(Bitmap image, float confidenceThreshold = 0.5f)
        {
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

            using (var resized = new Bitmap(_targetSize, _targetSize))
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

        private List<YoloPrediction> Postprocess(Tensor<float> output, int originalW, int originalH, float scale, float xPad, float yPad, float confThreshold)
        {
            var predictions = new List<YoloPrediction>();
            int dimensions = output.Dimensions[1];
            int numAnchors = output.Dimensions[2];
            int numClasses = dimensions - 4;

            for (int i = 0; i < numAnchors; i++)
            {
                float maxClassScore = 0;
                int maxClassId = -1;

                for (int c = 0; c < numClasses; c++)
                {
                    float score = output[0, 4 + c, i];
                    if (score > maxClassScore)
                    {
                        maxClassScore = score;
                        maxClassId = c;
                    }
                }

                if (maxClassScore < confThreshold) continue;

                float cx = output[0, 0, i];
                float cy = output[0, 1, i];
                float w = output[0, 2, i];
                float h = output[0, 3, i];

                float x = cx - w / 2;
                float y = cy - h / 2;

                x = (x - xPad) / scale;
                y = (y - yPad) / scale;
                w = w / scale;
                h = h / scale;

                x = Math.Max(0, x);
                y = Math.Max(0, y);
                w = Math.Min(w, originalW - x);
                h = Math.Min(h, originalH - y);

                predictions.Add(new YoloPrediction
                {
                    ClassId = maxClassId,
                    Label = _labels != null && maxClassId < _labels.Length ? _labels[maxClassId] : maxClassId.ToString(),
                    Confidence = maxClassScore,
                    Rectangle = new RectangleF(x, y, w, h)
                });
            }

            return NMS(predictions);
        }

        private List<YoloPrediction> NMS(List<YoloPrediction> predictions, float iouThreshold = 0.45f)
        {
            var result = new List<YoloPrediction>();
            var sorted = predictions.OrderByDescending(p => p.Confidence).ToList();

            while (sorted.Count > 0)
            {
                var current = sorted[0];
                result.Add(current);
                sorted.RemoveAt(0);

                for (int i = sorted.Count - 1; i >= 0; i--)
                {
                    if (CalculateIoU(current.Rectangle, sorted[i].Rectangle) > iouThreshold)
                    {
                        sorted.RemoveAt(i);
                    }
                }
            }

            return result;
        }

        private float CalculateIoU(RectangleF rect1, RectangleF rect2)
        {
            float x1 = Math.Max(rect1.X, rect2.X);
            float y1 = Math.Max(rect1.Y, rect2.Y);
            float x2 = Math.Min(rect1.Right, rect2.Right);
            float y2 = Math.Min(rect1.Bottom, rect2.Bottom);

            float intersectionW = Math.Max(0, x2 - x1);
            float intersectionH = Math.Max(0, y2 - y1);
            float intersectionArea = intersectionW * intersectionH;

            float area1 = rect1.Width * rect1.Height;
            float area2 = rect2.Width * rect2.Height;

            return intersectionArea / (area1 + area2 - intersectionArea);
        }

        public void Dispose()
        {
            _inferenceSession?.Dispose();
        }
    }
}
