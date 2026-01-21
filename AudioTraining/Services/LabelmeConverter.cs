using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace AudioTraining.Services
{
    public static class LabelmeConverter
    {
        public static void GenerateClassesFile(string folderPath, string classesFilePath)
        {
            var jsonFiles = Directory.GetFiles(folderPath, "*.json");
            var uniqueLabels = new HashSet<string>();

            foreach (var jsonFile in jsonFiles)
            {
                try
                {
                    string jsonContent = File.ReadAllText(jsonFile);
                    JObject data = JObject.Parse(jsonContent);
                    var shapes = data["shapes"] as JArray;
                    if (shapes != null)
                    {
                        foreach (var shape in shapes)
                        {
                            string label = shape["label"]?.ToString();
                            if (!string.IsNullOrWhiteSpace(label))
                            {
                                uniqueLabels.Add(label);
                            }
                        }
                    }
                }
                catch { }
            }

            var sortedLabels = uniqueLabels.OrderBy(l => l).ToList();
            File.WriteAllLines(classesFilePath, sortedLabels);
        }

        public static void ConvertFolder(string folderPath, string classesFile)
        {
            if (!Directory.Exists(folderPath)) return;
            if (!File.Exists(classesFile)) throw new FileNotFoundException("Classes file not found.", classesFile);

            var classNames = File.ReadAllLines(classesFile)
                .Where(l => !string.IsNullOrWhiteSpace(l))
                .Select(l => l.Trim())
                .ToList();

            var jsonFiles = Directory.GetFiles(folderPath, "*.json");

            foreach (var jsonFile in jsonFiles)
            {
                try
                {
                    ConvertJsonToYolo(jsonFile, classNames);
                }
                catch (Exception ex)
                {
                    LoggerService.Error(ex, $"Failed to convert {jsonFile}");
                }
            }
        }

        private static void ConvertJsonToYolo(string jsonPath, List<string> classNames)
        {
            string jsonContent = File.ReadAllText(jsonPath);
            JObject data = JObject.Parse(jsonContent);

            int imgWidth = data["imageWidth"]?.Value<int>() ?? 0;
            int imgHeight = data["imageHeight"]?.Value<int>() ?? 0;

            if (imgWidth == 0 || imgHeight == 0) return;

            var shapes = data["shapes"] as JArray;
            if (shapes == null) return;

            List<string> yoloLines = new List<string>();

            foreach (var shape in shapes)
            {
                string label = shape["label"]?.ToString();
                if (string.IsNullOrEmpty(label)) continue;

                int classId = classNames.IndexOf(label);
                if (classId == -1) continue; 

                var points = shape["points"] as JArray;
                if (points == null || points.Count == 0) continue;

                float minX = float.MaxValue, minY = float.MaxValue;
                float maxX = float.MinValue, maxY = float.MinValue;

                foreach (var point in points)
                {
                    float x = point[0].Value<float>();
                    float y = point[1].Value<float>();

                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                }

                float centerX = (minX + maxX) / 2.0f;
                float centerY = (minY + maxY) / 2.0f;
                float width = maxX - minX;
                float height = maxY - minY;

                float normX = centerX / imgWidth;
                float normY = centerY / imgHeight;
                float normW = width / imgWidth;
                float normH = height / imgHeight;

                normX = Math.Max(0, Math.Min(1, normX));
                normY = Math.Max(0, Math.Min(1, normY));
                normW = Math.Max(0, Math.Min(1, normW));
                normH = Math.Max(0, Math.Min(1, normH));

                yoloLines.Add($"{classId} {normX:F6} {normY:F6} {normW:F6} {normH:F6}");
            }

            string txtPath = Path.ChangeExtension(jsonPath, ".txt");
            File.WriteAllLines(txtPath, yoloLines);
        }

        public static void ConvertFolderToOBB(string folderPath, string classesFile)
        {
            if (!Directory.Exists(folderPath)) return;
            if (!File.Exists(classesFile)) throw new FileNotFoundException("Classes file not found.", classesFile);

            var classNames = File.ReadAllLines(classesFile)
                .Where(l => !string.IsNullOrWhiteSpace(l))
                .Select(l => l.Trim())
                .ToList();

            var jsonFiles = Directory.GetFiles(folderPath, "*.json");

            foreach (var jsonFile in jsonFiles)
            {
                try
                {
                    ConvertJsonToYoloOBB(jsonFile, classNames);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to convert {jsonFile}: {ex.Message}");
                }
            }
        }

        private static void ConvertJsonToYoloOBB(string jsonPath, List<string> classNames)
        {
            string jsonContent = File.ReadAllText(jsonPath);
            JObject data = JObject.Parse(jsonContent);

            int imgWidth = data["imageWidth"]?.Value<int>() ?? 0;
            int imgHeight = data["imageHeight"]?.Value<int>() ?? 0;

            if (imgWidth == 0 || imgHeight == 0) return;

            var shapes = data["shapes"] as JArray;
            if (shapes == null) return;

            List<string> yoloLines = new List<string>();

            foreach (var shape in shapes)
            {
                string label = shape["label"]?.ToString();
                if (string.IsNullOrEmpty(label)) continue;

                int classId = classNames.IndexOf(label);
                if (classId == -1) continue;

                var points = shape["points"] as JArray;
                if (points == null || points.Count == 0) continue;

                if (points.Count == 4)
                {
                    var pts = new List<(float x, float y)>();
                    foreach (var point in points)
                    {
                        float x = point[0].Value<float>();
                        float y = point[1].Value<float>();
                        pts.Add((x, y));
                    }

                    pts = SortPointsClockwise(pts);

                    var normalizedPts = new List<float>();
                    foreach (var pt in pts)
                    {
                        float normX = Math.Max(0, Math.Min(1, pt.x / imgWidth));
                        float normY = Math.Max(0, Math.Min(1, pt.y / imgHeight));
                        normalizedPts.Add(normX);
                        normalizedPts.Add(normY);
                    }

                    yoloLines.Add($"{classId} {string.Join(" ", normalizedPts.Select(v => v.ToString("F6")))}" );
                }
                else
                {
                    float minX = float.MaxValue, minY = float.MaxValue;
                    float maxX = float.MinValue, maxY = float.MinValue;

                    foreach (var point in points)
                    {
                        float x = point[0].Value<float>();
                        float y = point[1].Value<float>();

                        if (x < minX) minX = x;
                        if (x > maxX) maxX = x;
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                    }

                    float normX1 = Math.Max(0, Math.Min(1, minX / imgWidth));
                    float normY1 = Math.Max(0, Math.Min(1, minY / imgHeight));
                    float normX2 = Math.Max(0, Math.Min(1, maxX / imgWidth));
                    float normY2 = Math.Max(0, Math.Min(1, maxY / imgHeight));

                    yoloLines.Add($"{classId} {normX1:F6} {normY1:F6} {normX2:F6} {normY1:F6} {normX2:F6} {normY2:F6} {normX1:F6} {normY2:F6}");
                }
            }

            string txtPath = Path.ChangeExtension(jsonPath, ".txt");
            File.WriteAllLines(txtPath, yoloLines);
        }

        private static List<(float x, float y)> SortPointsClockwise(List<(float x, float y)> points)
        {
            float centerX = points.Average(p => p.x);
            float centerY = points.Average(p => p.y);

            var sorted = points.OrderBy(p =>
            {
                double angle = Math.Atan2(p.y - centerY, p.x - centerX);
                return angle;
            }).ToList();

            return sorted;
        }
    }
}
