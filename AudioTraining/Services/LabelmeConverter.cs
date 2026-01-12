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
                if (classId == -1) continue; // Skip unknown classes

                var points = shape["points"] as JArray;
                if (points == null || points.Count == 0) continue;

                // Calculate bounding box from points
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

                // Normalize
                float normX = centerX / imgWidth;
                float normY = centerY / imgHeight;
                float normW = width / imgWidth;
                float normH = height / imgHeight;

                // Clamp
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
                    LoggerService.Error(ex, $"Failed to convert {jsonFile} to OBB");
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

                // OBB format requires exactly 4 points for rotated rectangle
                // If polygon has 4 points, use them directly
                // Otherwise, calculate bounding box and convert to 4 corner points
                string shapeType = shape["shape_type"]?.ToString();

                if (shapeType == "polygon" && points.Count == 4)
                {
                    // Direct 4-point polygon - ideal for OBB
                    StringBuilder sb = new StringBuilder();
                    sb.Append(classId);

                    for (int i = 0; i < 4; i++)
                    {
                        float x = points[i][0].Value<float>() / imgWidth;
                        float y = points[i][1].Value<float>() / imgHeight;
                        
                        // Clamp to [0, 1]
                        x = Math.Max(0, Math.Min(1, x));
                        y = Math.Max(0, Math.Min(1, y));
                        
                        sb.Append($" {x:F6} {y:F6}");
                    }

                    yoloLines.Add(sb.ToString());
                }
                else
                {
                    // For non-4-point shapes, calculate axis-aligned bbox and convert to 4 corners
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

                    // Normalize coordinates
                    float x1 = minX / imgWidth;
                    float y1 = minY / imgHeight;
                    float x2 = maxX / imgWidth;
                    float y2 = minY / imgHeight;
                    float x3 = maxX / imgWidth;
                    float y3 = maxY / imgHeight;
                    float x4 = minX / imgWidth;
                    float y4 = maxY / imgHeight;

                    // Clamp all coordinates
                    x1 = Math.Max(0, Math.Min(1, x1));
                    y1 = Math.Max(0, Math.Min(1, y1));
                    x2 = Math.Max(0, Math.Min(1, x2));
                    y2 = Math.Max(0, Math.Min(1, y2));
                    x3 = Math.Max(0, Math.Min(1, x3));
                    y3 = Math.Max(0, Math.Min(1, y3));
                    x4 = Math.Max(0, Math.Min(1, x4));
                    y4 = Math.Max(0, Math.Min(1, y4));

                    yoloLines.Add($"{classId} {x1:F6} {y1:F6} {x2:F6} {y2:F6} {x3:F6} {y3:F6} {x4:F6} {y4:F6}");
                }
            }

            string txtPath = Path.ChangeExtension(jsonPath, ".txt");
            File.WriteAllLines(txtPath, yoloLines);
        }
    }
}
