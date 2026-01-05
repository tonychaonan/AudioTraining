using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

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
                    Console.WriteLine($"Failed to convert {jsonFile}: {ex.Message}");
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
    }
}
