using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace AudioTraining.Services
{
    public static class DatasetManager
    {
        public static string PrepareDataset(string sourceFolder, string classesFile, float valSplit = 0.2f)
        {
            // 1. Setup Target Directory
            // We create a "Dataset_Prepared" folder inside the source folder or a temp location.
            // Let's put it alongside the source folder to avoid recursive issues if source is root.
            string datasetRoot = Path.Combine(Directory.GetParent(sourceFolder).FullName, "Dataset_Prepared");
            if (Directory.Exists(datasetRoot))
            {
                // Simple cleanup or just overwrite. For safety, let's delete and recreate.
                try { Directory.Delete(datasetRoot, true); } catch { }
            }
            Directory.CreateDirectory(datasetRoot);

            string trainImgDir = Path.Combine(datasetRoot, "train", "images");
            string trainLblDir = Path.Combine(datasetRoot, "train", "labels");
            string valImgDir = Path.Combine(datasetRoot, "val", "images");
            string valLblDir = Path.Combine(datasetRoot, "val", "labels");

            Directory.CreateDirectory(trainImgDir);
            Directory.CreateDirectory(trainLblDir);
            Directory.CreateDirectory(valImgDir);
            Directory.CreateDirectory(valLblDir);

            // 2. Gather Files
            // We assume LabelmeConverter has already run, so we have .jpg/.png and .txt pairs.
            var ext = new List<string> { ".jpg", ".jpeg", ".png", ".bmp" };
            var imageFiles = Directory.GetFiles(sourceFolder, "*.*")
                .Where(s => ext.Contains(Path.GetExtension(s).ToLower()))
                .ToList();

            // Shuffle
            var rng = new Random();
            int n = imageFiles.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                var value = imageFiles[k];
                imageFiles[k] = imageFiles[n];
                imageFiles[n] = value;
            }

            // 3. Split and Copy
            int valCount = (int)(imageFiles.Count * valSplit);
            
            // Fix for small datasets: ensure at least 1 val image if we have >1 images
            if (valCount == 0 && imageFiles.Count > 1) valCount = 1;

            // Also ensure we don't put everything in val (though unlikely with default 0.2 split)
            if (valCount >= imageFiles.Count && imageFiles.Count > 1) valCount = imageFiles.Count - 1;

            // Special case: Single image -> copy to BOTH train and val to avoid empty set errors
            if (imageFiles.Count == 1)
            {
                CopyFilePair(imageFiles[0], sourceFolder, trainImgDir, trainLblDir);
                CopyFilePair(imageFiles[0], sourceFolder, valImgDir, valLblDir);
            }
            else
            {
                for (int i = 0; i < imageFiles.Count; i++)
                {
                    string targetImgDir = (i < valCount) ? valImgDir : trainImgDir;
                    string targetLblDir = (i < valCount) ? valLblDir : trainLblDir;
                    CopyFilePair(imageFiles[i], sourceFolder, targetImgDir, targetLblDir);
                }
            }

            // 4. Generate data.yaml
            GenerateDataYaml(datasetRoot, classesFile);

            return datasetRoot;
        }

        private static void CopyFilePair(string srcImgPath, string sourceFolder, string targetImgDir, string targetLblDir)
        {
            string fileNameNoExt = Path.GetFileNameWithoutExtension(srcImgPath);
            string srcTxtPath = Path.Combine(sourceFolder, fileNameNoExt + ".txt");

            // Copy Image
            string destImgPath = Path.Combine(targetImgDir, Path.GetFileName(srcImgPath));
            File.Copy(srcImgPath, destImgPath, true);

            // Copy Label (if exists)
            if (File.Exists(srcTxtPath))
            {
                string destTxtPath = Path.Combine(targetLblDir, fileNameNoExt + ".txt");
                File.Copy(srcTxtPath, destTxtPath, true);
            }
        }

        private static void GenerateDataYaml(string datasetRoot, string classesFile)
        {
            var lines = File.ReadAllLines(classesFile)
               .Where(x => !string.IsNullOrWhiteSpace(x))
               .Select(x => x.Trim())
               .ToList();

            if (lines.Count == 0) throw new Exception("classes.txt 内容为空。");

            var sb = new StringBuilder();
            // Use forward slashes for YAML compatibility, although standard YAML supports relative paths nicely
            // We will use absolute paths to be safe, or relative to the yaml file.
            
            // YOLOv8 expects 'train' and 'val' keys.
            // Since data.yaml is at datasetRoot, we can point to train/images and val/images
            
            sb.AppendLine($"path: {datasetRoot.Replace("\\", "/")}"); // Optional: dataset root dir
            sb.AppendLine("train: train/images");
            sb.AppendLine("val: val/images");
            
            sb.AppendLine();
            sb.AppendLine($"nc: {lines.Count}");
            sb.AppendLine("names:");
            for (int i = 0; i < lines.Count; i++)
            {
                sb.AppendLine($"  {i}: \"{lines[i]}\"");
            }

            string yamlPath = Path.Combine(datasetRoot, "data.yaml");
            File.WriteAllText(yamlPath, sb.ToString());
        }
    }
}
