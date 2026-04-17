using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace AudioTraining.Services
{
    public static class DatasetManager
    {
        /// <summary>
        /// 把 sourceFolder 里的图+标签按 train/val 划分并生成 data.yaml。
        ///
        /// val 集固定机制（修复原"每次训练随机分"的问题）：
        ///   - val_fixed.txt 和 Dataset_Prepared **同级** 存放（每行一个 val 文件名，无路径）
        ///   - 已经在 val_fixed.txt 里的图永远留在 val 集，不会跑到 train
        ///   - 新加入的图按 valSplit 比例随机分到 val/train，分到 val 的会追加进 val_fixed.txt
        ///   - 结果：同一张图历次训练始终在同一个集合里，新模型 vs 老模型的 mAP 可以直接对比
        ///
        /// 参数 datasetOutputRoot：
        ///   - 传 null（默认）：Dataset_Prepared 放在 sourceFolder 的父目录（flatten 模式用）
        ///   - 传具体路径：Dataset_Prepared 就用这个路径（快速模式用，避免污染用户目录的父目录）
        /// </summary>
        public static string PrepareDataset(string sourceFolder, string classesFile, float valSplit = 0.2f, string datasetOutputRoot = null)
        {
            var parentDir = Directory.GetParent(sourceFolder)?.FullName ?? sourceFolder;
            string datasetRoot = string.IsNullOrWhiteSpace(datasetOutputRoot)
                ? Path.Combine(parentDir, "Dataset_Prepared")
                : datasetOutputRoot;
            if (Directory.Exists(datasetRoot))
            {
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
            var ext = new List<string> { ".jpg", ".jpeg", ".png", ".bmp" };
            var imageFiles = Directory.GetFiles(sourceFolder, "*.*")
                .Where(s => ext.Contains(Path.GetExtension(s).ToLower()))
                .ToList();

            // val_fixed.txt 放在 Dataset_Prepared 同级目录（即用户数据根目录下），
            // 这样不管是 flatten 模式还是快速模式，路径都一致、跨训练次稳定
            string metaDir = Path.GetDirectoryName(datasetRoot) ?? parentDir;
            string valFixedPath = Path.Combine(metaDir, "val_fixed.txt");
            var valSet = LoadValFixedSet(valFixedPath);

            var rng = new Random();
            var valList = new List<string>();
            var trainList = new List<string>();

            foreach (var img in imageFiles)
            {
                string name = Path.GetFileName(img);
                if (valSet.Contains(name))
                {
                    valList.Add(img);
                }
                else
                {
                    // 新文件按 valSplit 概率划入 val，固化到 valSet
                    if (rng.NextDouble() < valSplit)
                    {
                        valList.Add(img);
                        valSet.Add(name);
                    }
                    else
                    {
                        trainList.Add(img);
                    }
                }
            }

            // 边界：整个数据集只有 1 张图 —— 两边都放一份
            if (imageFiles.Count == 1)
            {
                CopyFilePair(imageFiles[0], sourceFolder, trainImgDir, trainLblDir);
                CopyFilePair(imageFiles[0], sourceFolder, valImgDir, valLblDir);
            }
            else
            {
                // 边界：首次运行 + 新数据量少导致 val 空了，强制挑一张塞 val
                if (valList.Count == 0 && trainList.Count > 1)
                {
                    // 挑最后一张（避免总是挑第一张造成偏差）
                    var moved = trainList[trainList.Count - 1];
                    trainList.RemoveAt(trainList.Count - 1);
                    valList.Add(moved);
                    valSet.Add(Path.GetFileName(moved));
                }
                // 边界：val 意外占满全部 —— 移一张回 train
                if (trainList.Count == 0 && valList.Count > 1)
                {
                    var moved = valList[valList.Count - 1];
                    valList.RemoveAt(valList.Count - 1);
                    trainList.Add(moved);
                    valSet.Remove(Path.GetFileName(moved));
                }

                foreach (var img in trainList)
                    CopyFilePair(img, sourceFolder, trainImgDir, trainLblDir);
                foreach (var img in valList)
                    CopyFilePair(img, sourceFolder, valImgDir, valLblDir);
            }

            // 持久化 val 集（只保存当前真正存在于数据集里的文件，把已删除的历史条目清掉）
            SaveValFixedSet(valFixedPath, valSet.Intersect(imageFiles.Select(Path.GetFileName)));

            GenerateDataYaml(datasetRoot, classesFile);

            return datasetRoot;
        }

        private static HashSet<string> LoadValFixedSet(string path)
        {
            var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            if (!File.Exists(path)) return set;
            try
            {
                foreach (var line in File.ReadAllLines(path))
                {
                    var name = (line ?? string.Empty).Trim();
                    if (!string.IsNullOrWhiteSpace(name))
                        set.Add(name);
                }
            }
            catch { }
            return set;
        }

        private static void SaveValFixedSet(string path, IEnumerable<string> names)
        {
            try
            {
                File.WriteAllLines(path, names.OrderBy(x => x, StringComparer.OrdinalIgnoreCase), Encoding.UTF8);
            }
            catch { }
        }

        private static void CopyFilePair(string srcImgPath, string sourceFolder, string targetImgDir, string targetLblDir)
        {
            string fileNameNoExt = Path.GetFileNameWithoutExtension(srcImgPath);
            string srcTxtPath = Path.Combine(sourceFolder, fileNameNoExt + ".txt");

            string destImgPath = Path.Combine(targetImgDir, Path.GetFileName(srcImgPath));
            File.Copy(srcImgPath, destImgPath, true);

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
            
            sb.AppendLine($"path: {datasetRoot.Replace("\\", "/")}");
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
