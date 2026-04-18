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

            // ★ 分层 val split：对每个类别**独立**按 valSplit 比例抽 val，
            // 保证稀有/新类别在 val 集里必有代表样本（修复原"纯随机分导致新类 val=0"的问题）
            var (trainList, valList) = ComputeStratifiedSplit(imageFiles, sourceFolder, valSet, valSplit);

            // 边界：整个数据集只有 1 张图 —— 两边都放一份
            if (imageFiles.Count == 1)
            {
                CopyFilePair(imageFiles[0], sourceFolder, trainImgDir, trainLblDir);
                CopyFilePair(imageFiles[0], sourceFolder, valImgDir, valLblDir);
            }
            else
            {
                // 边界：val 空了（全是全新数据且 stratified 无命中）—— 强制挑一张塞 val
                if (valList.Count == 0 && trainList.Count > 1)
                {
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
            var finalValNames = valList.Select(Path.GetFileName).ToList();
            foreach (var n in finalValNames) valSet.Add(n);
            SaveValFixedSet(valFixedPath, valSet.Intersect(imageFiles.Select(Path.GetFileName)));

            GenerateDataYaml(datasetRoot, classesFile);

            return datasetRoot;
        }

        /// <summary>
        /// 分层 train/val 划分：对每个"主类别"分组后独立按 valSplit 比例抽 val。
        ///
        /// 主类别定义：一张图含多个类别时，选其中"覆盖最少"的类别作为主类——这样稀有类优先占坑，
        /// 保证它们在 val 集里必有代表（普通随机抽样时稀有类可能一张不进 val）。
        ///
        /// val_fixed.txt 里已固定的图**永远**保留在 val 集，跨训练次结果可对比。
        /// </summary>
        private static (List<string> trainList, List<string> valList) ComputeStratifiedSplit(
            List<string> imageFiles, string sourceFolder, HashSet<string> valSet, float valSplit)
        {
            var rng = new Random();

            // 1) 读每张图的类别集合
            var imageToClasses = new Dictionary<string, HashSet<int>>();
            foreach (var img in imageFiles)
            {
                var lbl = Path.Combine(sourceFolder, Path.GetFileNameWithoutExtension(img) + ".txt");
                imageToClasses[img] = ReadClassIdsFromLabel(lbl);
            }

            // 2) 类别覆盖度统计：多少张图含该类
            var classCoverage = new Dictionary<int, int>();
            foreach (var kv in imageToClasses)
                foreach (var c in kv.Value)
                {
                    classCoverage.TryGetValue(c, out int n);
                    classCoverage[c] = n + 1;
                }

            // 3) 每张图选主类：取它含的类别里"覆盖度最低"那个（稀有优先），
            //    没标注的图用 -1 归到独立桶走随机
            var primaryClass = new Dictionary<string, int>();
            foreach (var kv in imageToClasses)
            {
                if (kv.Value.Count == 0)
                {
                    primaryClass[kv.Key] = -1;
                }
                else
                {
                    int rarestClass = kv.Value.OrderBy(c => classCoverage[c]).First();
                    primaryClass[kv.Key] = rarestClass;
                }
            }

            // 4) 按主类分桶，每桶独立分层抽样
            var buckets = primaryClass.GroupBy(kv => kv.Value)
                                      .ToDictionary(g => g.Key, g => g.Select(x => x.Key).ToList());

            var trainList = new List<string>();
            var valList = new List<string>();

            foreach (var bucket in buckets)
            {
                var imgsInBucket = bucket.Value;

                // 已经在 val_fixed 里的强制进 val
                var fixedVal = imgsInBucket.Where(p => valSet.Contains(Path.GetFileName(p))).ToList();
                var free = imgsInBucket.Where(p => !valSet.Contains(Path.GetFileName(p)))
                                        .OrderBy(_ => rng.Next()).ToList();

                // 目标 val 数：至少 1（只要桶里图 ≥ 2），保证稀有类别有 val 代表
                int target = (int)Math.Round(imgsInBucket.Count * valSplit);
                if (imgsInBucket.Count >= 2 && target < 1) target = 1;

                int need = target - fixedVal.Count;
                if (need > 0)
                {
                    var newVal = free.Take(need).ToList();
                    valList.AddRange(fixedVal);
                    valList.AddRange(newVal);
                    trainList.AddRange(free.Skip(need));
                }
                else
                {
                    valList.AddRange(fixedVal);
                    trainList.AddRange(free);
                }
            }

            return (trainList, valList);
        }

        /// <summary>
        /// 从 YOLO 标签文件读取出现过的 class id 集合（每行首 token 是 int class_id）。
        /// 文件不存在或读失败返回空集。
        /// </summary>
        private static HashSet<int> ReadClassIdsFromLabel(string labelPath)
        {
            var set = new HashSet<int>();
            if (!File.Exists(labelPath)) return set;
            try
            {
                foreach (var line in File.ReadAllLines(labelPath))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var first = line.TrimStart().Split(new[] { ' ', '\t' }, 2, StringSplitOptions.RemoveEmptyEntries);
                    if (first.Length == 0) continue;
                    if (int.TryParse(first[0], out int cid)) set.Add(cid);
                }
            }
            catch { }
            return set;
        }

        /// <summary>
        /// 统计数据集的类别分布，供 UI/训练前警告使用。
        /// 返回：(类别ID -> 含该类别的图片数量)
        /// 一张图若含多个类别，会在多个类别上各 +1。
        /// </summary>
        public static Dictionary<int, int> ComputeClassImageCounts(string sourceFolder)
        {
            var result = new Dictionary<int, int>();
            if (string.IsNullOrWhiteSpace(sourceFolder) || !Directory.Exists(sourceFolder))
                return result;

            var ext = new[] { ".jpg", ".jpeg", ".png", ".bmp" };
            foreach (var img in Directory.GetFiles(sourceFolder, "*.*")
                     .Where(s => ext.Contains(Path.GetExtension(s).ToLower())))
            {
                var lbl = Path.Combine(sourceFolder, Path.GetFileNameWithoutExtension(img) + ".txt");
                foreach (var cid in ReadClassIdsFromLabel(lbl))
                {
                    result.TryGetValue(cid, out int n);
                    result[cid] = n + 1;
                }
            }
            return result;
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
