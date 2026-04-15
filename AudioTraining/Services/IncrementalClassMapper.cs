using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AudioTraining.Services
{
    /// <summary>
    /// 增量学习类别映射管理器
    /// 负责处理旧模型类别和新数据集类别的合并与映射
    /// </summary>
    public class IncrementalClassMapper
    {
        /// <summary>
        /// 合并旧类别和新类别，确保旧类别在前，新类别追加在后
        /// </summary>
        /// <param name="oldClasses">旧模型的类别列表</param>
        /// <param name="newClasses">新数据集的类别列表</param>
        /// <returns>合并后的类别映射 (类别名 -> 新ID)</returns>
        public Dictionary<string, int> MergeClasses(string[] oldClasses, string[] newClasses)
        {
            var merged = new Dictionary<string, int>();
            
            // 首先添加所有旧类别（保持原有ID）
            for (int i = 0; i < oldClasses.Length; i++)
            {
                merged[oldClasses[i]] = i;
            }

            // 添加新类别（分配新ID）
            int nextId = oldClasses.Length;
            foreach (var newClass in newClasses)
            {
                if (!merged.ContainsKey(newClass))
                {
                    merged[newClass] = nextId++;
                }
            }

            return merged;
        }

        /// <summary>
        /// 生成增量训练的classes.txt文件（旧类别在前，新类别在后）
        /// </summary>
        public string[] GenerateIncrementalClassesList(string[] oldClasses, string[] newClasses)
        {
            var result = new List<string>(oldClasses);
            
            foreach (var newClass in newClasses)
            {
                if (!result.Contains(newClass))
                {
                    result.Add(newClass);
                }
            }

            return result.ToArray();
        }

        /// <summary>
        /// 验证新数据集的类别是否兼容增量学习
        /// 增量学习要求：新类别列表必须以旧类别开头（顺序一致），可以在末尾添加新类别
        /// </summary>
        public (bool IsCompatible, string Message) ValidateIncrementalCompatibility(string[] oldClasses, string[] newClasses)
        {
            if (oldClasses == null || oldClasses.Length == 0)
            {
                return (true, "基础模型无类别信息，将作为标准训练");
            }

            if (newClasses == null || newClasses.Length == 0)
            {
                return (false, "新数据集没有类别信息");
            }

            if (newClasses.Length < oldClasses.Length)
            {
                return (false, $"新数据集类别数({newClasses.Length}) < 基础模型类别数({oldClasses.Length})，无法进行增量学习");
            }

            // 检查前N个类别是否完全一致（N = 旧类别数）
            for (int i = 0; i < oldClasses.Length; i++)
            {
                if (newClasses[i] != oldClasses[i])
                {
                    return (false, $"类别顺序不匹配：旧[{i}]={oldClasses[i]}, 新[{i}]={newClasses[i]}");
                }
            }

            // 检查是否有新增类别
            if (newClasses.Length > oldClasses.Length)
            {
                var newAdded = newClasses.Skip(oldClasses.Length).ToArray();
                return (true, $"增量学习：保留{oldClasses.Length}个旧类，新增{newAdded.Length}个类 [{string.Join(", ", newAdded)}]");
            }
            else
            {
                return (true, "纯微调模式（类别数未变）");
            }
        }

        /// <summary>
        /// 从模型文件读取类别列表（需要Python协助）
        /// </summary>
        public string[] ReadClassesFromModel(string modelPath)
        {
            // 这里需要调用Python脚本来读取.pt模型的classes
            // 暂时返回空数组，由调用者处理
            // TODO: 实现Python互操作读取模型类别
            return new string[0];
        }

        /// <summary>
        /// 从classes.txt文件读取类别列表
        /// </summary>
        public string[] ReadClassesFromFile(string classesFilePath)
        {
            if (!File.Exists(classesFilePath))
            {
                return new string[0];
            }

            try
            {
                return File.ReadAllLines(classesFilePath)
                    .Select(line => line.Trim())
                    .Where(line => !string.IsNullOrEmpty(line))
                    .ToArray();
            }
            catch (Exception ex)
            {
                LoggerService.Error(ex, $"读取classes.txt失败: {classesFilePath}");
                return new string[0];
            }
        }

        /// <summary>
        /// 重新映射标注文件中的类别ID（如果新旧类别顺序发生变化）
        /// </summary>
        public void RemapAnnotationFiles(string annotationFolder, Dictionary<int, int> idMapping)
        {
            if (!Directory.Exists(annotationFolder))
            {
                return;
            }

            var txtFiles = Directory.GetFiles(annotationFolder, "*.txt");
            
            foreach (var txtFile in txtFiles)
            {
                try
                {
                    var lines = File.ReadAllLines(txtFile);
                    var remappedLines = new List<string>();

                    foreach (var line in lines)
                    {
                        var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length == 0) continue;

                        if (int.TryParse(parts[0], out int oldClassId))
                        {
                            if (idMapping.TryGetValue(oldClassId, out int newClassId))
                            {
                                parts[0] = newClassId.ToString();
                                remappedLines.Add(string.Join(" ", parts));
                            }
                            else
                            {
                                // 保持原样
                                remappedLines.Add(line);
                            }
                        }
                        else
                        {
                            remappedLines.Add(line);
                        }
                    }

                    File.WriteAllLines(txtFile, remappedLines);
                }
                catch (Exception ex)
                {
                    LoggerService.Error(ex, $"重新映射标注文件失败: {txtFile}");
                }
            }
        }
    }
}
