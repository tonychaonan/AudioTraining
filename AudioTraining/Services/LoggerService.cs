using NLog;
using System;

namespace AudioTraining.Services
{
    /// <summary>
    /// 日志服务 
    /// </summary>
    public class LoggerService
    {
        private static readonly Logger _logger = LogManager.GetCurrentClassLogger();

        /// <summary>
        /// 记录信息日志
        /// </summary>
        public static void Info(string message)
        {
            _logger.Info(message);
        }

        /// <summary>
        /// 记录调试日志
        /// </summary>
        public static void Debug(string message)
        {
            _logger.Debug(message);
        }

        /// <summary>
        /// 记录警告日志
        /// </summary>
        public static void Warn(string message)
        {
            _logger.Warn(message);
        }

        /// <summary>
        /// 记录错误日志
        /// </summary>
        public static void Error(Exception ex, string message)
        {
            _logger.Error(ex, message);
        }

        /// <summary>
        /// 记录错误日志（仅消息）
        /// </summary>
        public static void Error(string message)
        {
            _logger.Error(message);
        }

        /// <summary>
        /// 记录训练开始
        /// </summary>
        public static void LogTrainingStart(string projectName, int epochs, int imageSize)
        {
            Info($"【训练开始】项目: {projectName}, 轮数: {epochs}, 图像尺寸: {imageSize}");
        }

        /// <summary>
        /// 记录训练完成
        /// </summary>
        public static void LogTrainingComplete(string projectName, bool success)
        {
            if (success)
                Info($"【训练完成】项目: {projectName}, 结果: 成功");
            else
                Warn($"【训练完成】项目: {projectName}, 结果: 失败");
        }

        /// <summary>
        /// 记录模型推理
        /// </summary>
        public static void LogInference(string modelType, int detectionCount, float maxConfidence)
        {
            Info($"【模型推理】类型: {modelType}, 检测数量: {detectionCount}, 最高置信度: {maxConfidence:F4}");
        }

        /// <summary>
        /// 记录OBB推理详情
        /// </summary>
        public static void LogOBBInference(int[] dimensions, int numClasses, int numAnchors, float threshold)
        {
            Debug($"【OBB推理】输出维度: [{dimensions[0]}, {dimensions[1]}, {dimensions[2]}], 类别数: {numClasses}, 锚点数: {numAnchors}, 阈值: {threshold:F4}");
        }

        /// <summary>
        /// 记录锚点检测详情
        /// </summary>
        public static void LogAnchorDetection(int anchorIndex, float[] scores, float maxScore, int classId)
        {
            Debug($"【锚点{anchorIndex}】分数: [{string.Join(", ", Array.ConvertAll(scores, s => s.ToString("F4")))}], 最大: {maxScore:F4}, 类别: {classId}");
        }
    }
}
