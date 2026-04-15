using AudioTraining.Services;
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace AudioTraining
{
    public class TrainingEventArgs : EventArgs
    {
        public string Message { get; set; }
        public double? BoxLoss { get; set; }
        public double? ClsLoss { get; set; }
        public double? DflLoss { get; set; }
        public int? CurrentEpoch { get; set; }
        public int? TotalEpochs { get; set; }
    }

    public class TrainingProcess
    {
        private Process _process;
        public event EventHandler<TrainingEventArgs> OutputReceived;
        public event EventHandler TrainingCompleted;

        public bool IsRunning => _process != null && !_process.HasExited;
        
        public int ExitCode { get; private set; }
        public string ErrorLog { get; private set; }
        public string OnnxModelPath { get; private set; }
        
        private StringBuilder _errorBuffer;

        public async Task StartTrainingAsync(string dataYamlPath, string modelSize, int epochs, int batchSize, string projectPath, string pythonPath, bool useOBB = false, int seed = 0, string baseModelPath = null)
        {
            _errorBuffer = new StringBuilder();
            ExitCode = 0;
            OnnxModelPath = null;
            ErrorLog = string.Empty;

            string scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Scripts", "train_wrapper.py");
            if (!File.Exists(scriptPath))
            {
                OnOutput($"Error: 找不到训练脚本: {scriptPath}");
                return;
            }

            string pythonExe = string.IsNullOrWhiteSpace(pythonPath) ? "python" : pythonPath;

            // Arguments: <yaml_path> <epochs> <img_size> <model_type> <device> <model_size> <batch_size> [seed] [base_model_path]
            int imgSize = 640;
            string modelType = useOBB ? "obb" : "detect";
            string device = "0";
            
            // Build arguments with required parameters
            string normalizedModelSize = string.IsNullOrWhiteSpace(modelSize) ? "n" : modelSize.Trim().ToLowerInvariant();
            string args = $"\"{scriptPath}\" \"{dataYamlPath}\" {epochs} {imgSize} {modelType} {device} {normalizedModelSize} {batchSize}";
            
            if (seed > 0)
            {
                args += $" {seed}";
                OnOutput($"Training with fixed random seed: {seed} (reproducible mode)");
                LoggerService.Info($"[Training] Random seed enabled: {seed}");
            }
            else
            {
                OnOutput("Training without fixed seed (non-deterministic mode, faster)");
                LoggerService.Info($"[Training] Random seed disabled (non-deterministic mode)");
            }

            if (!string.IsNullOrWhiteSpace(baseModelPath))
            {
                args += $" \"{baseModelPath}\"";
                OnOutput($"Continue training from base model: {baseModelPath}");
                LoggerService.Info($"[Training] Base model path: {baseModelPath}");
            }

            _process = new Process();
            _process.StartInfo.FileName = pythonExe;
            _process.StartInfo.Arguments = args;
            _process.StartInfo.WorkingDirectory = Path.GetDirectoryName(scriptPath); 
            _process.StartInfo.UseShellExecute = false;
            _process.StartInfo.RedirectStandardOutput = true;
            _process.StartInfo.RedirectStandardError = true;
            _process.StartInfo.CreateNoWindow = true;

            _process.OutputDataReceived += (s, e) => ParseOutput(e.Data, false);
            _process.ErrorDataReceived += (s, e) => ParseOutput(e.Data, true);

            _process.EnableRaisingEvents = true;
            _process.Exited += (s, e) => 
            {
                ExitCode = _process.ExitCode;
                ErrorLog = _errorBuffer.ToString();
                OnOutput("Python training process exited.");
                LoggerService.Info($"[Training] Process exited with code: {ExitCode}");
                if (ExitCode != 0)
                {
                    LoggerService.Error($"[Training] Process failed with exit code {ExitCode}. Error log: {ErrorLog}");
                }
                else
                {
                    LoggerService.Info($"[Training] Training completed successfully");
                }
                TrainingCompleted?.Invoke(this, EventArgs.Empty);
            };

            try
            {
                OnOutput($"Starting Python script: {pythonExe} {args}");
                LoggerService.Info($"[Training] Starting Python training process");
                LoggerService.Info($"[Training] Python executable: {pythonExe}");
                LoggerService.Info($"[Training] Arguments: {args}");
                LoggerService.Info($"[Training] Working directory: {_process.StartInfo.WorkingDirectory}");
                
                _process.Start();
                _process.BeginOutputReadLine();
                _process.BeginErrorReadLine();
                
                LoggerService.Info($"[Training] Process started with PID: {_process.Id}");
                await Task.Run(() => _process.WaitForExit());
            }
            catch (Exception ex)
            {
                OnOutput($"Error starting python process: {ex.Message}. Make sure 'python' is installed and in PATH.");
            }
        }

        private void ParseOutput(string line, bool isError)
        {
            if (string.IsNullOrEmpty(line)) return;

            if (isError)
            {
                _errorBuffer.AppendLine(line);
                LoggerService.Warn($"[Training] STDERR: {line}");
            }
            else
            {
                LoggerService.Info($"[Training] STDOUT: {line}");
            }

            if (line.Contains("--- ONNX Path:"))
            {
                var parts = line.Split(new[] { "--- ONNX Path:" }, StringSplitOptions.None);
                if (parts.Length > 1)
                {
                    OnnxModelPath = parts[1].Trim().Trim('-');
                    OnnxModelPath = OnnxModelPath.Trim();
                    LoggerService.Info($"[Training] ONNX model path captured: {OnnxModelPath}");
                }
            }

            var args = new TrainingEventArgs { Message = line };
            var epochMatch = Regex.Match(line, @"(\d+)/(\d+)");
            if (epochMatch.Success)
            {
                if (int.TryParse(epochMatch.Groups[1].Value, out int current) && 
                    int.TryParse(epochMatch.Groups[2].Value, out int total))
                {
                    args.CurrentEpoch = current;
                    args.TotalEpochs = total;
                }
            }
            
            OutputReceived?.Invoke(this, args);
        }

        private void OnOutput(string message)
        {
            LoggerService.Info($"[Training] {message}");
            OutputReceived?.Invoke(this, new TrainingEventArgs { Message = message });
        }

        /// <summary>
        /// 【需求3】启动增量学习训练（支持层冻结和类别扩展）
        /// </summary>
        public async Task StartIncrementalTrainingAsync(string dataYamlPath, string modelSize, int epochs, int batchSize, 
            string projectPath, string pythonPath, bool useOBB, string baseModelPath, int freezeLayers, int seed = 0)
        {
            _errorBuffer = new StringBuilder();
            ExitCode = 0;
            OnnxModelPath = null;
            ErrorLog = string.Empty;

            string scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Scripts", "train_incremental.py");
            if (!File.Exists(scriptPath))
            {
                OnOutput($"Error: 找不到增量训练脚本: {scriptPath}");
                OnOutput("请确保 Scripts/train_incremental.py 文件存在");
                return;
            }

            string pythonExe = string.IsNullOrWhiteSpace(pythonPath) ? "python" : pythonPath;

            // Arguments: <yaml_path> <epochs> <img_size> <model_type> <device> <model_size> <batch_size> <base_model_path> <freeze_layers> [seed]
            int imgSize = 640;
            string modelType = useOBB ? "obb" : "detect";
            string device = "0";
            string normalizedModelSize = string.IsNullOrWhiteSpace(modelSize) ? "n" : modelSize.Trim().ToLowerInvariant();
            
            string args = $"\"{scriptPath}\" \"{dataYamlPath}\" {epochs} {imgSize} {modelType} {device} {normalizedModelSize} {batchSize} \"{baseModelPath}\" {freezeLayers}";
            
            if (seed > 0)
            {
                args += $" {seed}";
                OnOutput($"增量训练（可重复模式）- 随机种子: {seed}");
                LoggerService.Info($"[IncrementalTraining] Random seed enabled: {seed}");
            }
            else
            {
                OnOutput("增量训练（非确定性模式，速度更快）");
                LoggerService.Info($"[IncrementalTraining] Random seed disabled");
            }

            OnOutput($"增量学习模式 - 基础模型: {Path.GetFileName(baseModelPath)}");
            OnOutput($"增量学习模式 - 冻结层数: {freezeLayers}");
            LoggerService.Info($"[IncrementalTraining] Base model: {baseModelPath}");
            LoggerService.Info($"[IncrementalTraining] Freeze layers: {freezeLayers}");

            _process = new Process();
            _process.StartInfo.FileName = pythonExe;
            _process.StartInfo.Arguments = args;
            _process.StartInfo.WorkingDirectory = Path.GetDirectoryName(scriptPath); 
            _process.StartInfo.UseShellExecute = false;
            _process.StartInfo.RedirectStandardOutput = true;
            _process.StartInfo.RedirectStandardError = true;
            _process.StartInfo.CreateNoWindow = true;

            _process.OutputDataReceived += (s, e) => ParseOutput(e.Data, false);
            _process.ErrorDataReceived += (s, e) => ParseOutput(e.Data, true);

            _process.EnableRaisingEvents = true;
            _process.Exited += (s, e) => 
            {
                ExitCode = _process.ExitCode;
                ErrorLog = _errorBuffer.ToString();
                OnOutput("增量训练进程已退出");
                LoggerService.Info($"[IncrementalTraining] Process exited with code: {ExitCode}");
                if (ExitCode != 0)
                {
                    LoggerService.Error($"[IncrementalTraining] Process failed with exit code {ExitCode}. Error log: {ErrorLog}");
                }
                else
                {
                    LoggerService.Info($"[IncrementalTraining] Training completed successfully");
                }
                TrainingCompleted?.Invoke(this, EventArgs.Empty);
            };

            try
            {
                OnOutput($"启动增量训练脚本: {pythonExe} {args}");
                LoggerService.Info($"[IncrementalTraining] Starting Python incremental training process");
                LoggerService.Info($"[IncrementalTraining] Python executable: {pythonExe}");
                LoggerService.Info($"[IncrementalTraining] Arguments: {args}");
                LoggerService.Info($"[IncrementalTraining] Working directory: {_process.StartInfo.WorkingDirectory}");
                
                _process.Start();
                _process.BeginOutputReadLine();
                _process.BeginErrorReadLine();
                
                LoggerService.Info($"[IncrementalTraining] Process started with PID: {_process.Id}");
                await Task.Run(() => _process.WaitForExit());
            }
            catch (Exception ex)
            {
                OnOutput($"Error starting python process: {ex.Message}. Make sure 'python' is installed and in PATH.");
            }
        }

        public void Stop()
        {
            try
            {
                if (_process != null && !_process.HasExited)
                {
                    LoggerService.Info($"[Training] Stopping training process (PID: {_process.Id})");
                    _process.Kill();
                    LoggerService.Info($"[Training] Training process stopped");
                }
            }
            catch (Exception ex)
            {
            }
        }
    }
}
