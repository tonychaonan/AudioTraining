using System;
using System.Diagnostics;
using System.IO;
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

        public async Task StartTrainingAsync(string dataYamlPath, string modelSize, int epochs, int batchSize, string projectPath)
        {
            string toolRoot = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Tools", "yolo");
            string yoloPath = Path.Combine(toolRoot, "yolo.exe");

            if (!File.Exists(yoloPath))
            {
                OnOutput($"Error: 找不到 yolo.exe，请确认文件是否存在于: {yoloPath}");
                return;
            }

            string modelName = $"yolov8{modelSize}.pt"; // n, s, m
            
            // Construct command
            // yolo detect train data=data.yaml model=yolov8n.pt epochs=100 batch=16 project=runs name=train_exp
            string args = $"detect train data=\"{dataYamlPath}\" model={modelName} epochs={epochs} batch={batchSize} project=\"{projectPath}\" name=exp exist_ok=True";

            _process = new Process();
            _process.StartInfo.FileName = yoloPath;
            _process.StartInfo.Arguments = args;
            _process.StartInfo.WorkingDirectory = toolRoot;
            _process.StartInfo.UseShellExecute = false;
            _process.StartInfo.RedirectStandardOutput = true;
            _process.StartInfo.RedirectStandardError = true;
            _process.StartInfo.CreateNoWindow = true;

            _process.OutputDataReceived += (s, e) => ParseOutput(e.Data);
            _process.ErrorDataReceived += (s, e) => ParseOutput(e.Data);

            _process.EnableRaisingEvents = true;
            _process.Exited += (s, e) => 
            {
                OnOutput("Training process exited.");
                ExportToOnnx(projectPath, modelSize);
                TrainingCompleted?.Invoke(this, EventArgs.Empty);
            };

            try
            {
                _process.Start();
                _process.BeginOutputReadLine();
                _process.BeginErrorReadLine();
                
                await Task.Run(() => _process.WaitForExit());
            }
            catch (Exception ex)
            {
                OnOutput($"Error starting training: {ex.Message}");
            }
        }

        private void ExportToOnnx(string projectPath, string modelSize)
        {
            try
            {
                string toolRoot = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Tools", "yolo");
                string yoloPath = Path.Combine(toolRoot, "yolo.exe");

                if (!File.Exists(yoloPath))
                {
                    OnOutput($"Error: 找不到 yolo.exe，无法导出模型。");
                    return;
                }

                // Best model path: projectPath/exp/weights/best.pt
                string bestWeights = Path.Combine(projectPath, "exp", "weights", "best.pt");
                if (File.Exists(bestWeights))
                {
                    OnOutput("Exporting to ONNX...");
                    var exportProcess = new Process();
                    exportProcess.StartInfo.FileName = yoloPath;
                    exportProcess.StartInfo.Arguments = $"export model=\"{bestWeights}\" format=onnx";
                    exportProcess.StartInfo.WorkingDirectory = toolRoot;
                    exportProcess.StartInfo.UseShellExecute = false;
                    exportProcess.StartInfo.RedirectStandardOutput = true;
                    exportProcess.StartInfo.RedirectStandardError = true;
                    exportProcess.StartInfo.CreateNoWindow = true;
                    
                    exportProcess.OutputDataReceived += (s, e) => OnOutput(e.Data);
                    exportProcess.ErrorDataReceived += (s, e) => OnOutput(e.Data);
                    
                    exportProcess.Start();
                    exportProcess.BeginOutputReadLine();
                    exportProcess.BeginErrorReadLine();
                    exportProcess.WaitForExit();
                    
                    OnOutput("Export completed.");
                }
                else
                {
                    OnOutput($"Could not find best.pt at {bestWeights}");
                }
            }
            catch (Exception ex)
            {
                OnOutput($"Export failed: {ex.Message}");
            }
        }

        private void ParseOutput(string line)
        {
            if (string.IsNullOrEmpty(line)) return;

            var args = new TrainingEventArgs { Message = line };

            // Example output to parse (varies by version, roughly):
            // 1/100     2.34G    0.87    1.2    1.1   12  640: 100%|...
            // Or typically stats line
            
            // Simple regex for Epoch parsing "1/100"
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

            // Simple regex for Loss (heuristic, might need adjustment based on specific yolo version output)
            // Look for "box_loss" or just floating point numbers in the progress bar line
            // This is brittle, but better than nothing for a demo.
            // Often: "box_loss" "cls_loss" "dfl_loss" in header, then numbers.
            
            // For now, we just pass the raw line to the UI console, 
            // and maybe try to grab 'box' loss if it's explicitly labeled or structured.
            
            OutputReceived?.Invoke(this, args);
        }

        private void OnOutput(string message)
        {
            OutputReceived?.Invoke(this, new TrainingEventArgs { Message = message });
        }

        public void Stop()
        {
            try
            {
                if (_process != null && !_process.HasExited)
                {
                    _process.Kill();
                }
            }
            catch { }
        }
    }
}
