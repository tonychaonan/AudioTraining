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

        public async Task StartTrainingAsync(string dataYamlPath, string modelSize, int epochs, int batchSize, string projectPath, string pythonPath)
        {
            // We need to find the python script
            string scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Scripts", "train_wrapper.py");
            if (!File.Exists(scriptPath))
            {
                OnOutput($"Error: 找不到训练脚本: {scriptPath}");
                return;
            }

            // Use provided python path or default to "python"
            string pythonExe = string.IsNullOrWhiteSpace(pythonPath) ? "python" : pythonPath;

            // Arguments: <yaml_path> <epochs> <img_size> [device]
            int imgSize = 640; 
            string args = $"\"{scriptPath}\" \"{dataYamlPath}\" {epochs} {imgSize}";

            _process = new Process();
            _process.StartInfo.FileName = pythonExe;
            _process.StartInfo.Arguments = args;
            _process.StartInfo.WorkingDirectory = Path.GetDirectoryName(scriptPath); // Run in Scripts folder or BaseDirectory
            _process.StartInfo.UseShellExecute = false;
            _process.StartInfo.RedirectStandardOutput = true;
            _process.StartInfo.RedirectStandardError = true;
            _process.StartInfo.CreateNoWindow = true;

            _process.OutputDataReceived += (s, e) => ParseOutput(e.Data);
            _process.ErrorDataReceived += (s, e) => ParseOutput(e.Data);

            _process.EnableRaisingEvents = true;
            _process.Exited += (s, e) => 
            {
                OnOutput("Python training process exited.");
                TrainingCompleted?.Invoke(this, EventArgs.Empty);
            };

            try
            {
                OnOutput($"Starting Python script: {pythonExe} {args}");
                _process.Start();
                _process.BeginOutputReadLine();
                _process.BeginErrorReadLine();
                
                await Task.Run(() => _process.WaitForExit());
            }
            catch (Exception ex)
            {
                OnOutput($"Error starting python process: {ex.Message}. Make sure 'python' is installed and in PATH.");
            }
        }

        // Removed separate ExportToOnnx as it is handled by the python script now.

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
