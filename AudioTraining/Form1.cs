using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace AudioTraining
{
    public partial class Form1 : Form
    {
        private string _currentImageFolder;
        private List<string> _imageFiles;
        private TrainingProcess _trainingProcess;
        private YoloInference _yoloInference;
        private Timer _monitorTimer;
        private string _currentTrainProject;
        private string _currentTrainName = "exp";
        private string _loadedOnnxPath;

        public Form1()
        {
            InitializeComponent();
            InitializeCustom();
        }

        private void InitializeCustom()
        {
            cmbModelSize.SelectedIndex = 0; // Default to 'n'
            
            _trainingProcess = new TrainingProcess();
            _trainingProcess.OutputReceived += OnTrainingOutputReceived;
            _trainingProcess.TrainingCompleted += OnTrainingCompleted;

            _yoloInference = new YoloInference();

            _monitorTimer = new Timer();
            _monitorTimer.Interval = 2000; // Check every 2 seconds
            _monitorTimer.Tick += MonitorTimer_Tick;

            // Setup Chart
            chartLoss.Series.Clear();
            chartLoss.Series.Add("BoxLoss");
            chartLoss.Series.Add("ClsLoss");
            chartLoss.Series.Add("DflLoss");
            chartLoss.Series["BoxLoss"].ChartType = SeriesChartType.Line;
            chartLoss.Series["ClsLoss"].ChartType = SeriesChartType.Line;
            chartLoss.Series["DflLoss"].ChartType = SeriesChartType.Line;
            
            chartLoss.ChartAreas[0].AxisX.Title = "Epochs";
            chartLoss.ChartAreas[0].AxisY.Title = "Loss";
        }

        // ==================== 1. Data Management ====================

        private void btnLoadFolder_Click(object sender, EventArgs e)
        {
            using (var fbd = new FolderBrowserDialog())
            {
                if (fbd.ShowDialog() == DialogResult.OK)
                {
                    LoadImagesFromFolder(fbd.SelectedPath);
                }
            }
        }

        private void LoadImagesFromFolder(string folderPath)
        {
            try
            {
                _currentImageFolder = folderPath;
                var ext = new List<string> { ".jpg", ".jpeg", ".png", ".bmp" };
                _imageFiles = Directory.GetFiles(folderPath, "*.*")
                                     .Where(s => ext.Contains(Path.GetExtension(s).ToLower()))
                                     .ToList();

                lstImages.Items.Clear();
                foreach (var file in _imageFiles)
                {
                    lstImages.Items.Add(Path.GetFileName(file));
                }

                if (lstImages.Items.Count > 0)
                    lstImages.SelectedIndex = 0;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"加载目录失败: {ex.Message}");
            }
        }

        private void lstImages_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (lstImages.SelectedIndex < 0 || _imageFiles == null) return;

            try
            {
                string fileName = lstImages.SelectedItem.ToString();
                string fullPath = Path.Combine(_currentImageFolder, fileName);
                
                if (picPreview.Image != null) picPreview.Image.Dispose();
                
                // Load without locking file
                using (var stream = new FileStream(fullPath, FileMode.Open, FileAccess.Read))
                {
                    picPreview.Image = Image.FromStream(stream);
                }
            }
            catch { }
        }

        private void btnLabelImg_Click(object sender, EventArgs e)
        {
            // Try to launch labelImg
            // Assuming it's installed in python environment or standalone
            try
            {
                Process.Start("labelImg", _currentImageFolder ?? "");
            }
            catch
            {
                MessageBox.Show("无法启动 'labelImg'。请确保已安装并添加到系统环境变量PATH中。\n(pip install labelImg)");
            }
        }

        // ==================== 2. Config & 3. Training ====================

        private void btnBrowseDataYaml_Click(object sender, EventArgs e)
        {
            using (var ofd = new OpenFileDialog())
            {
                ofd.Filter = "YAML files (*.yaml)|*.yaml|All files (*.*)|*.*";
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    txtDataYaml.Text = ofd.FileName;
                }
            }
        }

        private async void btnStartTrain_Click(object sender, EventArgs e)
        {
            string yamlPath = txtDataYaml.Text.Trim();
            if (!File.Exists(yamlPath))
            {
                MessageBox.Show("找不到 data.yaml 文件。");
                return;
            }

            string modelSize = "n"; // default
            if (cmbModelSize.SelectedItem != null)
            {
                // "n (Nano)" -> "n"
                modelSize = cmbModelSize.SelectedItem.ToString().Substring(0, 1);
            }

            int epochs = (int)numEpochs.Value;
            int batch = (int)numBatchSize.Value;

            // Setup output directory
            _currentTrainProject = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "runs", "detect");
            if (!Directory.Exists(_currentTrainProject)) Directory.CreateDirectory(_currentTrainProject);

            // Clear previous charts
            foreach (var s in chartLoss.Series) s.Points.Clear();
            txtConsole.Clear();

            btnStartTrain.Enabled = false;
            btnStopTrain.Enabled = true;
            
            _monitorTimer.Start();

            await _trainingProcess.StartTrainingAsync(yamlPath, modelSize, epochs, batch, _currentTrainProject);
        }

        private void btnStopTrain_Click(object sender, EventArgs e)
        {
            _trainingProcess.Stop();
            btnStopTrain.Enabled = false;
            AppendConsole("正在停止训练...");
        }

        private void OnTrainingOutputReceived(object sender, TrainingEventArgs e)
        {
            if (InvokeRequired)
            {
                Invoke(new Action(() => OnTrainingOutputReceived(sender, e)));
                return;
            }

            AppendConsole(e.Message);
        }

        private void OnTrainingCompleted(object sender, EventArgs e)
        {
            if (InvokeRequired)
            {
                Invoke(new Action(() => OnTrainingCompleted(sender, e)));
                return;
            }

            _monitorTimer.Stop();
            btnStartTrain.Enabled = true;
            btnStopTrain.Enabled = false;
            AppendConsole("=========== 训练结束 ===========");
            
            // Try final CSV read
            MonitorTimer_Tick(null, null);
        }

        private void AppendConsole(string text)
        {
            if (string.IsNullOrEmpty(text)) return;
            txtConsole.AppendText(text + Environment.NewLine);
            txtConsole.ScrollToCaret();
        }

        private void MonitorTimer_Tick(object sender, EventArgs e)
        {
            // Parse results.csv for loss plotting
            // Path: runs/detect/exp/results.csv (or exp2, exp3...)
            // YOLOv8 increments exp folder name automatically. We need to find the latest modified one.

            try
            {
                var runsDir = new DirectoryInfo(_currentTrainProject);
                if (!runsDir.Exists) return;

                var dirs = runsDir.GetDirectories().OrderByDescending(d => d.LastWriteTime).ToList();
                if (dirs.Count == 0) return;

                var latestExp = dirs[0];
                var csvPath = Path.Combine(latestExp.FullName, "results.csv");

                if (File.Exists(csvPath))
                {
                    UpdateChartFromCsv(csvPath);
                }
            }
            catch { }
        }

        private void UpdateChartFromCsv(string csvPath)
        {
            // CSV Format usually:
            // epoch, train/box_loss, train/cls_loss, train/dfl_loss, metrics/..., val/..., ...
            
            try
            {
                using (var fs = new FileStream(csvPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (var sr = new StreamReader(fs))
                {
                    string headerLine = sr.ReadLine();
                    if (headerLine == null) return;
                    
                    var headers = headerLine.Split(',').Select(h => h.Trim()).ToList();
                    int idxBox = headers.IndexOf("train/box_loss");
                    int idxCls = headers.IndexOf("train/cls_loss");
                    int idxDfl = headers.IndexOf("train/dfl_loss");
                    int idxEpoch = headers.IndexOf("epoch");

                    if (idxBox == -1 || idxCls == -1 || idxDfl == -1 || idxEpoch == -1) return;

                    // Read all lines
                    var lines = sr.ReadToEnd().Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                    
                    // We only add points that are not yet in the chart
                    // But simpler to just reload everything or check count
                    
                    // Optimization: check if we need to update
                    if (chartLoss.Series["BoxLoss"].Points.Count == lines.Length) return;

                    chartLoss.Series["BoxLoss"].Points.Clear();
                    chartLoss.Series["ClsLoss"].Points.Clear();
                    chartLoss.Series["DflLoss"].Points.Clear();

                    foreach (var line in lines)
                    {
                        var parts = line.Split(',').Select(p => p.Trim()).ToArray();
                        if (parts.Length <= Math.Max(idxBox, Math.Max(idxCls, idxDfl))) continue;

                        if (double.TryParse(parts[idxEpoch], out double epoch) &&
                            double.TryParse(parts[idxBox], out double box) &&
                            double.TryParse(parts[idxCls], out double cls) &&
                            double.TryParse(parts[idxDfl], out double dfl))
                        {
                            chartLoss.Series["BoxLoss"].Points.AddXY(epoch, box);
                            chartLoss.Series["ClsLoss"].Points.AddXY(epoch, cls);
                            chartLoss.Series["DflLoss"].Points.AddXY(epoch, dfl);
                        }
                    }
                }
            }
            catch { }
        }

        // ==================== 4. Validation ====================

        private void btnLoadModel_Click(object sender, EventArgs e)
        {
            using (var ofd = new OpenFileDialog())
            {
                ofd.Filter = "ONNX Models (*.onnx)|*.onnx|All files (*.*)|*.*";
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        // Try to find classes.txt or something for labels? 
                        // For now we use generic class indices or try to read data.yaml
                        
                        string[] labels = null;
                        // Attempt to load labels from adjacent data.yaml if exists?
                        // Simple fallback: 0..N
                        
                        _yoloInference.LoadModel(ofd.FileName, labels);
                        _loadedOnnxPath = ofd.FileName;
                        btnTestImage.Enabled = true;
                        MessageBox.Show("模型加载成功！");
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"模型加载失败: {ex.Message}");
                    }
                }
            }
        }

        private void btnTestImage_Click(object sender, EventArgs e)
        {
            using (var ofd = new OpenFileDialog())
            {
                ofd.Filter = "Images (*.jpg;*.png;*.bmp)|*.jpg;*.png;*.bmp";
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    RunValidation(ofd.FileName);
                }
            }
        }

        private void RunValidation(string imagePath)
        {
            if (!_yoloInference.IsModelLoaded) return;

            try
            {
                Bitmap bmp = new Bitmap(imagePath);
                var predictions = _yoloInference.Predict(bmp);

                // Draw result
                Bitmap drawn = new Bitmap(bmp);
                using (var g = Graphics.FromImage(drawn))
                {
                    foreach (var pred in predictions)
                    {
                        float x = pred.Rectangle.X;
                        float y = pred.Rectangle.Y;
                        float w = pred.Rectangle.Width;
                        float h = pred.Rectangle.Height;

                        using (var pen = new Pen(Color.Red, 2))
                        {
                            g.DrawRectangle(pen, x, y, w, h);
                        }

                        string label = $"{pred.Label} {pred.Confidence:F2}";
                        using (var brush = new SolidBrush(Color.Red))
                        using (var font = new Font("Arial", 12))
                        {
                            g.DrawString(label, font, brush, x, y - 20);
                        }
                    }
                }

                picValidPreview.Image?.Dispose();
                picValidPreview.Image = drawn;
                
                // Dispose original
                bmp.Dispose();

                // Text Result
                StringBuilder sb = new StringBuilder();
                sb.AppendLine($"Image: {Path.GetFileName(imagePath)}");
                sb.AppendLine($"Count: {predictions.Count}");
                foreach (var p in predictions)
                {
                    sb.AppendLine($"Class: {p.ClassId}, Conf: {p.Confidence:F2}, Rect: {p.Rectangle}");
                }
                txtValidResult.Text = sb.ToString();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"推理失败: {ex.Message}");
            }
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);
            _yoloInference?.Dispose();
            _trainingProcess?.Stop();
        }
    }
}
