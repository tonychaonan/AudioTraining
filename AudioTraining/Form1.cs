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
using AudioTraining.Services;
using Newtonsoft.Json.Linq;

namespace AudioTraining
{
    public partial class Form1 : Form
    {
        private string _currentImageFolder;
        private List<string> _imageFiles;
        private TrainingProcess _trainingProcess;
        private YoloInference _yoloInference;
        private YoloOBBInference _yoloOBBInference;
        private Timer _monitorTimer;
        private string _currentTrainProject;
        private string _currentTrainName = "exp";
        private string _loadedOnnxPath;
        private bool _useOBBModel = false;
        private readonly string[] _imageExtensions = new[] { ".jpg", ".jpeg", ".png", ".bmp" };

        public Form1()
        {
            InitializeComponent();
            InitializeCustom();
        }

        private void InitializeCustom()
        {
            cmbModelSize.SelectedIndex = 0; // Default to 'n'
            cmbModelType.SelectedIndex = 0; // Default to standard detection
            
            _trainingProcess = new TrainingProcess();
            _trainingProcess.OutputReceived += OnTrainingOutputReceived;
            _trainingProcess.TrainingCompleted += OnTrainingCompleted;

            _yoloInference = new YoloInference();
            _yoloOBBInference = new YoloOBBInference();

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
                _imageFiles = Directory.GetFiles(folderPath, "*.*")
                                     .Where(s => _imageExtensions.Contains(Path.GetExtension(s).ToLower()))
                                     .ToList();

                lstImages.Items.Clear();
                foreach (var file in _imageFiles)
                {
                    lstImages.Items.Add(Path.GetFileName(file));
                }

                if (lstImages.Items.Count > 0)
                    lstImages.SelectedIndex = 0;

                UpdateAutoAnnotateStatus();
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
            OpenCurrentFolderInLabelingTool();
        }

        private void btnBatchAutoAnnotate_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrWhiteSpace(_currentImageFolder) || !Directory.Exists(_currentImageFolder))
            {
                MessageBox.Show("请先加载图片目录！");
                return;
            }

            if (string.IsNullOrWhiteSpace(_loadedOnnxPath) || !File.Exists(_loadedOnnxPath))
            {
                MessageBox.Show("请先加载 ONNX 模型！");
                return;
            }

            try
            {
                var classes = BatchAutoAnnotateCurrentFolder();
                UpdateAutoAnnotateStatus();
                MessageBox.Show($"自动标注完成！\n生成类别数: {classes.Count}\n目录: {_currentImageFolder}");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"批量自动标注失败: {ex.Message}");
            }
        }

        private void btnGenerateXAnyLabelingYaml_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrWhiteSpace(_loadedOnnxPath) || !File.Exists(_loadedOnnxPath))
            {
                MessageBox.Show("请先在「4. 模型验证」页面加载 ONNX 模型，再生成配置文件！",
                    "未加载模型", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            string classesFile = string.IsNullOrWhiteSpace(_currentImageFolder)
                ? null
                : Path.Combine(_currentImageFolder, "classes.txt");

            List<string> classList = new List<string>();
            if (classesFile != null && File.Exists(classesFile))
            {
                classList = File.ReadAllLines(classesFile)
                    .Select(l => l.Trim())
                    .Where(l => !string.IsNullOrWhiteSpace(l))
                    .ToList();
            }

            if (classList.Count == 0)
            {
                classList.Add("defect");
                MessageBox.Show("未找到 classes.txt，将使用默认类别名称 \"defect\"。\n生成后请手动编辑 YAML 文件中的 classes 列表。",
                    "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }

            string modelType = _useOBBModel ? "yolov8_obb" : "yolov8";
            string modelName = Path.GetFileNameWithoutExtension(_loadedOnnxPath);
            string onnxAbsPath = Path.GetFullPath(_loadedOnnxPath).Replace("\\", "/");

            var sb = new StringBuilder();
            sb.AppendLine($"type: {modelType}");
            sb.AppendLine($"name: {modelName}");
            sb.AppendLine($"display_name: {modelName}");
            sb.AppendLine($"model_path: {onnxAbsPath}");
            sb.AppendLine("input_width: 640");
            sb.AppendLine("input_height: 640");
            sb.AppendLine("confidence_threshold: 0.25");
            sb.AppendLine("nms_threshold: 0.45");
            sb.AppendLine("classes:");
            foreach (var cls in classList)
                sb.AppendLine($"  - {cls}");

            string defaultYamlPath = Path.Combine(
                Path.GetDirectoryName(_loadedOnnxPath),
                modelName + "_xanylabeling.yaml");

            using (var sfd = new SaveFileDialog())
            {
                sfd.Title = "保存 X-AnyLabeling 模型配置文件";
                sfd.Filter = "YAML 配置文件 (*.yaml)|*.yaml";
                sfd.FileName = Path.GetFileName(defaultYamlPath);
                sfd.InitialDirectory = Path.GetDirectoryName(defaultYamlPath);

                if (sfd.ShowDialog() == DialogResult.OK)
                {
                    File.WriteAllText(sfd.FileName, sb.ToString(), Encoding.UTF8);
                    MessageBox.Show(
                        $"配置文件已生成：\n{sfd.FileName}\n\n" +
                        $"使用方法：\n" +
                        $"1. 在 X-AnyLabeling 中点击左侧「AI」按钮\n" +
                        $"2. 选择「加载自定义模型」\n" +
                        $"3. 选择此 YAML 文件即可自动标注",
                        "生成成功", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
        }

        private void btnReviewCurrentFolder_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrWhiteSpace(_currentImageFolder) || !Directory.Exists(_currentImageFolder))
            {
                MessageBox.Show("请先加载图片目录！");
                return;
            }

            OpenCurrentFolderInLabelingTool();
        }

        private void OpenCurrentFolderInLabelingTool()
        {
            if (string.IsNullOrEmpty(_currentImageFolder))
            {
                MessageBox.Show("请先加载图片目录！");
                return;
            }

            try
            {
                string toolRoot = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Tools", "X-AnyLabeling");
                string exePath = Path.Combine(toolRoot, "X-AnyLabeling-CPU.exe");

                if (File.Exists(exePath))
                {
                    // 1. 去除路径末尾的反斜杠，防止转义双引号
                    // 例如把 "D:\Images\" 变成 "D:\Images"
                    string cleanPath = _currentImageFolder.TrimEnd('\\', '/');

                    // 2. 再加上引号，处理路径中可能包含的空格
                    string args = $"\"{cleanPath}\"";

                    ProcessStartInfo psi = new ProcessStartInfo(exePath)
                    {
                        //Arguments = args,
                        WorkingDirectory = toolRoot,
                        UseShellExecute = false
                    };

                    try
                    {
                        Process.Start(psi);
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"启动失败: {ex.Message}");
                    }
                }
                else
                {
                    MessageBox.Show(
                        $"找不到 X-AnyLabeling.exe\n\n" +
                        $"预期路径: {exePath}\n\n" +
                        $"请下载 X-AnyLabeling 并放置到 Tools 文件夹中。\n" +
                        $"下载地址: https://github.com/CVHub520/X-AnyLabeling",
                        "X-AnyLabeling 未找到",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Information
                    );
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"无法启动 X-AnyLabeling: {ex.Message}");
            }
        }

        private List<string> BatchAutoAnnotateCurrentFolder()
        {
            var imageFiles = Directory.GetFiles(_currentImageFolder, "*.*")
                .Where(s => _imageExtensions.Contains(Path.GetExtension(s).ToLower()))
                .OrderBy(s => s)
                .ToList();

            if (imageFiles.Count == 0)
            {
                throw new InvalidOperationException("当前目录下没有图片文件。");
            }

            var classNameById = new Dictionary<int, string>();

            foreach (var imagePath in imageFiles)
            {
                using (var bitmap = new Bitmap(imagePath))
                {
                    var predictions = RunAutoAnnotationInference(bitmap);
                    string txtPath = Path.ChangeExtension(imagePath, ".txt");
                    WritePredictionLabels(txtPath, bitmap.Width, bitmap.Height, predictions);

                    foreach (var prediction in predictions)
                    {
                        if (!classNameById.ContainsKey(prediction.ClassId))
                        {
                            classNameById[prediction.ClassId] = string.IsNullOrWhiteSpace(prediction.Label)
                                ? prediction.ClassId.ToString()
                                : prediction.Label;
                        }
                    }
                }
            }

            var classes = BuildClassList(classNameById);
            File.WriteAllLines(Path.Combine(_currentImageFolder, "classes.txt"), classes, Encoding.UTF8);
            return classes;
        }

        private List<YoloOBBPrediction> RunAutoAnnotationInference(Bitmap bitmap)
        {
            if (_useOBBModel)
            {
                if (!_yoloOBBInference.IsModelLoaded)
                {
                    throw new InvalidOperationException("OBB 模型未加载。");
                }

                return _yoloOBBInference.Predict(bitmap, 0.25f);
            }

            if (!_yoloInference.IsModelLoaded)
            {
                throw new InvalidOperationException("标准检测模型未加载。");
            }

            var stdPredictions = _yoloInference.Predict(bitmap, 0.25f);
            return stdPredictions.Select(pred => new YoloOBBPrediction
            {
                ClassId = pred.ClassId,
                Label = pred.Label,
                Confidence = pred.Confidence,
                Angle = 0,
                RotatedBox = new PointF[4]
                {
                    new PointF(pred.Rectangle.Left, pred.Rectangle.Top),
                    new PointF(pred.Rectangle.Right, pred.Rectangle.Top),
                    new PointF(pred.Rectangle.Right, pred.Rectangle.Bottom),
                    new PointF(pred.Rectangle.Left, pred.Rectangle.Bottom)
                }
            }).ToList();
        }

        private void WritePredictionLabels(string txtPath, int imageWidth, int imageHeight, List<YoloOBBPrediction> predictions)
        {
            var lines = new List<string>();

            foreach (var prediction in predictions)
            {
                if (prediction?.RotatedBox == null || prediction.RotatedBox.Length < 4)
                {
                    continue;
                }

                if (_useOBBModel)
                {
                    var normalizedCorners = prediction.RotatedBox
                        .Take(4)
                        .SelectMany(pt => new[]
                        {
                            Clamp01(pt.X / imageWidth).ToString("F6"),
                            Clamp01(pt.Y / imageHeight).ToString("F6")
                        });

                    lines.Add($"{prediction.ClassId} {string.Join(" ", normalizedCorners)}");
                }
                else
                {
                    float minX = prediction.RotatedBox.Min(p => p.X);
                    float maxX = prediction.RotatedBox.Max(p => p.X);
                    float minY = prediction.RotatedBox.Min(p => p.Y);
                    float maxY = prediction.RotatedBox.Max(p => p.Y);

                    float centerX = (minX + maxX) / 2f;
                    float centerY = (minY + maxY) / 2f;
                    float width = maxX - minX;
                    float height = maxY - minY;

                    lines.Add($"{prediction.ClassId} {Clamp01(centerX / imageWidth):F6} {Clamp01(centerY / imageHeight):F6} {Clamp01(width / imageWidth):F6} {Clamp01(height / imageHeight):F6}");
                }
            }

            File.WriteAllLines(txtPath, lines, Encoding.UTF8);
        }

        private List<string> BuildClassList(Dictionary<int, string> classNameById)
        {
            if (classNameById.Count == 0)
            {
                return new List<string>();
            }

            int maxClassId = classNameById.Keys.Max();
            var result = new List<string>();
            for (int i = 0; i <= maxClassId; i++)
            {
                result.Add(classNameById.TryGetValue(i, out string className) ? className : i.ToString());
            }

            return result;
        }

        private void UpdateAutoAnnotateStatus()
        {
            if (lblAutoAnnotateStatus == null)
            {
                return;
            }

            if (string.IsNullOrWhiteSpace(_currentImageFolder) || !Directory.Exists(_currentImageFolder))
            {
                lblAutoAnnotateStatus.Text = "自动标注：未加载目录";
                return;
            }

            int imageCount = Directory.GetFiles(_currentImageFolder, "*.*")
                .Count(s => _imageExtensions.Contains(Path.GetExtension(s).ToLower()));
            int labelCount = Directory.GetFiles(_currentImageFolder, "*.txt")
                .Count(s => !string.Equals(Path.GetFileName(s), "classes.txt", StringComparison.OrdinalIgnoreCase));

            lblAutoAnnotateStatus.Text = $"自动标注：图片 {imageCount} / 标签 {labelCount}";
        }

        private double Clamp01(double value)
        {
            if (value < 0) return 0;
            if (value > 1) return 1;
            return value;
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

        private void btnBrowsePython_Click(object sender, EventArgs e)
        {
            using (var ofd = new OpenFileDialog())
            {
                ofd.Filter = "Python Executable (python.exe)|python.exe|All files (*.*)|*.*";
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    txtPythonPath.Text = ofd.FileName;
                }
            }
        }

        private void chkEnableSeed_CheckedChanged(object sender, EventArgs e)
        {
            numSeed.Enabled = chkEnableSeed.Checked;
        }

        private void chkContinueTrain_CheckedChanged(object sender, EventArgs e)
        {
            txtBaseModelPath.Enabled = chkContinueTrain.Checked;
            btnBrowseBaseModel.Enabled = chkContinueTrain.Checked;
        }

        private void btnBrowseBaseModel_Click(object sender, EventArgs e)
        {
            using (var ofd = new OpenFileDialog())
            {
                ofd.Filter = "PyTorch Model (*.pt)|*.pt|All files (*.*)|*.*";
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    txtBaseModelPath.Text = ofd.FileName;
                }
            }
        }

        private async void btnStartTrain_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(_currentImageFolder))
            {
                MessageBox.Show("请先加载图片目录并完成标注！");
                return;
            }

            // Get Python Path
            string pythonPath = txtPythonPath.Text.Trim();
            if (string.IsNullOrEmpty(pythonPath))
            {
                pythonPath = "python"; // default fallback
            }

            // 1. Validate X-AnyLabeling annotations
            try
            {
                string classesFile = Path.Combine(_currentImageFolder, "classes.txt");
                
                // Check if classes.txt exists (required by X-AnyLabeling)
                if (!File.Exists(classesFile))
                {
                    MessageBox.Show($"未找到 classes.txt 文件！\n\n请确保使用 X-AnyLabeling 标注工具，并已导出标注数据。\n路径: {classesFile}");
                    return;
                }

                // Validate that TXT annotation files exist
                var imageFiles = Directory.GetFiles(_currentImageFolder, "*.jpg")
                    .Concat(Directory.GetFiles(_currentImageFolder, "*.png"))
                    .Concat(Directory.GetFiles(_currentImageFolder, "*.bmp"))
                    .ToList();

                if (imageFiles.Count == 0)
                {
                    MessageBox.Show("未找到图片文件！请检查数据文件夹。");
                    return;
                }

                int txtCount = 0;
                foreach (var imgFile in imageFiles)
                {
                    string txtFile = Path.ChangeExtension(imgFile, ".txt");
                    if (File.Exists(txtFile))
                    {
                        txtCount++;
                    }
                }

                AppendConsole($"检测到 classes.txt: {classesFile}");
                AppendConsole($"图片数量: {imageFiles.Count}, 标注文件(TXT): {txtCount}");

                if (txtCount == 0)
                {
                    MessageBox.Show($"未找到任何标注文件(.txt)！\n\n请使用 X-AnyLabeling 标注工具完成标注并导出。");
                    return;
                }

                if (txtCount < imageFiles.Count)
                {
                    var result = MessageBox.Show(
                        $"警告：部分图片缺少标注文件\n\n图片: {imageFiles.Count}\n标注: {txtCount}\n\n是否继续训练？",
                        "标注不完整",
                        MessageBoxButtons.YesNo,
                        MessageBoxIcon.Warning
                    );
                    
                    if (result == DialogResult.No)
                    {
                        return;
                    }
                }

                AppendConsole("X-AnyLabeling 标注数据验证通过，准备训练...");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"标注数据验证失败: {ex.Message}");
                return;
            }

            // 2. Prepare Dataset (Standard Structure & Split)
            string datasetRoot = "";
            string yamlPath = "";
            try
            {
                AppendConsole("正在整理数据集目录结构 (Train/Val Split)...");
                // Use DatasetManager to create Dataset_Prepared folder
                datasetRoot = await Task.Run(() => DatasetManager.PrepareDataset(_currentImageFolder, Path.Combine(_currentImageFolder, "classes.txt")));
                yamlPath = Path.Combine(datasetRoot, "data.yaml");
                AppendConsole($"数据集整理完成: {datasetRoot}");
                AppendConsole($"配置文件生成: {yamlPath}");
                
                // Update UI
                txtDataYaml.Text = yamlPath;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"数据集整理失败: {ex.Message}");
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

            // 【关键】从UI读取模型类型
            _useOBBModel = (cmbModelType.SelectedIndex == 1); // 0=Standard, 1=OBB
            AppendConsole($"模型类型: {(_useOBBModel ? "旋转检测 (OBB)" : "标准检测 (Standard)")}");
            LoggerService.Info($"模型类型: {(_useOBBModel ? "旋转检测 (OBB)" : "标准检测 (Standard)")}");

            string baseModelPath = null;
            if (chkContinueTrain.Checked)
            {
                baseModelPath = txtBaseModelPath.Text.Trim();
                if (string.IsNullOrWhiteSpace(baseModelPath))
                {
                    MessageBox.Show("请先选择已有的 .pt 模型文件！");
                    return;
                }

                if (!File.Exists(baseModelPath))
                {
                    MessageBox.Show($"基础模型不存在：\n{baseModelPath}");
                    return;
                }
            }

            // Setup output directory
            _currentTrainProject = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "runs", "detect");
            // Python script handles its own output structure usually, but we pass project path to it or let it use default.
            // In TrainingProcess, we didn't pass projectPath to the python script in the new implementation? 
            // Let's check TrainingProcess. It seems I didn't pass projectPath in the new args.
            // The python script hardcodes 'train_output'.
            // For now, we rely on the python script's internal logic as requested.
            
            // Clear previous charts
            foreach (var s in chartLoss.Series) s.Points.Clear();
            txtConsole.Clear();

            btnStartTrain.Enabled = false;
            btnStopTrain.Enabled = true;
            
            _monitorTimer.Start();

            // Note: TrainingProcess now calls python script. 
            // The python script uses its own output dir 'train_output' in CWD.
            // We might need to point _currentTrainProject to that for the chart to work.
            // The script puts it in os.getcwd()/train_output/current_exp
            // TrainingProcess runs in Scripts folder or BaseDirectory?
            // TrainingProcess sets WorkingDirectory to Path.GetDirectoryName(scriptPath) which is .../Scripts.
            // So output will be .../Scripts/train_output/current_exp.
            // We need to update _currentTrainProject so the chart monitor can find results.csv.
            
            string scriptsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Scripts");
            _currentTrainProject = Path.Combine(scriptsDir, "train_output");

            // Get seed value from UI
            int seed = chkEnableSeed.Checked ? (int)numSeed.Value : 0;
            
            await _trainingProcess.StartTrainingAsync(yamlPath, modelSize, epochs, batch, _currentTrainProject, pythonPath, _useOBBModel, seed, baseModelPath);
        }

        // GenerateDataYaml removed as it is now in DatasetManager
        // private void GenerateDataYaml(string imageFolder, string outputPath) ... 


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

            // Popup Result
            if (_trainingProcess.ExitCode == 0 && !string.IsNullOrEmpty(_trainingProcess.OnnxModelPath))
            {
                MessageBox.Show($"训练成功！\nONNX模型已导出至:\n{_trainingProcess.OnnxModelPath}", 
                                "训练完成", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            else
            {
                string errorInfo = "未知错误";
                if (!string.IsNullOrEmpty(_trainingProcess.ErrorLog))
                {
                    // Limit log length for messagebox
                    errorInfo = _trainingProcess.ErrorLog.Length > 800 
                        ? "..." + _trainingProcess.ErrorLog.Substring(_trainingProcess.ErrorLog.Length - 800) 
                        : _trainingProcess.ErrorLog;
                }
                
                MessageBox.Show($"训练失败 (ExitCode: {_trainingProcess.ExitCode})\n原因:\n{errorInfo}", 
                                "训练出错", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void AppendConsole(string text)
        {
            if (string.IsNullOrEmpty(text)) return;

            // Prevent OutOfMemoryException by limiting log size
            if (txtConsole.TextLength > 30000)
            {
                txtConsole.Clear();
                txtConsole.AppendText("[Log cleared to save memory...]" + Environment.NewLine);
            }

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
            var result = MessageBox.Show(
                "请选择模型类型：\n\n是(Yes) = OBB旋转模型\n否(No) = 标准YOLO模型",
                "选择模型类型",
                MessageBoxButtons.YesNoCancel,
                MessageBoxIcon.Question
            );

            if (result == DialogResult.Cancel)
                return;

            _useOBBModel = (result == DialogResult.Yes);

            using (var ofd = new OpenFileDialog())
            {
                ofd.Filter = "ONNX Models (*.onnx)|*.onnx|All files (*.*)|*.*";
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    try
                    {
                        string[] labels = null;
                        
                        if (_useOBBModel)
                        {
                            _yoloOBBInference.LoadModel(ofd.FileName, labels);
                            MessageBox.Show($"OBB模型加载成功！\n路径: {ofd.FileName}");
                        }
                        else
                        {
                            _yoloInference.LoadModel(ofd.FileName, labels);
                            MessageBox.Show($"标准YOLO模型加载成功！\n路径: {ofd.FileName}");
                        }
                        
                        _loadedOnnxPath = ofd.FileName;
                        btnTestImage.Enabled = true;
                        btnCSharpTest.Enabled = true;
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"模型加载失败: {ex.Message}");
                        LoggerService.Error(ex, "模型加载失败");
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

        private void btnCSharpTest_Click(object sender, EventArgs e)
        {
            if (!_yoloInference.IsModelLoaded && !_yoloOBBInference.IsModelLoaded)
            {
                MessageBox.Show("请先加载模型！");
                return;
            }

            using (OpenFileDialog ofd = new OpenFileDialog())
            {
                ofd.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp";
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    RunCSharpValidation(ofd.FileName);
                }
            }
        }

        private void RunValidation(string imagePath)
        {
            try
            {
                Bitmap bmp = new Bitmap(imagePath);
                Bitmap drawn = new Bitmap(bmp);
                StringBuilder sb = new StringBuilder();
                sb.AppendLine($"Image: {Path.GetFileName(imagePath)}");

                // 使用Python推理（推荐用于训练验证）
                var predictions = PredictWithPython(imagePath, 0.25f, _useOBBModel);

                // 绘制检测结果（统一处理OBB和标准检测）
                using (var g = Graphics.FromImage(drawn))
                {
                    g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                    
                    foreach (var pred in predictions)
                    {
                        // 绘制边界框（多边形，支持旋转框和矩形）
                        using (var pen = new Pen(Color.Red, 2))
                        {
                            g.DrawPolygon(pen, pred.RotatedBox);
                        }

                        // 绘制标签
                        string label = _useOBBModel 
                            ? $"{pred.Label} {pred.Confidence:F2} ({pred.Angle:F1}°)"
                            : $"{pred.Label} {pred.Confidence:F2}";
                        
                        using (var brush = new SolidBrush(Color.Red))
                        using (var font = new Font("Arial", 12))
                        {
                            g.DrawString(label, font, brush, pred.RotatedBox[0].X, pred.RotatedBox[0].Y - 20);
                        }
                    }
                }

                // 显示统计信息
                sb.AppendLine($"推理方式: Python (Ultralytics)");
                sb.AppendLine($"检测数量: {predictions.Count}");
                
                if (predictions.Count > 0)
                {
                    float maxConf = predictions.Max(p => p.Confidence);
                    sb.AppendLine($"最高置信度: {maxConf:F4}");
                }

                foreach (var p in predictions)
                {
                    if (_useOBBModel)
                    {
                        sb.AppendLine($"类别: {p.ClassId}, 置信度: {p.Confidence:F2}, 角度: {p.Angle:F1}°");
                    }
                    else
                    {
                        sb.AppendLine($"类别: {p.ClassId}, 置信度: {p.Confidence:F2}");
                    }
                }

                picValidPreview.Image?.Dispose();
                picValidPreview.Image = drawn;
                bmp.Dispose();
                
                txtValidResult.Text = sb.ToString();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"推理失败: {ex.Message}");
            }
        }

        private void RunCSharpValidation(string imagePath)
        {
            try
            {
                Bitmap bmp = new Bitmap(imagePath);
                Bitmap drawn = new Bitmap(bmp);
                StringBuilder sb = new StringBuilder();
                sb.AppendLine($"Image: {Path.GetFileName(imagePath)}");

                List<YoloOBBPrediction> predictions = new List<YoloOBBPrediction>();

                // 使用C#推理（ONNX Runtime）
                if (_useOBBModel)
                {
                    if (!_yoloOBBInference.IsModelLoaded)
                    {
                        MessageBox.Show("请先加载OBB模型！");
                        return;
                    }
                    predictions = _yoloOBBInference.Predict(bmp, 0.25f);
                }
                else
                {
                    if (!_yoloInference.IsModelLoaded)
                    {
                        MessageBox.Show("请先加载模型！");
                        return;
                    }
                    var stdPredictions = _yoloInference.Predict(bmp, 0.25f);
                    
                    // 转换为统一格式
                    foreach (var pred in stdPredictions)
                    {
                        var corners = new PointF[4]
                        {
                            new PointF(pred.Rectangle.Left, pred.Rectangle.Top),
                            new PointF(pred.Rectangle.Right, pred.Rectangle.Top),
                            new PointF(pred.Rectangle.Right, pred.Rectangle.Bottom),
                            new PointF(pred.Rectangle.Left, pred.Rectangle.Bottom)
                        };
                        
                        predictions.Add(new YoloOBBPrediction
                        {
                            ClassId = pred.ClassId,
                            Label = pred.Label,
                            Confidence = pred.Confidence,
                            RotatedBox = corners,
                            Angle = 0
                        });
                    }
                }

                // 绘制检测结果
                using (var g = Graphics.FromImage(drawn))
                {
                    g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                    
                    foreach (var pred in predictions)
                    {
                        // 绘制边界框（多边形，支持旋转框和矩形）
                        using (var pen = new Pen(Color.Blue, 2))  // 使用蓝色区分C#推理
                        {
                            g.DrawPolygon(pen, pred.RotatedBox);
                        }

                        // 绘制标签
                        string label = _useOBBModel 
                            ? $"{pred.Label} {pred.Confidence:F2} ({pred.Angle:F1}°)"
                            : $"{pred.Label} {pred.Confidence:F2}";
                        
                        using (var brush = new SolidBrush(Color.Blue))
                        using (var font = new Font("Arial", 12))
                        {
                            g.DrawString(label, font, brush, pred.RotatedBox[0].X, pred.RotatedBox[0].Y - 20);
                        }
                    }
                }

                // 显示统计信息
                sb.AppendLine($"推理方式: C# (ONNX Runtime)");
                sb.AppendLine($"检测数量: {predictions.Count}");
                
                if (predictions.Count > 0)
                {
                    float maxConf = predictions.Max(p => p.Confidence);
                    sb.AppendLine($"最高置信度: {maxConf:F4}");
                }

                foreach (var p in predictions)
                {
                    if (_useOBBModel)
                    {
                        sb.AppendLine($"类别: {p.ClassId}, 置信度: {p.Confidence:F2}, 角度: {p.Angle:F1}°");
                    }
                    else
                    {
                        sb.AppendLine($"类别: {p.ClassId}, 置信度: {p.Confidence:F2}");
                    }
                }

                picValidPreview.Image?.Dispose();
                picValidPreview.Image = drawn;
                bmp.Dispose();
                
                txtValidResult.Text = sb.ToString();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"C#推理失败: {ex.Message}\n{ex.StackTrace}");
            }
        }

        // Python推理方法（用于训练验证）
        private List<YoloOBBPrediction> PredictWithPython(string imagePath, float confThreshold, bool isOBB)
        {
            var predictions = new List<YoloOBBPrediction>();
            
            try
            {
                string pythonPath = txtPythonPath.Text;
                if (string.IsNullOrWhiteSpace(pythonPath) || !File.Exists(pythonPath))
                {
                    pythonPath = "python"; // 使用系统Python
                }
                
                string scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Scripts", "inference_wrapper.py");
                string modelPath = _loadedOnnxPath;
                
                // 如果加载的是ONNX，尝试找对应的.pt文件
                if (modelPath.EndsWith(".onnx"))
                {
                    string ptPath = modelPath.Replace(".onnx", ".pt");
                    if (File.Exists(ptPath))
                    {
                        modelPath = ptPath; // 优先使用.pt文件
                    }
                }
                
                string modelType = isOBB ? "obb" : "detect";
                string args = $"\"{scriptPath}\" \"{modelPath}\" \"{imagePath}\" {confThreshold} {modelType}";
                
                var psi = new ProcessStartInfo
                {
                    FileName = pythonPath,
                    Arguments = args,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };
                
                using (var process = Process.Start(psi))
                {
                    string output = process.StandardOutput.ReadToEnd();
                    string error = process.StandardError.ReadToEnd();
                    process.WaitForExit();
                    
                    if (!string.IsNullOrWhiteSpace(error))
                    {
                        LoggerService.Error($"Python推理错误: {error}");
                    }
                    
                    if (string.IsNullOrWhiteSpace(output))
                    {
                        LoggerService.Error("Python推理无输出");
                        return predictions;
                    }
                    
                    // 解析JSON结果
                    var json = JObject.Parse(output);
                    
                    if (json["success"]?.Value<bool>() == false)
                    {
                        string errorMsg = json["error"]?.Value<string>() ?? "Unknown error";
                        LoggerService.Error($"Python推理失败: {errorMsg}");
                        return predictions;
                    }
                    
                    var predArray = json["predictions"] as JArray;
                    if (predArray == null) return predictions;
                    
                    foreach (var pred in predArray)
                    {
                        string type = pred["type"]?.Value<string>();
                        int classId = pred["class_id"]?.Value<int>() ?? 0;
                        float confidence = pred["confidence"]?.Value<float>() ?? 0;

                        /* 后面测试需要，保留这段代码
                        if (type == "obb")
                        {
                            var points = pred["points"]?.Values<float>().ToArray();
                            if (points != null && points.Length == 8)
                            {
                                var corners = new PointF[4];
                                for (int i = 0; i < 4; i++)
                                {
                                    corners[i] = new PointF(points[i * 2], points[i * 2 + 1]);
                                }

                                // 从4个角点计算角度（与标注一致）
                                float angle = CalculateOBBAngleFromCorners(corners);

                                predictions.Add(new YoloOBBPrediction
                                {
                                    ClassId = classId,
                                    Label = classId.ToString(),
                                    Confidence = confidence,
                                    RotatedBox = corners,
                                    Angle = angle
                                });
                            }
                        } */
                        if (type == "obb")
                        {
                            var points = pred["points"]?.Values<float>().ToArray();
                            if (points != null && points.Length == 8)
                            {
                                var corners = new PointF[4];
                                for (int i = 0; i < 4; i++)
                                {
                                    corners[i] = new PointF(points[i * 2], points[i * 2 + 1]);
                                }

                                // 使用Python返回的模型角度（如果有）
                                float angle = pred["angle"]?.Value<float>() ?? 0;

                                predictions.Add(new YoloOBBPrediction
                                {
                                    ClassId = classId,
                                    Label = classId.ToString(),
                                    Confidence = confidence,
                                    RotatedBox = corners,
                                    Angle = angle  // 使用模型输出的角度
                                });
                            }
                        }
                        else if (type == "detect")
                        {
                            // 标准检测: [x1, y1, x2, y2]
                            var box = pred["box"]?.Values<float>().ToArray();
                            if (box != null && box.Length == 4)
                            {
                                // 转换为4个角点（矩形）
                                var corners = new PointF[4]
                                {
                                    new PointF(box[0], box[1]), // 左上
                                    new PointF(box[2], box[1]), // 右上
                                    new PointF(box[2], box[3]), // 右下
                                    new PointF(box[0], box[3])  // 左下
                                };
                                
                                predictions.Add(new YoloOBBPrediction
                                {
                                    ClassId = classId,
                                    Label = classId.ToString(),
                                    Confidence = confidence,
                                    RotatedBox = corners,
                                    Angle = 0
                                });
                            }
                        }
                    }
                    
                    LoggerService.Info($"Python推理完成: 检测到 {predictions.Count} 个目标");
                }
            }
            catch (Exception ex)
            {
                LoggerService.Error($"Python推理异常: {ex.Message}");
            }
            
            return predictions;
        }

        /// <summary>
        /// 从OBB的4个角点计算旋转角度（与标注一致的角度）
        /// </summary>
        /// <param name="corners">4个角点坐标</param>
        /// <returns>旋转角度（度）</returns>
        private float CalculateOBBAngleFromCorners(PointF[] corners)
        {
            if (corners == null || corners.Length != 4)
                return 0;
            // 方法1：找出最左上的角点作为起始点
            // 使用 x + y 最小的点作为"左上角"
            int topLeftIdx = 0;
            float minSum = corners[0].X + corners[0].Y;

            for (int i = 1; i < 4; i++)
            {
                float sum = corners[i].X + corners[i].Y;
                if (sum < minSum)
                {
                    minSum = sum;
                    topLeftIdx = i;
                }
            }

            // 找到距离最左上角点最近的相邻点
            PointF topLeft = corners[topLeftIdx];
            float minDist = float.MaxValue;
            int nextIdx = 0;

            for (int i = 0; i < 4; i++)
            {
                if (i == topLeftIdx) continue;

                float dx = corners[i].X - topLeft.X;
                float dy = corners[i].Y - topLeft.Y;
                float dist = (float)Math.Sqrt(dx * dx + dy * dy);

                if (dist < minDist)
                {
                    minDist = dist;
                    nextIdx = i;
                }
            }

            // 计算从topLeft到next的角度
            float deltaX = corners[nextIdx].X - topLeft.X;
            float deltaY = corners[nextIdx].Y - topLeft.Y;
            float angle = (float)(Math.Atan2(deltaY, deltaX) * 180.0 / Math.PI);

            return angle;
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            base.OnFormClosing(e);
            _yoloInference?.Dispose();
            _trainingProcess?.Stop();
        }
    }
}
