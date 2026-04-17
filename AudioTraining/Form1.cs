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
using Newtonsoft.Json;
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
        private string _autoDatasetRootFolder;
        private FileSystemWatcher _autoDatasetWatcher;
        private readonly object _autoDatasetSyncRoot = new object();
        private const double AUTO_ANNOTATE_HIGH_CONFIDENCE = 0.85;
        private const double AUTO_ANNOTATE_REVIEW_CONFIDENCE = 0.55;
        
        // 增量学习相关字段 (需求3)
        private IncrementalClassMapper _incrementalClassMapper;
        private bool _isIncrementalMode = false;
        private ToolTip _trainParamToolTip;

        public Form1()
        {
            InitializeComponent();
            InitializeCustom();
        }

        private void InitializeCustom()
        {
            cmbModelSize.SelectedIndex = 0; 
            cmbModelType.SelectedIndex = 0; 
            
            _trainingProcess = new TrainingProcess();
            _trainingProcess.OutputReceived += OnTrainingOutputReceived;
            _trainingProcess.TrainingCompleted += OnTrainingCompleted;

            _yoloInference = new YoloInference();
            _yoloOBBInference = new YoloOBBInference();
            _incrementalClassMapper = new IncrementalClassMapper();

            _trainParamToolTip = new ToolTip
            {
                AutoPopDelay = 20000,
                InitialDelay = 500,
                ReshowDelay = 100,
                ShowAlways = true,
                ToolTipTitle = "训练调参提示"
            };
            
            // 初始化增量学习UI控件 (需求3)
            InitializeTrainingParameterControls();
            InitializeIncrementalLearningControls();

            _monitorTimer = new Timer();
            _monitorTimer.Interval = 2000;
            _monitorTimer.Tick += MonitorTimer_Tick;

            InitializeClosedLoopBridge();

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
        
        /// <summary>
        /// 初始化增量学习UI控件 (需求3)
        /// </summary>
        private void InitializeTrainingParameterControls()
        {
            ApplyTrainingToolTips();
        }

        private void ApplyTrainingToolTips()
        {
            if (_trainParamToolTip == null) return;

            _trainParamToolTip.SetToolTip(numEpochs, "建议先用 100。400 张样本通常 80~120 够用，太大容易过拟合。");
            _trainParamToolTip.SetToolTip(numBatchSize, "建议 4~8。显存足够可试 8，太大容易不稳定。");
            _trainParamToolTip.SetToolTip(cmbModelSize, "推荐先用 s；n 更快但能力弱，m 更强但更吃显存。");
            _trainParamToolTip.SetToolTip(cmbModelType, "普通外观检测选 Standard；需要旋转框/方向关系选 OBB。");
            _trainParamToolTip.SetToolTip(chkContinueTrain, "必须勾选后才能从旧模型继续训练；增量学习也依赖它。");
            _trainParamToolTip.SetToolTip(numLearningRate, "增量训练建议 0.0005~0.0015。默认 0.001；太大容易冲坏旧模型。") ;
            _trainParamToolTip.SetToolTip(numImgSize, "边缘和细节检测建议 960；显存不够时降到 640。目标越细，越不建议过低分辨率。");
            _trainParamToolTip.SetToolTip(numPatience, "建议 20~40。小样本增量训练常用 30，太大只会浪费时间。");
            _trainParamToolTip.SetToolTip(chkMosaic, "几何关系/细边缘任务建议先关闭；如果泛化不足再开启。") ;
            _trainParamToolTip.SetToolTip(chkMixup, "工业缺陷和细节任务通常关闭；只有外观差异很大时再开启。");
            _trainParamToolTip.SetToolTip(numValSplit, "建议 0.15~0.20。400 张样本至少要留出一部分临界样本做验证。");
            _trainParamToolTip.SetToolTip(numSeed, "开启后训练可重复，便于对比参数。正式上线时可关闭。") ;
            _trainParamToolTip.SetToolTip(txtBaseModelPath, "选择旧模型 .pt。增量训练必须基于可靠的基础模型。");
            _trainParamToolTip.SetToolTip(txtDataYaml, "数据配置文件。训练前会根据验证集比例自动重建。") ;
            _trainParamToolTip.SetToolTip(txtPythonPath, "Python 解释器路径，确保已安装 ultralytics 和依赖。") ;
        }

        private void InitializeIncrementalLearningControls()
        {
            chkEnableIncrementalMode.CheckedChanged += ChkEnableIncrementalMode_CheckedChanged;
            numFreezeLayers.Enabled = false;
        }

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
                _imageFiles = Directory.GetFiles(folderPath, "*.*", SearchOption.AllDirectories)
                                     .Where(s => _imageExtensions.Contains(Path.GetExtension(s).ToLower()))
                                     .Where(s => !s.Contains(Path.DirectorySeparatorChar + "_IncrementalWorkspace" + Path.DirectorySeparatorChar))
                                     .Where(s => !s.Contains(Path.DirectorySeparatorChar + "Dataset_Prepared" + Path.DirectorySeparatorChar))
                                     .ToList();

                lstImages.Items.Clear();
                foreach (var file in _imageFiles)
                {
                    lstImages.Items.Add(GetRelativeDisplayPath(folderPath, file));
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
                string fullPath = _imageFiles[lstImages.SelectedIndex];
                
                if (picPreview.Image != null) picPreview.Image.Dispose();
                
                using (var stream = new FileStream(fullPath, FileMode.Open, FileAccess.Read))
                {
                    picPreview.Image = Image.FromStream(stream);
                }

                var annotationResult = TryAutoAnnotateSingleImage(fullPath, overwriteExisting: false);
                if (annotationResult != null)
                {
                    AppendConsole($"预览自动标注: {Path.GetFileName(fullPath)} | 框数:{annotationResult.PredictionCount} | 最高置信度:{annotationResult.TopConfidence:F3} | 待复核:{(annotationResult.NeedsReview ? "是" : "否")}");
                    UpdateAutoAnnotateStatus();
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
                    // 去除路径末尾的反斜杠，防止转义双引号
                    // 例如把 "D:\Images\" 变成 "D:\Images"
                    string cleanPath = _currentImageFolder.TrimEnd('\\', '/');

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
            var imageFiles = Directory.GetFiles(_currentImageFolder, "*.*", SearchOption.AllDirectories)
                .Where(s => _imageExtensions.Contains(Path.GetExtension(s).ToLower()))
                .Where(s => !s.Contains(Path.DirectorySeparatorChar + "_IncrementalWorkspace" + Path.DirectorySeparatorChar))
                .Where(s => !s.Contains(Path.DirectorySeparatorChar + "Dataset_Prepared" + Path.DirectorySeparatorChar))
                .OrderBy(s => s)
                .ToList();

            if (imageFiles.Count == 0)
            {
                throw new InvalidOperationException("当前目录下没有图片文件。");
            }

            var classNameById = new Dictionary<int, string>();
            var reviewItems = new List<AutoReviewItem>();

            foreach (var imagePath in imageFiles)
            {
                var result = TryAutoAnnotateSingleImage(imagePath, overwriteExisting: true);
                if (result == null)
                {
                    continue;
                }

                foreach (var item in result.Predictions)
                {
                    if (!classNameById.ContainsKey(item.ClassId))
                    {
                        classNameById[item.ClassId] = string.IsNullOrWhiteSpace(item.Label)
                            ? item.ClassId.ToString()
                            : item.Label;
                    }
                }

                if (result.NeedsReview)
                {
                    reviewItems.Add(new AutoReviewItem
                    {
                        ImagePath = imagePath,
                        TopConfidence = result.TopConfidence,
                        PredictionCount = result.PredictionCount,
                        Note = result.ReviewReason
                    });
                }
            }

            var classes = BuildClassList(classNameById);
            File.WriteAllLines(Path.Combine(_currentImageFolder, "classes.txt"), classes, Encoding.UTF8);
            SaveReviewManifest(_currentImageFolder, reviewItems);
            return classes;
        }

        private AutoAnnotationResult TryAutoAnnotateSingleImage(string imagePath, bool overwriteExisting)
        {
            if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
            {
                return null;
            }

            if (string.IsNullOrWhiteSpace(_loadedOnnxPath) || !File.Exists(_loadedOnnxPath))
            {
                return null;
            }

            try
            {
                using (var bitmap = new Bitmap(imagePath))
                {
                    var predictions = RunAutoAnnotationInference(bitmap);
                    string txtPath = Path.ChangeExtension(imagePath, ".txt");
                    if (overwriteExisting || !File.Exists(txtPath))
                    {
                        WritePredictionLabels(txtPath, bitmap.Width, bitmap.Height, predictions);
                    }

                    var topConfidence = predictions.Count > 0 ? predictions.Max(p => (double)p.Confidence) : 0;
                    bool needsReview = predictions.Count == 0 || topConfidence < AUTO_ANNOTATE_REVIEW_CONFIDENCE;
                    string reviewReason = predictions.Count == 0
                        ? "未识别到目标"
                        : (topConfidence < AUTO_ANNOTATE_REVIEW_CONFIDENCE
                            ? $"置信度过低:{topConfidence:F3}"
                            : (topConfidence < AUTO_ANNOTATE_HIGH_CONFIDENCE ? $"建议复核:{topConfidence:F3}" : ""));

                    SaveSingleReviewSidecar(imagePath, predictions, needsReview, reviewReason);

                    return new AutoAnnotationResult
                    {
                        ImagePath = imagePath,
                        Predictions = predictions,
                        TopConfidence = topConfidence,
                        PredictionCount = predictions.Count,
                        NeedsReview = needsReview,
                        ReviewReason = reviewReason
                    };
                }
            }
            catch (Exception ex)
            {
                AppendConsole($"自动标注失败: {Path.GetFileName(imagePath)} | {ex.Message}");
                return null;
            }
        }

        private void SaveSingleReviewSidecar(string imagePath, List<YoloOBBPrediction> predictions, bool needsReview, string reviewReason)
        {
            try
            {
                var review = new AutoReviewItem
                {
                    ImagePath = imagePath,
                    TopConfidence = predictions.Count > 0 ? predictions.Max(p => (double)p.Confidence) : 0,
                    PredictionCount = predictions.Count,
                    NeedsReview = needsReview,
                    Note = reviewReason
                };

                File.WriteAllText(Path.ChangeExtension(imagePath, ".review.json"), JsonConvert.SerializeObject(review, Formatting.Indented), Encoding.UTF8);
            }
            catch (Exception ex)
            {
                AppendConsole($"写入复核侧车失败: {Path.GetFileName(imagePath)} | {ex.Message}");
            }
        }

        private void SaveReviewManifest(string folderPath, List<AutoReviewItem> reviewItems)
        {
            try
            {
                string manifestPath = Path.Combine(folderPath, "review_manifest.json");
                File.WriteAllText(manifestPath, JsonConvert.SerializeObject(reviewItems, Formatting.Indented), Encoding.UTF8);
            }
            catch (Exception ex)
            {
                AppendConsole($"写入复核清单失败: {ex.Message}");
            }
        }

        private void InitializeClosedLoopBridge()
        {
            _autoDatasetRootFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "AutoDataset");
            try
            {
                Directory.CreateDirectory(_autoDatasetRootFolder);
            }
            catch { }

            try
            {
                _autoDatasetWatcher?.Dispose();
                _autoDatasetWatcher = new FileSystemWatcher(_autoDatasetRootFolder)
                {
                    IncludeSubdirectories = true,
                    NotifyFilter = NotifyFilters.DirectoryName | NotifyFilters.FileName | NotifyFilters.LastWrite | NotifyFilters.CreationTime | NotifyFilters.Size,
                    Filter = "*.*",
                    EnableRaisingEvents = true
                };

                _autoDatasetWatcher.Created += AutoDatasetWatcher_Changed;
                _autoDatasetWatcher.Changed += AutoDatasetWatcher_Changed;
                _autoDatasetWatcher.Deleted += AutoDatasetWatcher_Changed;
                _autoDatasetWatcher.Renamed += AutoDatasetWatcher_Renamed;
            }
            catch (Exception ex)
            {
                AppendConsole($"自动采图目录监视器初始化失败: {ex.Message}");
            }
        }

        private void AutoDatasetWatcher_Changed(object sender, FileSystemEventArgs e)
        {
            if (InvokeRequired)
            {
                BeginInvoke(new Action(UpdateAutoAnnotateStatus));
                return;
            }

            UpdateAutoAnnotateStatus();
        }

        private void AutoDatasetWatcher_Renamed(object sender, RenamedEventArgs e)
        {
            AutoDatasetWatcher_Changed(sender, e);
        }

        /// <summary>
        /// 缓存上次构建的 workspace，避免一次"开始训练"里连续调用 3 次造成重复 IO。
        /// 用源目录 + 目录最后修改时间做 key；key 不变就直接返回缓存。
        /// </summary>
        private string _lastWorkspaceSourceKey;
        private string _lastWorkspaceRoot;
        private string _lastWorkspaceClassesFile;

        /// <summary>
        /// workspace 构建结果（替代原来的 out 参数，这样可以直接塞进 Task.Run）
        /// </summary>
        private sealed class WorkspaceBuildResult
        {
            public string WorkspaceRoot;
            public string ClassesFile;
        }

        /// <summary>
        /// ★ 快速路径：用户数据已经是"单层目录"时走这里，完全跳过 flatten/Copy。
        /// 直接返回 sourceFolder 当 workspaceRoot，classes.txt 也读用户目录下的那份。
        /// 对 500 张图来说从 "几十秒 File.Copy" 降到 "几毫秒"。
        /// </summary>
        private WorkspaceBuildResult BuildFastDatasetWorkspace(string sourceFolder)
        {
            if (string.IsNullOrWhiteSpace(sourceFolder) || !Directory.Exists(sourceFolder))
            {
                throw new DirectoryNotFoundException("训练源目录不存在。");
            }

            string classesFile = Path.Combine(sourceFolder, "classes.txt");
            if (!File.Exists(classesFile))
            {
                // 快速模式强依赖用户目录下已存在 classes.txt；不存在给一个清晰的错误让用户去建
                throw new FileNotFoundException(
                    "快速模式要求数据根目录下必须有 classes.txt（每行一个类别名）。\n" +
                    $"请在此处创建：{classesFile}\n" +
                    "或取消勾选\"快速模式\"改走 flatten 模式。", classesFile);
            }

            return new WorkspaceBuildResult
            {
                WorkspaceRoot = sourceFolder,
                ClassesFile = classesFile
            };
        }

        /// <summary>
        /// ★ 跑在后台线程的版本。点击 "开始训练" 时调用方必须用
        /// await Task.Run(() => BuildIncrementalTrainingWorkspace(_currentImageFolder))
        /// 包装，否则 500 张图的 File.Copy + SHA1 会把 UI 线程卡死几秒到几十秒。
        /// </summary>
        private WorkspaceBuildResult BuildIncrementalTrainingWorkspace(string sourceFolder)
        {
            if (string.IsNullOrWhiteSpace(sourceFolder) || !Directory.Exists(sourceFolder))
            {
                throw new DirectoryNotFoundException("训练源目录不存在。");
            }

            // 缓存命中：同一个源目录 + 最近修改时间未变，直接返回，避免一次训练里 3 次 rebuild
            string cacheKey = sourceFolder.TrimEnd('\\', '/') + "|" + Directory.GetLastWriteTimeUtc(sourceFolder).Ticks;
            if (cacheKey == _lastWorkspaceSourceKey
                && !string.IsNullOrEmpty(_lastWorkspaceRoot)
                && Directory.Exists(_lastWorkspaceRoot)
                && File.Exists(_lastWorkspaceClassesFile))
            {
                return new WorkspaceBuildResult
                {
                    WorkspaceRoot = _lastWorkspaceRoot,
                    ClassesFile = _lastWorkspaceClassesFile
                };
            }

            string workspaceRoot = Path.Combine(sourceFolder, "_IncrementalWorkspace");
            if (Directory.Exists(workspaceRoot))
            {
                try { Directory.Delete(workspaceRoot, true); } catch { }
            }

            Directory.CreateDirectory(workspaceRoot);

            // 类别顺序修复（原来的 SortedSet 会按字母排序，会和 base model 的类别 ID 错位）：
            // 1) 根目录 classes.txt 优先，保持它的原顺序
            // 2) 子目录 classes.txt 只用来追加"根目录里没有"的新类别（按发现顺序）
            // 3) 用 List + HashSet 保证去重且顺序稳定
            var classNames = new List<string>();
            var classNameSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            string rootClassesFile = Path.Combine(sourceFolder, "classes.txt");
            if (File.Exists(rootClassesFile))
            {
                try
                {
                    foreach (var line in File.ReadAllLines(rootClassesFile))
                    {
                        string value = line?.Trim();
                        if (!string.IsNullOrWhiteSpace(value) && classNameSet.Add(value))
                        {
                            classNames.Add(value);
                        }
                    }
                }
                catch { }
            }

            foreach (var classFile in Directory.GetFiles(sourceFolder, "classes.txt", SearchOption.AllDirectories))
            {
                // 根目录那份已经读过了
                if (string.Equals(Path.GetFullPath(classFile), Path.GetFullPath(rootClassesFile), StringComparison.OrdinalIgnoreCase))
                    continue;

                // 跳过历史残留：工作区和 Prepared 目录里的 classes.txt 不参与合并
                if (classFile.IndexOf(Path.DirectorySeparatorChar + "_IncrementalWorkspace" + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase) >= 0)
                    continue;
                if (classFile.IndexOf(Path.DirectorySeparatorChar + "Dataset_Prepared" + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase) >= 0)
                    continue;

                try
                {
                    foreach (var line in File.ReadAllLines(classFile))
                    {
                        string value = line?.Trim();
                        if (!string.IsNullOrWhiteSpace(value) && classNameSet.Add(value))
                        {
                            classNames.Add(value);
                        }
                    }
                }
                catch { }
            }

            var imageFiles = Directory.GetFiles(sourceFolder, "*.*", SearchOption.AllDirectories)
                .Where(s => _imageExtensions.Contains(Path.GetExtension(s).ToLower()))
                .Where(s => !s.Contains(Path.DirectorySeparatorChar + "_IncrementalWorkspace" + Path.DirectorySeparatorChar))
                .Where(s => !s.Contains(Path.DirectorySeparatorChar + "Dataset_Prepared" + Path.DirectorySeparatorChar))
                .ToList();

            // 文件名 flatten 冲突修复：相对路径做短 hash，追加到文件名，避免不同目录同名图互相覆盖
            var usedNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var imagePath in imageFiles)
            {
                string labelPath = Path.ChangeExtension(imagePath, ".txt");
                if (!File.Exists(labelPath))
                {
                    continue;
                }

                string relative = imagePath.Substring(sourceFolder.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar).Length)
                    .TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
                string flatBase = relative.Replace(Path.DirectorySeparatorChar, '_').Replace(Path.AltDirectorySeparatorChar, '_');
                string ext = Path.GetExtension(flatBase);
                string nameNoExt = Path.GetFileNameWithoutExtension(flatBase);
                // 用相对路径算短 hash，保证相同原路径哈希稳定；不同路径哈希不同
                string shortHash = ComputeShortPathHash(relative);
                string flatName = $"{nameNoExt}__{shortHash}{ext}";

                // 极端防御：万一 hash 仍然撞车（比如两张图相对路径完全相同，不应该发生），再加计数后缀
                if (!usedNames.Add(flatName))
                {
                    int counter = 1;
                    string probe;
                    do
                    {
                        probe = $"{nameNoExt}__{shortHash}_{counter}{ext}";
                        counter++;
                    } while (!usedNames.Add(probe));
                    flatName = probe;
                }

                string destImgPath = Path.Combine(workspaceRoot, flatName);
                string destLblPath = Path.ChangeExtension(destImgPath, ".txt");
                File.Copy(imagePath, destImgPath, true);
                File.Copy(labelPath, destLblPath, true);
            }

            string classesFile = Path.Combine(workspaceRoot, "classes.txt");
            if (classNames.Count == 0)
            {
                classNames.Add("defect");
            }

            File.WriteAllLines(classesFile, classNames, Encoding.UTF8);

            // 更新缓存
            _lastWorkspaceSourceKey = cacheKey;
            _lastWorkspaceRoot = workspaceRoot;
            _lastWorkspaceClassesFile = classesFile;

            return new WorkspaceBuildResult
            {
                WorkspaceRoot = workspaceRoot,
                ClassesFile = classesFile
            };
        }

        /// <summary>
        /// 相对路径短 hash（8 位 hex），用来区分不同目录下的同名图，避免 flatten 后覆盖
        /// </summary>
        private static string ComputeShortPathHash(string relativePath)
        {
            if (string.IsNullOrEmpty(relativePath)) return "00000000";
            using (var sha = System.Security.Cryptography.SHA1.Create())
            {
                byte[] bytes = sha.ComputeHash(Encoding.UTF8.GetBytes(relativePath.ToLowerInvariant()));
                var sb = new StringBuilder(8);
                for (int i = 0; i < 4; i++) sb.Append(bytes[i].ToString("x2"));
                return sb.ToString();
            }
        }

        private sealed class AutoAnnotationResult
        {
            public string ImagePath { get; set; }
            public List<YoloOBBPrediction> Predictions { get; set; }
            public double TopConfidence { get; set; }
            public int PredictionCount { get; set; }
            public bool NeedsReview { get; set; }
            public string ReviewReason { get; set; }
        }

        private sealed class AutoReviewItem
        {
            public string ImagePath { get; set; }
            public double TopConfidence { get; set; }
            public int PredictionCount { get; set; }
            public bool NeedsReview { get; set; }
            public string Note { get; set; }
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

            int imageCount = Directory.GetFiles(_currentImageFolder, "*.*", SearchOption.AllDirectories)
                .Count(s => _imageExtensions.Contains(Path.GetExtension(s).ToLower()) && !s.Contains(Path.DirectorySeparatorChar + "_IncrementalWorkspace" + Path.DirectorySeparatorChar));
            int labelCount = Directory.GetFiles(_currentImageFolder, "*.txt", SearchOption.AllDirectories)
                .Count(s => !string.Equals(Path.GetFileName(s), "classes.txt", StringComparison.OrdinalIgnoreCase));
            int reviewCount = Directory.GetFiles(_currentImageFolder, "*.review.json", SearchOption.AllDirectories).Length;

            lblAutoAnnotateStatus.Text = $"自动标注：图片 {imageCount} / 标签 {labelCount} / 待复核 {reviewCount}";
        }

        private string GetRelativeDisplayPath(string rootFolder, string fullPath)
        {
            if (string.IsNullOrWhiteSpace(rootFolder) || string.IsNullOrWhiteSpace(fullPath))
            {
                return fullPath;
            }

            string normalizedRoot = rootFolder.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
                + Path.DirectorySeparatorChar;
            if (fullPath.StartsWith(normalizedRoot, StringComparison.OrdinalIgnoreCase))
            {
                return fullPath.Substring(normalizedRoot.Length);
            }

            return Path.GetFileName(fullPath);
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
            
            // 【需求3】仅当继续训练时才能启用增量学习
            chkEnableIncrementalMode.Enabled = chkContinueTrain.Checked;
            if (!chkContinueTrain.Checked)
            {
                chkEnableIncrementalMode.Checked = false;
            }
        }
        
        /// <summary>
        /// 【需求3】增量学习模式复选框事件
        /// </summary>
        private void ChkEnableIncrementalMode_CheckedChanged(object sender, EventArgs e)
        {
            _isIncrementalMode = chkEnableIncrementalMode.Checked;
            numFreezeLayers.Enabled = _isIncrementalMode;
            
            if (_isIncrementalMode)
            {
                MessageBox.Show(
                    "增量学习模式说明：\n\n" +
                    "1. 新数据集的类别列表必须以旧类别开头（顺序一致）\n" +
                    "2. 可以在末尾添加新类别，例如：\n" +
                    "   旧模型: [A, B, C]\n" +
                    "   新数据: [A, B, C, D, E] ✓ 正确\n" +
                    "   新数据: [B, C, D] ✗ 错误（顺序不匹配）\n\n" +
                    "3. 冻结层数建议：\n" +
                    "   - 0层 = 全模型训练（类似普通训练）\n" +
                    "   - 10层 = 冻结前10层（推荐，平衡速度和效果）\n" +
                    "   - -1层 = 冻结整个backbone（仅训练检测头）",
                    "增量学习模式",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information
                );
            }
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

            string pythonPath = txtPythonPath.Text.Trim();
            if (string.IsNullOrEmpty(pythonPath))
            {
                pythonPath = "python"; 
            }

            // ★ 在进入 try 之前把 UI 读起来，后续 Task.Run 里不再访问 UI 控件
            bool fastDatasetMode = chkFastDatasetMode.Checked;

            try
            {
                // ★ 这里是 UI 卡死的元凶：500 张图 × 2 次 File.Copy 全在主线程跑。
                //   改成 Task.Run 到后台，UI 不再冻结。
                AppendConsole(fastDatasetMode
                    ? "[快速模式] 跳过 flatten，直接使用源目录..."
                    : "正在构建训练工作区 (flatten 子目录 + 合并 classes.txt)...");
                btnStartTrain.Enabled = false;
                Application.DoEvents(); // 让 UI 先把 disabled 状态画出来

                var buildResult = await Task.Run(() => fastDatasetMode
                    ? BuildFastDatasetWorkspace(_currentImageFolder)
                    : BuildIncrementalTrainingWorkspace(_currentImageFolder));
                string workspaceRoot = buildResult.WorkspaceRoot;
                string classesFile = buildResult.ClassesFile;

                // Check if classes.txt exists (required by X-AnyLabeling)
                if (!File.Exists(classesFile))
                {
                    MessageBox.Show($"未找到 classes.txt 文件！\n\n请确保使用 X-AnyLabeling 标注工具，并已导出标注数据。\n路径: {classesFile}");
                    btnStartTrain.Enabled = true;
                    return;
                }

                var imageFiles = Directory.GetFiles(workspaceRoot, "*.jpg")
                    .Concat(Directory.GetFiles(workspaceRoot, "*.png"))
                    .Concat(Directory.GetFiles(workspaceRoot, "*.bmp"))
                    .ToList();

                if (imageFiles.Count == 0)
                {
                    MessageBox.Show("未找到图片文件！请检查数据文件夹。");
                    btnStartTrain.Enabled = true;
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
                AppendConsole($"训练工作区: {workspaceRoot}");
                AppendConsole($"图片数量: {imageFiles.Count}, 标注文件(TXT): {txtCount}");

                if (txtCount == 0)
                {
                    MessageBox.Show($"未找到任何标注文件(.txt)！\n\n请使用 X-AnyLabeling 标注工具完成标注并导出。");
                    btnStartTrain.Enabled = true;
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
                        btnStartTrain.Enabled = true;
                        return;
                    }
                }

                AppendConsole("X-AnyLabeling 标注数据验证通过，准备训练...");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"标注数据验证失败: {ex.Message}");
                btnStartTrain.Enabled = true;
                return;
            }

            string datasetRoot = "";
            string yamlPath = "";
            try
            {
                AppendConsole("正在整理数据集目录结构 (Train/Val Split)...");
                // 缓存命中时这次调用是瞬间返回的；第一次没命中则也扔后台避免再卡一次
                var buildResult = await Task.Run(() => fastDatasetMode
                    ? BuildFastDatasetWorkspace(_currentImageFolder)
                    : BuildIncrementalTrainingWorkspace(_currentImageFolder));
                string workspaceRoot = buildResult.WorkspaceRoot;
                string classesFile = buildResult.ClassesFile;
                float valSplit1 = (float)numValSplit.Value;
                // 快速模式下显式指定 Dataset_Prepared 放在用户数据根目录下，
                // 避免默认行为把它写到 sourceFolder 的父目录（导致用户父目录被污染）
                string datasetOutputRoot = fastDatasetMode
                    ? Path.Combine(_currentImageFolder, "Dataset_Prepared")
                    : null;
                datasetRoot = await Task.Run(() => DatasetManager.PrepareDataset(workspaceRoot, classesFile, valSplit1, datasetOutputRoot));
                yamlPath = Path.Combine(datasetRoot, "data.yaml");
                AppendConsole($"数据集整理完成: {datasetRoot}");
                AppendConsole($"配置文件生成: {yamlPath}");
                
                txtDataYaml.Text = yamlPath;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"数据集整理失败: {ex.Message}");
                btnStartTrain.Enabled = true;
                return;
            }

            string modelSize = "n"; 
            if (cmbModelSize.SelectedItem != null)
            {
                modelSize = cmbModelSize.SelectedItem.ToString().Substring(0, 1);
            }

            int epochs = (int)numEpochs.Value;
            int batch = (int)numBatchSize.Value;
            int imgSize = (int)numImgSize.Value;
            double learningRate = (double)numLearningRate.Value;
            int patience = (int)numPatience.Value;
            float valSplit = (float)numValSplit.Value;
            bool useMosaic = chkMosaic.Checked;
            bool useMixup = chkMixup.Checked;

            _useOBBModel = (cmbModelType.SelectedIndex == 1); // 0=Standard, 1=OBB
            AppendConsole($"模型类型: {(_useOBBModel ? "旋转检测 (OBB)" : "标准检测 (Standard)")}");
            LoggerService.Info($"模型类型: {(_useOBBModel ? "旋转检测 (OBB)" : "标准检测 (Standard)")}");
            AppendConsole($"训练调参: lr0={learningRate:F4}, img size={imgSize}, patience={patience}, Mosaic={(useMosaic ? "ON" : "OFF")}, MixUp={(useMixup ? "ON" : "OFF")}, val split={valSplit:F2}");

            string baseModelPath = null;
            if (chkContinueTrain.Checked)
            {
                baseModelPath = txtBaseModelPath.Text.Trim();
                if (string.IsNullOrWhiteSpace(baseModelPath))
                {
                    MessageBox.Show("请先选择已有的 .pt 模型文件！");
                    btnStartTrain.Enabled = true;
                    return;
                }

                if (!File.Exists(baseModelPath))
                {
                    MessageBox.Show($"基础模型不存在：\n{baseModelPath}");
                    btnStartTrain.Enabled = true;
                    return;
                }
                
                // 【需求3】如果启用增量学习模式，验证类别兼容性
                if (_isIncrementalMode)
                {
                    try
                    {
                        // 此处 99% 会命中缓存（上面已经 build 过），瞬间返回；万一没命中也扔后台
                        var buildResult = await Task.Run(() => fastDatasetMode
                            ? BuildFastDatasetWorkspace(_currentImageFolder)
                            : BuildIncrementalTrainingWorkspace(_currentImageFolder));
                        string[] newClasses = _incrementalClassMapper.ReadClassesFromFile(buildResult.ClassesFile);
                        
                        // 注意：无法直接从.pt读取类别，这里简化处理，由Python脚本验证
                        // 实际验证会在train_incremental.py中进行
                        AppendConsole($"增量学习模式已启用，冻结层数: {(int)numFreezeLayers.Value}");
                        AppendConsole($"新数据集类别数: {newClasses.Length}");
                        AppendConsole("类别兼容性将由Python训练脚本验证...");
                    }
                    catch (Exception ex)
                    {
                        MessageBox.Show($"读取类别信息失败: {ex.Message}");
                        btnStartTrain.Enabled = true;
                        return;
                    }
                }
            }

            // Setup output directory
            _currentTrainProject = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "runs", "detect");
            foreach (var s in chartLoss.Series) s.Points.Clear();
            txtConsole.Clear();

            btnStartTrain.Enabled = false;
            btnStopTrain.Enabled = true;
            
            _monitorTimer.Start();
            
            string scriptsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Scripts");
            _currentTrainProject = Path.Combine(scriptsDir, "train_output");

            int seed = chkEnableSeed.Checked ? (int)numSeed.Value : 0;
            int freezeLayers = _isIncrementalMode ? (int)numFreezeLayers.Value : 0;
            
            // 【需求3】根据是否启用增量学习，调用不同的训练方法
            if (_isIncrementalMode && chkContinueTrain.Checked)
            {
                await _trainingProcess.StartIncrementalTrainingAsync(
                    yamlPath, modelSize, epochs, batch, _currentTrainProject, 
                    pythonPath, _useOBBModel, baseModelPath, freezeLayers, seed,
                    imgSize, learningRate, patience, useMosaic, useMixup);
            }
            else
            {
                await _trainingProcess.StartTrainingAsync(
                    yamlPath, modelSize, epochs, batch, _currentTrainProject, 
                    pythonPath, _useOBBModel, seed, baseModelPath,
                    imgSize, learningRate, patience, useMosaic, useMixup);
            }
        }

        private void btnStopTrain_Click(object sender, EventArgs e)
        {
            _trainingProcess.Stop();
            btnStopTrain.Enabled = false;
            AppendConsole("正在停止训练...");
        }

        private void OnTrainingOutputReceived(object sender, TrainingEventArgs e)
        {
            // 用 BeginInvoke：Python 输出频繁时（比如每个 batch 打一行）同步 Invoke 会把后台回调线程卡在 UI 队列上，
            // 反过来拖慢 STDOUT 读取速度，进而导致输出丢帧或进程卡死。BeginInvoke 异步派发不阻塞后台线程。
            if (InvokeRequired)
            {
                try { BeginInvoke(new Action(() => OnTrainingOutputReceived(sender, e))); }
                catch (InvalidOperationException) { /* 窗体正在关闭时忽略 */ }
                //catch (ObjectDisposedException) { /* 窗体已释放 */ }
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
            
            MonitorTimer_Tick(null, null);

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

                    var lines = sr.ReadToEnd().Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);                  
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
