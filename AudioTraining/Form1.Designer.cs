namespace AudioTraining
{
    partial class Form1
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要修改
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea2 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend2 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series2 = new System.Windows.Forms.DataVisualization.Charting.Series();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabData = new System.Windows.Forms.TabPage();
            this.splitContainerData = new System.Windows.Forms.SplitContainer();
            this.lstImages = new System.Windows.Forms.ListBox();
            this.picPreview = new System.Windows.Forms.PictureBox();
            this.panelDataTop = new System.Windows.Forms.Panel();
            this.btnLabelImg = new System.Windows.Forms.Button();
            this.btnLoadFolder = new System.Windows.Forms.Button();
            this.tabConfig = new System.Windows.Forms.TabPage();
            this.grpTrainParams = new System.Windows.Forms.GroupBox();
            this.btnBrowseDataYaml = new System.Windows.Forms.Button();
            this.txtDataYaml = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.labelPython = new System.Windows.Forms.Label();
            this.btnBrowsePython = new System.Windows.Forms.Button();
            this.txtPythonPath = new System.Windows.Forms.TextBox();
            this.numBatchSize = new System.Windows.Forms.NumericUpDown();
            this.label2 = new System.Windows.Forms.Label();
            this.numEpochs = new System.Windows.Forms.NumericUpDown();
            this.label1 = new System.Windows.Forms.Label();
            this.grpModel = new System.Windows.Forms.GroupBox();
            this.cmbModelSize = new System.Windows.Forms.ComboBox();
            this.label3 = new System.Windows.Forms.Label();
            this.tabTrain = new System.Windows.Forms.TabPage();
            this.tableLayoutPanelTrain = new System.Windows.Forms.TableLayoutPanel();
            this.chartLoss = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.txtConsole = new System.Windows.Forms.RichTextBox();
            this.panelTrainTop = new System.Windows.Forms.Panel();
            this.btnStopTrain = new System.Windows.Forms.Button();
            this.btnStartTrain = new System.Windows.Forms.Button();
            this.tabValid = new System.Windows.Forms.TabPage();
            this.splitContainerValid = new System.Windows.Forms.SplitContainer();
            this.picValidPreview = new System.Windows.Forms.PictureBox();
            this.txtValidResult = new System.Windows.Forms.TextBox();
            this.panelValidTop = new System.Windows.Forms.Panel();
            this.btnTestImage = new System.Windows.Forms.Button();
            this.btnLoadModel = new System.Windows.Forms.Button();
            this.tabControl1.SuspendLayout();
            this.tabData.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainerData)).BeginInit();
            this.splitContainerData.Panel1.SuspendLayout();
            this.splitContainerData.Panel2.SuspendLayout();
            this.splitContainerData.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.picPreview)).BeginInit();
            this.panelDataTop.SuspendLayout();
            this.tabConfig.SuspendLayout();
            this.grpTrainParams.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numBatchSize)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numEpochs)).BeginInit();
            this.grpModel.SuspendLayout();
            this.tabTrain.SuspendLayout();
            this.tableLayoutPanelTrain.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.chartLoss)).BeginInit();
            this.panelTrainTop.SuspendLayout();
            this.tabValid.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainerValid)).BeginInit();
            this.splitContainerValid.Panel1.SuspendLayout();
            this.splitContainerValid.Panel2.SuspendLayout();
            this.splitContainerValid.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.picValidPreview)).BeginInit();
            this.panelValidTop.SuspendLayout();
            this.SuspendLayout();
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabData);
            this.tabControl1.Controls.Add(this.tabConfig);
            this.tabControl1.Controls.Add(this.tabTrain);
            this.tabControl1.Controls.Add(this.tabValid);
            this.tabControl1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tabControl1.Location = new System.Drawing.Point(0, 0);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(984, 661);
            this.tabControl1.TabIndex = 0;
            // 
            // tabData
            // 
            this.tabData.Controls.Add(this.splitContainerData);
            this.tabData.Controls.Add(this.panelDataTop);
            this.tabData.Location = new System.Drawing.Point(4, 25);
            this.tabData.Name = "tabData";
            this.tabData.Padding = new System.Windows.Forms.Padding(3);
            this.tabData.Size = new System.Drawing.Size(976, 632);
            this.tabData.TabIndex = 0;
            this.tabData.Text = "1. 数据管理";
            this.tabData.UseVisualStyleBackColor = true;
            // 
            // splitContainerData
            // 
            this.splitContainerData.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainerData.Location = new System.Drawing.Point(3, 63);
            this.splitContainerData.Name = "splitContainerData";
            // 
            // splitContainerData.Panel1
            // 
            this.splitContainerData.Panel1.Controls.Add(this.lstImages);
            // 
            // splitContainerData.Panel2
            // 
            this.splitContainerData.Panel2.Controls.Add(this.picPreview);
            this.splitContainerData.Size = new System.Drawing.Size(970, 566);
            this.splitContainerData.SplitterDistance = 250;
            this.splitContainerData.TabIndex = 1;
            // 
            // lstImages
            // 
            this.lstImages.Dock = System.Windows.Forms.DockStyle.Fill;
            this.lstImages.FormattingEnabled = true;
            this.lstImages.ItemHeight = 15;
            this.lstImages.Location = new System.Drawing.Point(0, 0);
            this.lstImages.Name = "lstImages";
            this.lstImages.Size = new System.Drawing.Size(250, 566);
            this.lstImages.TabIndex = 0;
            this.lstImages.SelectedIndexChanged += new System.EventHandler(this.lstImages_SelectedIndexChanged);
            // 
            // picPreview
            // 
            this.picPreview.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.picPreview.Dock = System.Windows.Forms.DockStyle.Fill;
            this.picPreview.Location = new System.Drawing.Point(0, 0);
            this.picPreview.Name = "picPreview";
            this.picPreview.Size = new System.Drawing.Size(716, 566);
            this.picPreview.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.picPreview.TabIndex = 0;
            this.picPreview.TabStop = false;
            // 
            // panelDataTop
            // 
            this.panelDataTop.Controls.Add(this.btnLabelImg);
            this.panelDataTop.Controls.Add(this.btnLoadFolder);
            this.panelDataTop.Dock = System.Windows.Forms.DockStyle.Top;
            this.panelDataTop.Location = new System.Drawing.Point(3, 3);
            this.panelDataTop.Name = "panelDataTop";
            this.panelDataTop.Size = new System.Drawing.Size(970, 60);
            this.panelDataTop.TabIndex = 0;
            // 
            // btnLabelImg
            // 
            this.btnLabelImg.Location = new System.Drawing.Point(150, 15);
            this.btnLabelImg.Name = "btnLabelImg";
            this.btnLabelImg.Size = new System.Drawing.Size(120, 30);
            this.btnLabelImg.TabIndex = 1;
            this.btnLabelImg.Text = "启动 LabelImg";
            this.btnLabelImg.UseVisualStyleBackColor = true;
            this.btnLabelImg.Click += new System.EventHandler(this.btnLabelImg_Click);
            // 
            // btnLoadFolder
            // 
            this.btnLoadFolder.Location = new System.Drawing.Point(15, 15);
            this.btnLoadFolder.Name = "btnLoadFolder";
            this.btnLoadFolder.Size = new System.Drawing.Size(120, 30);
            this.btnLoadFolder.TabIndex = 0;
            this.btnLoadFolder.Text = "加载图片目录";
            this.btnLoadFolder.UseVisualStyleBackColor = true;
            this.btnLoadFolder.Click += new System.EventHandler(this.btnLoadFolder_Click);
            // 
            // tabConfig
            // 
            this.tabConfig.Controls.Add(this.grpTrainParams);
            this.tabConfig.Controls.Add(this.grpModel);
            this.tabConfig.Location = new System.Drawing.Point(4, 25);
            this.tabConfig.Name = "tabConfig";
            this.tabConfig.Padding = new System.Windows.Forms.Padding(3);
            this.tabConfig.Size = new System.Drawing.Size(976, 632);
            this.tabConfig.TabIndex = 1;
            this.tabConfig.Text = "2. 模型配置";
            this.tabConfig.UseVisualStyleBackColor = true;
            // 
            // grpTrainParams
            // 
            this.grpTrainParams.Controls.Add(this.btnBrowseDataYaml);
            this.grpTrainParams.Controls.Add(this.txtDataYaml);
            this.grpTrainParams.Controls.Add(this.label4);
            this.grpTrainParams.Controls.Add(this.btnBrowsePython);
            this.grpTrainParams.Controls.Add(this.txtPythonPath);
            this.grpTrainParams.Controls.Add(this.labelPython);
            this.grpTrainParams.Controls.Add(this.numBatchSize);
            this.grpTrainParams.Controls.Add(this.label2);
            this.grpTrainParams.Controls.Add(this.numEpochs);
            this.grpTrainParams.Controls.Add(this.label1);
            this.grpTrainParams.Location = new System.Drawing.Point(20, 120);
            this.grpTrainParams.Name = "grpTrainParams";
            this.grpTrainParams.Size = new System.Drawing.Size(600, 200);
            this.grpTrainParams.TabIndex = 1;
            this.grpTrainParams.TabStop = false;
            this.grpTrainParams.Text = "训练参数";
            // 
            // btnBrowseDataYaml
            // 
            this.btnBrowseDataYaml.Location = new System.Drawing.Point(519, 121);
            this.btnBrowseDataYaml.Name = "btnBrowseDataYaml";
            this.btnBrowseDataYaml.Size = new System.Drawing.Size(75, 23);
            this.btnBrowseDataYaml.TabIndex = 6;
            this.btnBrowseDataYaml.Text = "浏览...";
            this.btnBrowseDataYaml.UseVisualStyleBackColor = true;
            this.btnBrowseDataYaml.Click += new System.EventHandler(this.btnBrowseDataYaml_Click);
            // 
            // txtDataYaml
            // 
            this.txtDataYaml.Location = new System.Drawing.Point(119, 118);
            this.txtDataYaml.Name = "txtDataYaml";
            this.txtDataYaml.Size = new System.Drawing.Size(394, 25);
            this.txtDataYaml.TabIndex = 5;
            this.txtDataYaml.Text = "data.yaml";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(6, 125);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(123, 15);
            this.label4.TabIndex = 4;
            this.label4.Text = "数据配置(Yaml):";
            // 
            // labelPython
            // 
            this.labelPython.AutoSize = true;
            this.labelPython.Location = new System.Drawing.Point(6, 160);
            this.labelPython.Name = "labelPython";
            this.labelPython.Size = new System.Drawing.Size(100, 15);
            this.labelPython.TabIndex = 7;
            this.labelPython.Text = "Python 解释器:";
            // 
            // txtPythonPath
            // 
            this.txtPythonPath.Location = new System.Drawing.Point(119, 157);
            this.txtPythonPath.Name = "txtPythonPath";
            this.txtPythonPath.Size = new System.Drawing.Size(394, 25);
            this.txtPythonPath.TabIndex = 8;
            this.txtPythonPath.Text = "python";
            // 
            // btnBrowsePython
            // 
            this.btnBrowsePython.Location = new System.Drawing.Point(519, 156);
            this.btnBrowsePython.Name = "btnBrowsePython";
            this.btnBrowsePython.Size = new System.Drawing.Size(75, 23);
            this.btnBrowsePython.TabIndex = 9;
            this.btnBrowsePython.Text = "浏览...";
            this.btnBrowsePython.UseVisualStyleBackColor = true;
            this.btnBrowsePython.Click += new System.EventHandler(this.btnBrowsePython_Click);
            // 
            // numBatchSize
            // 
            this.numBatchSize.Location = new System.Drawing.Point(110, 75);
            this.numBatchSize.Maximum = new decimal(new int[] {
            512,
            0,
            0,
            0});
            this.numBatchSize.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numBatchSize.Name = "numBatchSize";
            this.numBatchSize.Size = new System.Drawing.Size(120, 25);
            this.numBatchSize.TabIndex = 3;
            this.numBatchSize.Value = new decimal(new int[] {
            16,
            0,
            0,
            0});
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(20, 77);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(95, 15);
            this.label2.TabIndex = 2;
            this.label2.Text = "Batch Size:";
            // 
            // numEpochs
            // 
            this.numEpochs.Location = new System.Drawing.Point(110, 35);
            this.numEpochs.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numEpochs.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numEpochs.Name = "numEpochs";
            this.numEpochs.Size = new System.Drawing.Size(120, 25);
            this.numEpochs.TabIndex = 1;
            this.numEpochs.Value = new decimal(new int[] {
            100,
            0,
            0,
            0});
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(20, 37);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(63, 15);
            this.label1.TabIndex = 0;
            this.label1.Text = "Epochs:";
            // 
            // grpModel
            // 
            this.grpModel.Controls.Add(this.cmbModelSize);
            this.grpModel.Controls.Add(this.label3);
            this.grpModel.Location = new System.Drawing.Point(20, 20);
            this.grpModel.Name = "grpModel";
            this.grpModel.Size = new System.Drawing.Size(600, 80);
            this.grpModel.TabIndex = 0;
            this.grpModel.TabStop = false;
            this.grpModel.Text = "模型选择";
            // 
            // cmbModelSize
            // 
            this.cmbModelSize.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbModelSize.FormattingEnabled = true;
            this.cmbModelSize.Items.AddRange(new object[] {
            "n (Nano)",
            "s (Small)",
            "m (Medium)"});
            this.cmbModelSize.Location = new System.Drawing.Point(110, 30);
            this.cmbModelSize.Name = "cmbModelSize";
            this.cmbModelSize.Size = new System.Drawing.Size(120, 23);
            this.cmbModelSize.TabIndex = 1;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(20, 33);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(75, 15);
            this.label3.TabIndex = 0;
            this.label3.Text = "模型大小:";
            // 
            // tabTrain
            // 
            this.tabTrain.Controls.Add(this.tableLayoutPanelTrain);
            this.tabTrain.Controls.Add(this.panelTrainTop);
            this.tabTrain.Location = new System.Drawing.Point(4, 25);
            this.tabTrain.Name = "tabTrain";
            this.tabTrain.Size = new System.Drawing.Size(976, 632);
            this.tabTrain.TabIndex = 2;
            this.tabTrain.Text = "3. 训练监控";
            this.tabTrain.UseVisualStyleBackColor = true;
            // 
            // tableLayoutPanelTrain
            // 
            this.tableLayoutPanelTrain.ColumnCount = 1;
            this.tableLayoutPanelTrain.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanelTrain.Controls.Add(this.chartLoss, 0, 0);
            this.tableLayoutPanelTrain.Controls.Add(this.txtConsole, 0, 1);
            this.tableLayoutPanelTrain.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tableLayoutPanelTrain.Location = new System.Drawing.Point(0, 60);
            this.tableLayoutPanelTrain.Name = "tableLayoutPanelTrain";
            this.tableLayoutPanelTrain.RowCount = 2;
            this.tableLayoutPanelTrain.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanelTrain.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 50F));
            this.tableLayoutPanelTrain.Size = new System.Drawing.Size(976, 572);
            this.tableLayoutPanelTrain.TabIndex = 1;
            // 
            // chartLoss
            // 
            chartArea2.Name = "ChartArea1";
            this.chartLoss.ChartAreas.Add(chartArea2);
            this.chartLoss.Dock = System.Windows.Forms.DockStyle.Fill;
            legend2.Name = "Legend1";
            this.chartLoss.Legends.Add(legend2);
            this.chartLoss.Location = new System.Drawing.Point(3, 3);
            this.chartLoss.Name = "chartLoss";
            series2.ChartArea = "ChartArea1";
            series2.ChartType = System.Windows.Forms.DataVisualization.Charting.SeriesChartType.Line;
            series2.Legend = "Legend1";
            series2.Name = "Loss";
            this.chartLoss.Series.Add(series2);
            this.chartLoss.Size = new System.Drawing.Size(970, 280);
            this.chartLoss.TabIndex = 0;
            this.chartLoss.Text = "chart1";
            // 
            // txtConsole
            // 
            this.txtConsole.BackColor = System.Drawing.Color.Black;
            this.txtConsole.Dock = System.Windows.Forms.DockStyle.Fill;
            this.txtConsole.Font = new System.Drawing.Font("Consolas", 10F);
            this.txtConsole.ForeColor = System.Drawing.Color.Lime;
            this.txtConsole.Location = new System.Drawing.Point(3, 289);
            this.txtConsole.Name = "txtConsole";
            this.txtConsole.ReadOnly = true;
            this.txtConsole.Size = new System.Drawing.Size(970, 280);
            this.txtConsole.TabIndex = 1;
            this.txtConsole.Text = "";
            // 
            // panelTrainTop
            // 
            this.panelTrainTop.Controls.Add(this.btnStopTrain);
            this.panelTrainTop.Controls.Add(this.btnStartTrain);
            this.panelTrainTop.Dock = System.Windows.Forms.DockStyle.Top;
            this.panelTrainTop.Location = new System.Drawing.Point(0, 0);
            this.panelTrainTop.Name = "panelTrainTop";
            this.panelTrainTop.Size = new System.Drawing.Size(976, 60);
            this.panelTrainTop.TabIndex = 0;
            // 
            // btnStopTrain
            // 
            this.btnStopTrain.Enabled = false;
            this.btnStopTrain.Location = new System.Drawing.Point(150, 15);
            this.btnStopTrain.Name = "btnStopTrain";
            this.btnStopTrain.Size = new System.Drawing.Size(120, 30);
            this.btnStopTrain.TabIndex = 1;
            this.btnStopTrain.Text = "停止训练";
            this.btnStopTrain.UseVisualStyleBackColor = true;
            this.btnStopTrain.Click += new System.EventHandler(this.btnStopTrain_Click);
            // 
            // btnStartTrain
            // 
            this.btnStartTrain.Location = new System.Drawing.Point(15, 15);
            this.btnStartTrain.Name = "btnStartTrain";
            this.btnStartTrain.Size = new System.Drawing.Size(120, 30);
            this.btnStartTrain.TabIndex = 0;
            this.btnStartTrain.Text = "开始训练";
            this.btnStartTrain.UseVisualStyleBackColor = true;
            this.btnStartTrain.Click += new System.EventHandler(this.btnStartTrain_Click);
            // 
            // tabValid
            // 
            this.tabValid.Controls.Add(this.splitContainerValid);
            this.tabValid.Controls.Add(this.panelValidTop);
            this.tabValid.Location = new System.Drawing.Point(4, 25);
            this.tabValid.Name = "tabValid";
            this.tabValid.Size = new System.Drawing.Size(976, 632);
            this.tabValid.TabIndex = 3;
            this.tabValid.Text = "4. 模型验证";
            this.tabValid.UseVisualStyleBackColor = true;
            // 
            // splitContainerValid
            // 
            this.splitContainerValid.Dock = System.Windows.Forms.DockStyle.Fill;
            this.splitContainerValid.Location = new System.Drawing.Point(0, 60);
            this.splitContainerValid.Name = "splitContainerValid";
            // 
            // splitContainerValid.Panel1
            // 
            this.splitContainerValid.Panel1.Controls.Add(this.picValidPreview);
            // 
            // splitContainerValid.Panel2
            // 
            this.splitContainerValid.Panel2.Controls.Add(this.txtValidResult);
            this.splitContainerValid.Size = new System.Drawing.Size(976, 572);
            this.splitContainerValid.SplitterDistance = 700;
            this.splitContainerValid.TabIndex = 1;
            // 
            // picValidPreview
            // 
            this.picValidPreview.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.picValidPreview.Dock = System.Windows.Forms.DockStyle.Fill;
            this.picValidPreview.Location = new System.Drawing.Point(0, 0);
            this.picValidPreview.Name = "picValidPreview";
            this.picValidPreview.Size = new System.Drawing.Size(700, 572);
            this.picValidPreview.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.picValidPreview.TabIndex = 0;
            this.picValidPreview.TabStop = false;
            // 
            // txtValidResult
            // 
            this.txtValidResult.Dock = System.Windows.Forms.DockStyle.Fill;
            this.txtValidResult.Location = new System.Drawing.Point(0, 0);
            this.txtValidResult.Multiline = true;
            this.txtValidResult.Name = "txtValidResult";
            this.txtValidResult.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.txtValidResult.Size = new System.Drawing.Size(272, 572);
            this.txtValidResult.TabIndex = 0;
            // 
            // panelValidTop
            // 
            this.panelValidTop.Controls.Add(this.btnTestImage);
            this.panelValidTop.Controls.Add(this.btnLoadModel);
            this.panelValidTop.Dock = System.Windows.Forms.DockStyle.Top;
            this.panelValidTop.Location = new System.Drawing.Point(0, 0);
            this.panelValidTop.Name = "panelValidTop";
            this.panelValidTop.Size = new System.Drawing.Size(976, 60);
            this.panelValidTop.TabIndex = 0;
            // 
            // btnTestImage
            // 
            this.btnTestImage.Enabled = false;
            this.btnTestImage.Location = new System.Drawing.Point(150, 15);
            this.btnTestImage.Name = "btnTestImage";
            this.btnTestImage.Size = new System.Drawing.Size(120, 30);
            this.btnTestImage.TabIndex = 1;
            this.btnTestImage.Text = "加载测试图片";
            this.btnTestImage.UseVisualStyleBackColor = true;
            this.btnTestImage.Click += new System.EventHandler(this.btnTestImage_Click);
            // 
            // btnLoadModel
            // 
            this.btnLoadModel.Location = new System.Drawing.Point(15, 15);
            this.btnLoadModel.Name = "btnLoadModel";
            this.btnLoadModel.Size = new System.Drawing.Size(120, 30);
            this.btnLoadModel.TabIndex = 0;
            this.btnLoadModel.Text = "加载ONNX模型";
            this.btnLoadModel.UseVisualStyleBackColor = true;
            this.btnLoadModel.Click += new System.EventHandler(this.btnLoadModel_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(984, 661);
            this.Controls.Add(this.tabControl1);
            this.Name = "Form1";
            this.Text = "Audio Training Toolbox";
            this.tabControl1.ResumeLayout(false);
            this.tabData.ResumeLayout(false);
            this.splitContainerData.Panel1.ResumeLayout(false);
            this.splitContainerData.Panel2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.splitContainerData)).EndInit();
            this.splitContainerData.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.picPreview)).EndInit();
            this.panelDataTop.ResumeLayout(false);
            this.tabConfig.ResumeLayout(false);
            this.grpTrainParams.ResumeLayout(false);
            this.grpTrainParams.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.numBatchSize)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numEpochs)).EndInit();
            this.grpModel.ResumeLayout(false);
            this.grpModel.PerformLayout();
            this.tabTrain.ResumeLayout(false);
            this.tableLayoutPanelTrain.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.chartLoss)).EndInit();
            this.panelTrainTop.ResumeLayout(false);
            this.tabValid.ResumeLayout(false);
            this.splitContainerValid.Panel1.ResumeLayout(false);
            this.splitContainerValid.Panel2.ResumeLayout(false);
            this.splitContainerValid.Panel2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.splitContainerValid)).EndInit();
            this.splitContainerValid.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.picValidPreview)).EndInit();
            this.panelValidTop.ResumeLayout(false);
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage tabData;
        private System.Windows.Forms.TabPage tabConfig;
        private System.Windows.Forms.TabPage tabTrain;
        private System.Windows.Forms.TabPage tabValid;
        private System.Windows.Forms.SplitContainer splitContainerData;
        private System.Windows.Forms.ListBox lstImages;
        private System.Windows.Forms.PictureBox picPreview;
        private System.Windows.Forms.Panel panelDataTop;
        private System.Windows.Forms.Button btnLabelImg;
        private System.Windows.Forms.Button btnLoadFolder;
        private System.Windows.Forms.GroupBox grpTrainParams;
        private System.Windows.Forms.NumericUpDown numBatchSize;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.NumericUpDown numEpochs;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.GroupBox grpModel;
        private System.Windows.Forms.ComboBox cmbModelSize;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanelTrain;
        private System.Windows.Forms.DataVisualization.Charting.Chart chartLoss;
        private System.Windows.Forms.RichTextBox txtConsole;
        private System.Windows.Forms.Panel panelTrainTop;
        private System.Windows.Forms.Button btnStopTrain;
        private System.Windows.Forms.Button btnStartTrain;
        private System.Windows.Forms.SplitContainer splitContainerValid;
        private System.Windows.Forms.PictureBox picValidPreview;
        private System.Windows.Forms.TextBox txtValidResult;
        private System.Windows.Forms.Panel panelValidTop;
        private System.Windows.Forms.Button btnTestImage;
        private System.Windows.Forms.Button btnLoadModel;
        private System.Windows.Forms.Button btnBrowseDataYaml;
        private System.Windows.Forms.TextBox txtDataYaml;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label labelPython;
        private System.Windows.Forms.TextBox txtPythonPath;
        private System.Windows.Forms.Button btnBrowsePython;
    }
}
