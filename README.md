# 中文段内情感与时长可控语音合成

我使用开源 MeloTTS 中文模型实现了段内韵律情感控制与指定词语、短语时长控制，并提供了对照音频、客观评测、主观试听和结果分析。方法、具体数值和局限见 [实验说明](实验说明.md)。

## 查看现有结果

1. 打开 `results/audio`，可以直接播放已经生成的 40 条 WAV。
2. 打开 `实验说明.md` 查看实验方法和结果分析，`results/figures` 中保存了对应图片。
3. 双击 `results/listening.html`，可以在浏览器中完成三项主观评分。页面支持离线播放，无需启动服务。
4. `results/objective_metrics.csv` 和 `results/segment_metrics.csv` 可用 Excel 打开，文件采用 UTF-8 BOM 编码。

同一组音频以相同编号开头，例如 `s05_whole.wav` 为整句原始合成，`s05_baseline.wav` 为分段中性基线，`s05_emotion.wav` 为情感控制，`s05_duration.wav` 为目标时长控制，`s05_combined.wav` 为联合控制。s05 的目标是“安全第一”，s06 为“马上出发”。

## 运行环境与复现

我在 Windows、Python 3.9.1 环境下完成了本轮实验。建议使用 64 位 Python 3.9–3.11。项目采用 CPU 推理，不需要 CUDA 或服务器。首次安装依赖、缺失模型下载需要联网，模型准备好后，合成、评测和试听均可离线进行。

将整个“hw3”文件夹复制到电脑上，安装好 64 位 Python 3.9–3.11。在“hw3”文件夹打开 PowerShell，执行

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

脚本会在“hw3”文件夹内创建 `.venv` 并安装 `requirements.txt` 中的依赖。后续运行复用该环境。若该环境已有部分依赖，需重新安装或补齐依赖时执行

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Install
```

执行后依次检查模型、生成 40 条音频、验证采样点与时长、运行客观评测、更新图表和说明文档。首次运行音频处理库可能需要额外初始化时间。结果会覆盖同名输出，修改配置前可复制 `results` 留存原始对照。

如果在 Linux 服务器复现，可以使用

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_models.py
python run_experiments.py
python verify_results.py
python evaluate.py
python make_review.py
python build_report.py
```

Linux 命令已按可移植写法提供，本轮实际验证的环境为 Windows。项目保留已下载的模型和来源文件，可复制整个文件夹交接。模型体积占主要部分，下载脚本会跳过已存在的模型。

## 分步使用

以下命令均在“hw3”文件夹内执行。首次运行时先创建环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

环境准备好后，可按顺序执行

```powershell
.\.venv\Scripts\python.exe download_models.py
.\.venv\Scripts\python.exe run_experiments.py
.\.venv\Scripts\python.exe verify_results.py
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe make_review.py
.\.venv\Scripts\python.exe build_report.py
```

## 修改文本和控制参数

测试句保存在 `configs/cases.json`。每个条目包含唯一 `id`、`duration_factor` 和按朗读顺序排列的 `segments`。我通过主动切分文本定位目标词语，重复出现的词语可以按所在片段分别指定。当前默认实验只选择一个目标片段，也支持同句标记多个目标，它们共享该句的时长倍率。

例如，下面的配置在“安全第一”上使用激动韵律，并在时长控制条件下将该短语拉长到 1.5 倍

```json
[
  {
    "id": "demo01",
    "duration_factor": 1.5,
    "segments": [
      {"text": "请你把", "emotion": "neutral"},
      {"text": "安全第一", "emotion": "excited", "target": true},
      {"text": "这句话记在心里。", "emotion": "neutral"}
    ]
  }
]
```

将配置另存为 JSON 后可以执行

```powershell
.\.venv\Scripts\python.exe run_experiments.py --config configs/my_cases.json --threads 4
```

`duration_factor` 的支持范围为 0.5–2.0，大于 1 表示拉长，小于 1 表示缩短。`emotion` 支持 `neutral`、`excited`、`sad`。情感类型对应的半音、基频变化范围、增益和默认时长系数位于 `src/control.py` 的 `PROFILES`。

联合控制会将情感本身的时长系数与显式时长倍率相乘。系统使用固定的声学规则实现情感相关控制。新增情感标签时，需要定义其参数，并用真实听辨或情感分类验证其感知效果。

修改模型、采样噪声、说话人或基础合成参数后，应清理 `results/cache` 再生成，避免复用旧波形。当前噪声设为 0，且缓存键包含当前模型和参数版本。

## 主观试听与真实评分汇总

我把 32 条待评音频随机排序，听众看到匿名编号，三项评分均为 1–5 分。可懂度评价能否听清，自然度评价发音、节奏和衔接，音色相似度与页面提供的同文本整句原始音频比较。

本次使用我本人的评分，原始文件已保存在 `results/ratings_1.json`，汇总已写入实验说明。可使用以下命令从原始评分重新生成汇总和相关图表

```powershell
.\.venv\Scripts\python.exe summarize_listening.py results/ratings_1.json
```

每位听众填写唯一匿名编号，例如 L01，并完成全部评分后点击“导出评分 JSON”。浏览器会下载 `ratings_L01.json`。将文件放到 `results`，在“hw3”文件夹执行

```powershell
.\.venv\Scripts\python.exe summarize_listening.py results/ratings_L01.json
```

多名听众的文件可以一次传入

```powershell
.\.venv\Scripts\python.exe summarize_listening.py results/ratings_L01.json results/ratings_L02.json
```

程序检查编号重复、漏评和评分范围，生成 `results/subjective_summary.csv`，并自动更新 `实验说明.md`。汇总采用每名听众每个条件的均分，再在听众之间求平均。听众人数较少时，只能作为小样本结果。浏览器保存的进度与文件路径、浏览器有关，换设备时应重新填写或保留已导出的文件。

为避免实验发生变化后混用评分，正式试听后应保留当前 `configs/cases.json`、音频和 `results/blind_key.json`，修改文本或控制参数后需要重新组织试听。

## 源码结构

| 文件 | 作用 |
|---|---|
| src/control.py | 模型加载、原始波形缓存、WORLD 情感韵律处理、时长伸缩与拼接 |
| configs/cases.json | 测试文本、情感标记和目标时长倍率 |
| download_models.py | 模型下载、安全路径检查、来源与哈希记录 |
| run_experiments.py | 生成五种条件的音频和采样点标注 |
| verify_results.py | 检查音频格式、时长误差和未控制片段一致性 |
| evaluate.py | 中文识别 CER、DNSMOS、说话人相似度和片段测量 |
| make_review.py | 离线试听页面、匿名映射和图表 |
| summarize_listening.py | 真实主观评分校验与汇总 |
| build_report.py | 从实际结果更新第一人称实验说明 |
| run_all.ps1 | Windows 完整复现入口 |
| requirements.txt | 本轮核心依赖版本 |

## 常见问题

- 模型加载失败时，先运行 `download_models.py` 检查模型是否完整。`models/sources.json` 记录了本轮模型的 SHA-256，可用于核对复制后的文件。
- 项目目录包含中文字符，入口脚本会切换工作目录，并给底层推理库使用英文相对模型路径，以兼容 Windows 文件路径处理。
- 如果播放音量偏小，可以调高设备音量。本轮输出使用所有条件一致的增益，请在评测过程中保持播放器音量一致。
- DNSMOS 是语音质量预测指标，主观自然度、情感类别和转折流畅度仍需真实试听分析。详细限制见实验说明。

我保留了第三方模型自带的 README 和许可证。模型、推理框架与声码器来源见 `实验说明.md` 和 `models/sources.json`。
