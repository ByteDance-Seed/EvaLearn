<div align="center">
<img src="logo.png" alt="Bytedance-seed" width="300"/>
</div>

<div align="center">
<h2>EvaLearn: Quantifying the Learning Capability and
Efficiency of LLMs via Sequential Problem Solving</h2>

[![论文](https://img.shields.io/badge/Paper-arXiv-blue.svg?style=for-the-badge)](https://arxiv.org/abs/2506.02672)
[![会议论文集](https://img.shields.io/badge/Proceedings-NeurIPS%202025-8A2BE2.svg?style=for-the-badge)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b7383744ec8a316f93e235a1f7f03468-Abstract-Conference.html)
[![会议](https://img.shields.io/badge/Venue-NeurIPS%202025-orange.svg?style=for-the-badge)](https://neurips.cc/virtual/2025/poster/115778)
[![代码许可](https://img.shields.io/badge/Code_License-Apache_2.0-yellow.svg?style=for-the-badge)](./LICENSE)
[![数据许可](https://img.shields.io/badge/Data_License-CC_BY_4.0-red.svg?style=for-the-badge)](./DATA_LICENSE)

</div>

## 📰 最新动态
- **📅 2025年12月4日**: EvaLearn 已在 [NeurIPS 2025 会议论文集](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b7383744ec8a316f93e235a1f7f03468-Abstract-Conference.html)正式出版，并作为 Poster 在 NeurIPS 2025（圣地亚哥）现场展示！🎉
- **📅 2025年9月18日**: EvaLearn 以 5/5/5/5 的高分被 NeurIPS 2025 主会接收！🎉
- **📅 2025年7月15日**: 我们更新了新版本！🎉 开源了完整的中文评分标准（rubric），更新了中文README文档，并优化了评测脚本，提升了评估效率和准确性。
- **📅 2025年6月5日**: EvaLearn 正式开源！🚀 我们发布了这个创新的基准测试，用于评估大语言模型的学习能力和效率。

## 📚 概述

EvaLearn 是一个旨在评估大语言模型（LLMs）学习能力和效率的基准测试。它包含 648 个具有挑战性的问题，涵盖六种任务类型，分为 182 个序列。与传统的并行评估模型的基准测试不同，EvaLearn 要求模型按顺序解决问题，使它们能够利用从之前解决方案中获得的经验。

### 🧩 框架组件

EvaLearn 评估框架包含：

1. 一个精简的序列评估工具（`Evaluate/evaluate.py`），用于处理问题序列
2. 一个问题定义数据集（`Dataset/EvaLearn_Problem.json`）
3. 一个序列定义数据集（`Dataset/EvaLearn_Sequence.json`）
4. 一个用于分析结果的指标评估工具（`Evaluate/evaluate_metric.py`）

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/YOUR_USERNAME/EvaLearn.git
cd EvaLearn
pip install -r requirements.txt
```

## 🛠️ 使用方法

### 命令行界面

运行评估：

```bash
python Evaluate/evaluate.py --input Dataset/EvaLearn_Problem.json \
                               --seq Dataset/EvaLearn_Sequence.json \
                               --output results.json \
                               --workers 4 \
                               --client-api-key YOUR_CLIENT_API_KEY \
                               --judge-api-key YOUR_JUDGE_API_KEY
```

### 命令行参数


| 参数                    | 描述                                    |
| ----------------------- | --------------------------------------- |
| `--input`               | 问题 JSON 文件的路径                    |
| `--seq`                 | 序列 JSON 文件的路径                    |
| `--output`              | 保存评估结果的路径                      |
| `--workers`             | 并行处理的工作线程数                    |
| `--no-check-empty`      | 跳过对空响应的检查                      |
| `--judge-api-key`       | Judge模型的 API 密钥                    |
| `--client-api-key`      | Client模型的 API 密钥                   |
| `--judge-model`         | Judge模型（默认："gpt-4o-2024-11-20"）  |
| `--client-model`        | Client模型（默认："gpt-4o-2024-11-20"） |
| `--judge-api-base-url`  | Judge API 调用的 Base URL               |
| `--client-api-base-url` | Client API 调用的 Base URL              |

### 主要特性

- **检查点恢复**: 自动恢复中断的评估
- **API 兼容性**: 支持自定义 API 端点
- **并行处理**: 多线程执行以加快处理速度

### 库使用

```python
from Evaluate.evaluate import sequentialEval

sequentialEval(
    input_json_path="Dataset/EvaLearn_Problem.json",
    seq_json_path="Dataset/EvaLearn_Sequence.json",
    output_json_path="results.json",
    client_api_key="YOUR_CLIENT_API_KEY",
    judge_api_key="YOUR_JUDGE_API_KEY"
)
```

## 📈 评估指标

使用 `Evaluate/evaluate_metric.py` 从结果中计算学习指标：

```bash
python Evaluate/evaluate_metric.py --results results.json --output report.json
```

### 指标

- **整体序列准确率**
- **位置准确率**
- **拟合准确率曲线的斜率**
- **第一次正确解决方案的平均位置**
- **序列中连续正确解决问题的平均长度**
- **热身后准确率**

详细的指标描述请参阅论文第2.3节。

### 使用方法

#### 1. 准备您的结果

您的结果应该是一个 JSON 文件，其中每个项目至少包含：`sequence_id`: 序列的唯一标识符

- `position_in_sequence`: 问题在序列中的位置（从1开始）
- `type`: （可选）任务类型/类别
- `gpt4judge`: 包含带有 `answer_score` 字段的 JSON 字符串

#### 2. 运行评估

```bash
python Evaluate/evaluate_metric.py --results <results.json> [--problems 7] [--warmup 3] [--output <report.json>]
```

- `--results`: 您的结果 JSON 文件路径（**必需**）
- `--problems`: 每个序列的问题数量（默认：7）
- `--warmup`: 热身后准确率要排除的初始问题数量（默认：3）
- `--output`: 保存报告为 JSON 的路径（默认：`report_<results.json>`）

#### 3. 输出

- 向控制台打印所有指标的摘要，包括：
  - 整体指标
  - 位置准确性
  - 按任务类型分类的指标
- 将详细报告保存为 JSON 文件（如果指定了 `--output`）。

#### 4. 示例

```bash
python Evaluate/evaluate_metric.py --results my_eval_results.json --problems 7 --warmup 3 --output my_report.json
```

### 日志记录

- 日志保存到 `evaluation_metrics.log` 并打印到控制台。

## 📊 数据格式

### 问题 JSON 格式

`Dataset/EvaLearn_Problem.json` 中的每个问题都有以下结构：

```json
{
  "id": 1,
  "type": "Logical Reasoning",
  "source": "LogicGame-crypto_puzzle",
  "level": 1,
  "prompt": ["The question text that will be presented to the model"],
  "rubric_zh": "用于判断模型回答质量的中文评分标准",
  "rubric_en": "English evaluation criteria used by the judge model",
  "canonical_answer": "The expected correct answer"
}
```


| 字段               | 描述                                       |
| ------------------ | ------------------------------------------ |
| `id`               | 问题的唯一标识符                           |
| `type`             | 问题的类别（例如，"逻辑推理"、"数学推理"） |
| `source`           | 问题的来源                                 |
| `level`            | 难度级别                                   |
| `prompt`           | 问题文本（可以是字符串或字符串数组）       |
| `rubric_zh`        | 判断模型使用的中文评估标准                 |
| `rubric_en`        | 判断模型使用的英文评估标准                 |
| `canonical_answer` | 预期的正确答案                             |

**注意**: 我们论文中的结果使用的是中文评分标准，该标准由我们的标注团队精心标注，质量很高。英文版本是使用大语言模型翻译的，以帮助理解评分标准的含义。因此，我们**强烈建议**大家使用**中文评分标准**进行评估。我们未来也会更新为高质量的英文评分标准。

### 序列 JSON 格式

`Dataset/EvaLearn_Sequence.json` 中的每个序列都有以下结构：

```json
{
  "sequence_id": 1,
  "type": "Extraction",
  "question_ids": [252, 258, 297, 263, 245, 273, 241]
}
```


| 字段           | 描述                                   |
| -------------- | -------------------------------------- |
| `sequence_id`  | 序列的唯一标识符                       |
| `type`         | 序列的类别（例如，"提取"、"逻辑推理"） |
| `question_ids` | 构成序列的问题 ID 的有序列表           |

## 🔑 主要函数

### 主要评估函数

#### `sequentialEval`

处理问题序列的主要评估函数，具有检查点恢复和 API 灵活性。

```python
sequentialEval(
    input_json_path,
    seq_json_path,
    output_json_path,
    worker_nums=None,
    check_empty=True,
    judge_api_key=None,
    client_api_key=None,
    judge_model="gpt-4o-2024-11-20",
    client_model="gpt-4o-2024-11-20",
    judge_api_base_url=None,
    client_api_base_url=None
)
```

**参数:**

- `input_json_path`: 问题 JSON 文件的路径
- `seq_json_path`: 序列 JSON 文件的路径
- `output_json_path`: 保存评估结果的路径
- `worker_nums`: 工作线程数（默认：5）
- `check_empty`: 是否检查并重新处理空响应（默认：True）
- `judge_api_key`: Judge模型的 API 密钥
- `client_api_key`: Client模型的 API 密钥
- `judge_model`: 判断的模型名称（默认："gpt-4o-2024-11-20"）
- `client_model`: 响应的模型名称（默认："gpt-4o-2024-11-20"）
- `judge_api_base_url`: Judge API 调用的 Base URL
- `client_api_base_url`: Client API 调用的 Base URL

### 核心处理函数

#### `sequential_infer_and_judge`

使用推理和判断处理问题序列。

#### `process_sequence_batch`

并行处理序列批次，具有线程安全的结果保存。

#### `find_empty_responses`

识别具有空响应的序列以进行重新处理。

### 实用函数

#### `load_json` / `save_json`

处理 JSON 文件加载和保存，具有错误处理和备份机制。

#### `get_history_prompt`

从之前的交互生成格式化的历史提示。

### 配置

脚本使用 `CONFIG` 字典来配置各种参数：

- `max_retries`: 最大 API 调用重试次数（默认：10）
- `initial_delay`: 初始重试延迟（秒）（默认：1）
- `max_delay`: 最大重试延迟（秒）（默认：60）
- `worker_nums`: 默认工作线程数（默认：5）
- `questions_per_sequence`: 每个序列的预期问题数（默认：7）

## 📄 许可证

本项目根据 Apache-2.0 许可证授权 - 请参阅 LICENSE 文件了解详情。

## 📧 联系方式

窦士涵（Shihan Dou）: shdou24@m.fudan.edu.cn

张明（Ming Zhang）: mingzhang23@m.fudan.edu.cn

## ❤️ 致谢

我们由衷地感谢**字节跳动标注团队**所做出的重要贡献，他们的勤奋工作对本论文的成功至关重要 ❤️❤️。标注团队的核心成员：Di Cheng, Linhua Deng, Yanxi Fu, Yafei Qiao, Chaoqian Ren, Mei Su, Ying Wu, Baitong Yang, and Xingyu Zhu.

我们还要向**一家未公开的第三方标注公司 ❤️❤️**表示诚挚的感谢，感谢他们在数据标注方面的大力支持。

最后，我们要感谢所有参与和支持本项目的个人，感谢他们宝贵的投入。

## 👋 引用

如果您觉得 EvaLearn 对您的研究有帮助，请引用 NeurIPS 2025 正式版本：

```bibtex
@inproceedings{NEURIPS2025_b7383744,
 author    = {Dou, Shihan and Zhang, Ming and Huang, Chenhao and Chen, Jiayi and Chen, Feng and Liu, Shichun and Liu, Yan and Liu, Chenxiao and Zhong, Cheng and Zhang, Zongzhang and Gui, Tao and Xin, Chao and Wei, Chengzhi and Yan, Lin and Zhang, Qi and Huang, Xuanjing},
 title     = {{EvaLearn}: Quantifying the Learning Capability and Efficiency of {LLMs} via Sequential Problem Solving},
 booktitle = {Advances in Neural Information Processing Systems},
 editor    = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
 pages     = {125892--125945},
 publisher = {Curran Associates, Inc.},
 volume    = {38},
 year      = {2025},
 url       = {https://proceedings.neurips.cc/paper_files/paper/2025/hash/b7383744ec8a316f93e235a1f7f03468-Abstract-Conference.html}
}
```
