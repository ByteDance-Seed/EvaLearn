<div align="center">
<img src="logo.png" alt="Bytedance-seed" width="300"/>
</div>

<div align="center">
<h2>EvaLearn: Quantifying the Learning Capability and
Efficiency of LLMs via Sequential Problem Solving</h2>

[![Paper](https://img.shields.io/badge/Paper-Arxiv-blue.svg?style=for-the-badge)](https://arxiv.org/abs/2506.02672)
[![Code License](https://img.shields.io/badge/Code_License-Apache_2.0-yellow.svg?style=for-the-badge)](./LICENSE)
[![Data License](https://img.shields.io/badge/Data_License-CC_BY_4.0-red.svg?style=for-the-badge)](./DATA_LICENSE)

</div>

## 📰 News

- **📅 Jan 13, 2026**: New update! 🎉 We've released **four evaluation scripts** covering different learning paradigms (Zero-shot, Few-shot, Demonstration Learning, Feedback Learning), and **published canonical answers** for all problems in the dataset!
- **📅 Sep 18, 2025**: EvaLearn was accepted to the NeurIPS 2025 main track with a high score of 5/5/5/5! 🎉
- **📅 Jul 15, 2025**: We've released a new version! 🎉 Open-sourced complete Chinese rubrics, updated Chinese README documentation, and optimized evaluation scripts for improved efficiency and accuracy.
- **📅 Jun 5, 2025**: EvaLearn is officially open-sourced! 🚀 We released this innovative benchmark for evaluating the learning capability and efficiency of large language models.

## 📚 Overview

EvaLearn is a benchmark designed to evaluate large language models (LLMs) on their learning capability and efficiency. It contains 648 challenging problems across six task types, grouped into 182 sequences. Unlike traditional benchmarks that evaluate models in parallel, EvaLearn requires models to solve problems sequentially, allowing them to leverage experience from previous solutions.

### 🧩 Framework Components

The EvaLearn evaluation framework consists of:

1. **Four evaluation scripts** for different learning paradigms:
   - `Evaluate/zero_shot.py` - Zero-shot evaluation (no context)
   - `Evaluate/few_shot.py` - Few-shot evaluation (with example Q&A pairs)
   - `Evaluate/demonstration_learning.py` - Demonstration learning (with cumulative standard answers)
   - `Evaluate/feedback_learning.py` - Feedback learning (with cumulative answers and teacher evaluations)
2. A dataset of problem definitions (`Dataset/EvaLearn_Problem.json`) **with canonical answers**
3. A dataset of sequence definitions (`Dataset/EvaLearn_Sequence.json`)
4. A metrics evaluation tool (`Evaluate/evaluate_metric.py`) for analyzing results

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/EvaLearn.git
cd EvaLearn
pip install -r requirements.txt
```

## 🛠️ Usage

We provide **four evaluation scripts** for different learning paradigms. Each script evaluates how well the model learns under different conditions.

### 📋 Evaluation Paradigms

| Paradigm | Script | Description |
| -------- | ------ | ----------- |
| **Zero-shot** | `zero_shot.py` | Model answers each question independently without any context |
| **Few-shot** | `few_shot.py` | Model receives example Q&A pairs of the same task type before answering |
| **Demonstration Learning** | `demonstration_learning.py` | Model accumulates standard answers from previous questions in sequence |
| **Feedback Learning** | `feedback_learning.py` | Model accumulates its own answers and teacher evaluations from previous questions |

### Command Line Interface

#### Zero-shot Evaluation
```bash
python Evaluate/zero_shot.py --input Dataset/EvaLearn_Problem.json \
                             --seq Dataset/EvaLearn_Sequence.json \
                             --output results_zero_shot.json \
                             --workers 4 \
                             --client-api-key YOUR_CLIENT_API_KEY \
                             --judge-api-key YOUR_JUDGE_API_KEY
```

#### Few-shot Evaluation
```bash
python Evaluate/few_shot.py --input Dataset/EvaLearn_Problem.json \
                            --seq Dataset/EvaLearn_Sequence.json \
                            --output results_few_shot.json \
                            --workers 4 \
                            --client-api-key YOUR_CLIENT_API_KEY \
                            --judge-api-key YOUR_JUDGE_API_KEY
```

#### Demonstration Learning Evaluation
```bash
python Evaluate/demonstration_learning.py --input Dataset/EvaLearn_Problem.json \
                                          --seq Dataset/EvaLearn_Sequence.json \
                                          --output results_demonstration.json \
                                          --workers 4 \
                                          --client-api-key YOUR_CLIENT_API_KEY \
                                          --judge-api-key YOUR_JUDGE_API_KEY
```

#### Feedback Learning Evaluation
```bash
python Evaluate/feedback_learning.py --input Dataset/EvaLearn_Problem.json \
                                     --seq Dataset/EvaLearn_Sequence.json \
                                     --output results_feedback.json \
                                     --workers 4 \
                                     --client-api-key YOUR_CLIENT_API_KEY \
                                     --judge-api-key YOUR_JUDGE_API_KEY
```

### Command Line Arguments

| Argument                  | Description                                                      |
| ------------------------- | ---------------------------------------------------------------- |
| `--input`               | Path to the problem JSON file                                    |
| `--seq`                 | Path to the sequence JSON file                                   |
| `--output`              | Path to save the evaluation results                              |
| `--workers`             | Number of worker threads for parallel processing                 |
| `--no-check-empty`      | Skip checking for empty responses                                |
| `--judge-api-key`       | API key for the judge model                                      |
| `--client-api-key`      | API key for the client model                                     |
| `--judge-model`         | Model to use for judging (default: "gpt-4o-2024-11-20")          |
| `--client-model`        | Model to use for client responses (default: "gpt-4o-2024-11-20") |
| `--judge-api-base-url`  | Custom base URL for judge API calls                              |
| `--client-api-base-url` | Custom base URL for client API calls                             |

### Key Features

- **Checkpoint Recovery**: Automatically resumes interrupted evaluations
- **API Compatibility**: Support for custom API endpoints
- **Parallel Processing**: Multi-threaded execution for faster processing

### Library Usage

```python
# Zero-shot evaluation
from Evaluate.zero_shot import zeroShotEval
zeroShotEval(
    input_json_path="Dataset/EvaLearn_Problem.json",
    seq_json_path="Dataset/EvaLearn_Sequence.json",
    output_json_path="results_zero_shot.json",
    client_api_key="YOUR_CLIENT_API_KEY",
    judge_api_key="YOUR_JUDGE_API_KEY"
)

# Few-shot evaluation
from Evaluate.few_shot import fewShotEval
fewShotEval(
    input_json_path="Dataset/EvaLearn_Problem.json",
    seq_json_path="Dataset/EvaLearn_Sequence.json",
    output_json_path="results_few_shot.json",
    client_api_key="YOUR_CLIENT_API_KEY",
    judge_api_key="YOUR_JUDGE_API_KEY"
)

# Demonstration learning evaluation
from Evaluate.demonstration_learning import demonstrationEval
demonstrationEval(
    input_json_path="Dataset/EvaLearn_Problem.json",
    seq_json_path="Dataset/EvaLearn_Sequence.json",
    output_json_path="results_demonstration.json",
    client_api_key="YOUR_CLIENT_API_KEY",
    judge_api_key="YOUR_JUDGE_API_KEY"
)

# Feedback learning evaluation
from Evaluate.feedback_learning import sequentialEval
sequentialEval(
    input_json_path="Dataset/EvaLearn_Problem.json",
    seq_json_path="Dataset/EvaLearn_Sequence.json",
    output_json_path="results_feedback.json",
    client_api_key="YOUR_CLIENT_API_KEY",
    judge_api_key="YOUR_JUDGE_API_KEY"
)
```

## 📈 Evaluation Metrics

Use `Evaluate/evaluate_metric.py` to compute learning metrics from your results:

```bash
python Evaluate/evaluate_metric.py --results results.json --output report.json
```

### Metrics

- Overall sequence accuracy
- Position-wise Accuracy
- Slope of fitted accuracy curve
- Average position of first correct solution
- Average number of consecutive correct solutions
- Post-warmup Accuracy

For detailed metric descriptions, please refer to the Section 2.3 of the paper.

### Usage

#### 1. Prepare Your Results

Your results should be in a JSON file, where each item contains at least:`sequence_id`: Unique identifier for a sequence

- `position_in_sequence`: Position (1-based) of the problem in the sequence
- `type`: (Optional) Task type/category
- `gpt4judge`: String containing a JSON with an `answer_score` field

#### 2. Run the Evaluation

```bash
python Evaluate/evaluate_metric.py --results <results.json> [--problems 7] [--warmup 3] [--output <report.json>]
```

- `--results`: Path to your results JSON file (**required**)
- `--problems`: Number of problems per sequence (default: 7)
- `--warmup`: Number of initial problems to exclude for post-warmup accuracy (default: 3)
- `--output`: Path to save the report as JSON (default: `report_<results.json>`)

#### 3. Output

- Prints a summary of all metrics to the console, including:
  - Overall metrics
  - Position-wise accuracy
  - Metrics by task type
- Saves a detailed report as a JSON file (if `--output` is specified).

#### 4. Example

```bash
python Evaluate/evaluate_metric.py --results my_eval_results.json --problems 7 --warmup 3 --output my_report.json
```

### Logging

- Logs are saved to `evaluation_metrics.log` and also printed to the console.

## 📊 Data Format

### Problem JSON Format

Each problem in `Dataset/EvaLearn_Problem.json` has the following structure:

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

| Field                | Description                                                                   |
| -------------------- | ----------------------------------------------------------------------------- |
| `id`               | Unique identifier for the problem                                             |
| `type`             | Category of the problem (e.g., "Logical Reasoning", "Mathematical Reasoning") |
| `source`           | Origin of the problem                                                         |
| `level`            | Difficulty level                                                              |
| `prompt`           | The question text (can be a string or an array of strings)                    |
| `rubric_zh`        | Chinese evaluation criteria used by the judge model                           |
| `rubric_en`        | English evaluation criteria used by the judge model                           |
| `canonical_answer` | **🆕 The expected correct answer (now publicly available!)**                  |

**Note**: 
- The results in our paper use the Chinese rubric, which was carefully annotated by our annotation team and is of high quality. The English version was translated using a large language model to help understand the meaning of the rubric. Therefore, we **strongly recommend** that everyone use the **Chinese rubric** for evaluation. We will also update it with a high-quality English rubric in the future.
- **🆕 Canonical answers are now included in the dataset!** These can be used for few-shot and demonstration learning evaluations.

### Sequence JSON Format

Each sequence in `Dataset/EvaLearn_Sequence.json` has the following structure:

```json
{
  "sequence_id": 1,
  "type": "Extraction",
  "question_ids": [252, 258, 297, 263, 245, 273, 241]
}
```

| Field            | Description                                                        |
| ---------------- | ------------------------------------------------------------------ |
| `sequence_id`  | Unique identifier for the sequence                                 |
| `type`         | Category of the sequence (e.g., "Extraction", "Logical Reasoning") |
| `question_ids` | Ordered list of problem IDs that form the sequence                 |

## 🔑 Key Functions

### Main Evaluation Functions

We provide four main evaluation functions, one for each learning paradigm:

| Function | Script | Description |
| -------- | ------ | ----------- |
| `zeroShotEval` | `zero_shot.py` | Zero-shot evaluation without context |
| `fewShotEval` | `few_shot.py` | Few-shot evaluation with example Q&A pairs |
| `demonstrationEval` | `demonstration_learning.py` | Demonstration learning with cumulative standard answers |
| `sequentialEval` | `feedback_learning.py` | Feedback learning with cumulative answers and evaluations |

All functions share the same parameter signature:

```python
eval_function(
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

**Parameters:**

- `input_json_path`: Path to the problem JSON file
- `seq_json_path`: Path to the sequence JSON file
- `output_json_path`: Path to save evaluation results
- `worker_nums`: Number of worker threads (default: 5)
- `check_empty`: Whether to check and reprocess empty responses (default: True)
- `judge_api_key`: API key for judge model
- `client_api_key`: API key for client model
- `judge_model`: Model name for judging (default: "gpt-4o-2024-11-20")
- `client_model`: Model name for responses (default: "gpt-4o-2024-11-20")
- `judge_api_base_url`: Custom base URL for judge API
- `client_api_base_url`: Custom base URL for client API

### Core Processing Functions

Each evaluation script contains paradigm-specific inference functions:

- **`zero_shot_infer_and_judge`**: Processes questions without any context
- **`few_shot_infer_and_judge`**: Processes questions with few-shot examples
- **`demonstration_infer_and_judge`**: Processes questions with cumulative standard answers
- **`sequential_infer_and_judge`**: Processes questions with cumulative feedback

### Utility Functions

#### `load_json` / `save_json`

Handle JSON file loading and saving with error handling and backup mechanisms.

#### `find_empty_responses`

Identifies sequences with empty responses for reprocessing.

### Configuration

Each script uses a `CONFIG` dictionary for various parameters:

- `max_retries`: Maximum API call retries (default: 10)
- `initial_delay`: Initial retry delay in seconds (default: 1)
- `max_delay`: Maximum retry delay in seconds (default: 60)
- `worker_nums`: Default number of worker threads (default: 5)
- `questions_per_sequence`: Expected questions per sequence (default: 7)

## 📄 License

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.

## 📧 Contact

Shihan Dou: shdou24@m.fudan.edu.cn

Ming Zhang: mingzhang23@m.fudan.edu.cn

## ❤️ Acknowledgement

We gratefully acknowledge the significant contributions made by **the annotation teams at ByteDance**, whose diligent work was essential to the success of this paper ❤️❤️. The core members of the annotation team include **Di Cheng, Linhua Deng, Yanxi Fu, Yafei Qiao, Chaoqian Ren, Mei Su, Ying Wu, Baitong Yang, and Xingyu Zhu.**

We also wish to express our sincere appreciation to **an undisclosed third-party annotation company ❤️❤️** for their substantial support in data annotation.
Finally, we would like to thank all individuals who participated in and supported this project for their valuable input.

## 👋Citation

```
@article{dou2025evalearn,
  title={EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving},
  author={Dou, Shihan and Zhang, Ming and Huang, Chenhao and Chen, Jiayi and Chen, Feng and Liu, Shichun and Liu, Yan and Liu, Chenxiao and Zhong, Cheng and Zhang, Zongzhang and others},
  journal={arXiv preprint arXiv:2506.02672},
  year={2025}
}
```

