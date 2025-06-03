<div align="center">
<img src="logo.png" alt="Bytedance-seed" width="300"/>
</div>

<div align="center">
<h2>EvaLearn: Quantifying the Learning Capability and
Efficiency of LLMs via Sequential Problem Solving</h2>

[![Paper](https://img.shields.io/badge/Paper-Arxiv-blue.svg?style=for-the-badge)](./EvaLearn-paper.pdf)
[![Code License](https://img.shields.io/badge/Code_License-Apache_2.0-yellow.svg?style=for-the-badge)](./LICENSE)
[![Data License](https://img.shields.io/badge/Data_License-CC_BY_4.0-red.svg?style=for-the-badge)](./DATA_LICENSE)

</div>

## 📚 Overview

EvaLearn is a pioneering benchmark designed to evaluate large language models (LLMs) on their learning capability and efficiency in challenging tasks.

EvaLearn contains 648 challenging problems across six task types, grouped into 182 sequences, each sequence dedicated to one task type.

Diverging from most existing benchmarks that evaluate models in parallel, EvaLearn requires models to solve problems sequentially, allowing them to leverage the experience gained from previous solutions.

### 🧩 Framework Components

The EvaLearn evaluation framework consists of:

1. A sequential evaluation tool (`evaluate.py`) that processes sequences of questions
2. A dataset of problem definitions (`EvaLearn_Problem.json`)
3. A dataset of sequence definitions (`EvaLearn_Sequence.json`)

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- API Keys for OpenAI and Alternative LLMs

### Installation

```bash
git clone https://github.com/yourusername/EvaLearn.git
cd EvaLearn
pip install -r requirements.txt
```

## 🛠️ Usage

### Command Line Interface

The main entry point is the `sequentialEval` function in `evaluate.py`. You can run it from the command line:

```bash
python EvaLearn/Evaluate/evaluate.py --input EvaLearn/Dataset/EvaLearn_Problem.json \
                                    --seq EvaLearn/Dataset/EvaLearn_Sequence.json \
                                    --output results.json \
                                    --workers 4 \
                                    --client-api-key YOUR_CLIENT_API_KEY \
                                    --judge-api-key YOUR_JUDGE_API_KEY
```

### Command Line Arguments


| Argument           | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| `--input`          | Path to the problem JSON file                                    |
| `--seq`            | Path to the sequence JSON file                                   |
| `--output`         | Path to save the evaluation results                              |
| `--workers`        | Number of worker threads for parallel processing                 |
| `--no-check-empty` | Skip checking for empty responses                                |
| `--judge-api-key`  | API key for the judge model                                      |
| `--client-api-key` | API key for the client model                                     |
| `--judge-model`    | Model to use for judging (default: "gpt-4o-2024-11-20")          |
| `--client-model`   | Model to use for client responses (default: "gpt-4o-2024-11-20") |

### Library Usage

You can also import and use the functions directly in your Python code:

```python
from EvaLearn.Evaluate.evaluate import sequentialEval, load_evaluation_data, select_sequences_for_evaluation

# Load data
sequences, problems_dict = load_evaluation_data(
    "EvaLearn/Dataset/EvaLearn_Sequence.json",
    "EvaLearn/Dataset/EvaLearn_Problem.json"
)

# Select specific sequences
selected_sequences = select_sequences_for_evaluation(
    sequences, 
    num_sequences=5,  # Randomly select 5 sequences
    sequence_types=["Logical Reasoning"]  # Only select sequences of this type
)

# Run evaluation
sequentialEval(
    input_json_path="EvaLearn/Dataset/EvaLearn_Problem.json",
    seq_json_path="EvaLearn/Dataset/EvaLearn_Sequence.json",
    output_json_path="results.json",
    worker_nums=4,
    client_api_key="YOUR_CLIENT_API_KEY",
    judge_api_key="YOUR_JUDGE_API_KEY"
)
```

## 📈 Evaluation Metrics

EvaLearn provides a systematic set of evaluation metrics to quantify the learning capability and efficiency of large language models in sequential problem solving. You can use `EvaLearn/Evaluate/evaluate_metric.py` to automatically compute the following five core metrics from your model's output (in JSON format):

# EvaLearn Metrics Evaluation

This repository provides a script for evaluating model performance on sequential problem-solving tasks using a variety of metrics. The main script is `Evaluate/evaluate_metric.py`.

## Features

- **Overall sequence accuracy**
- **Slope of fitted accuracy curve (learning speed)**
- **Average position of first correct solution**
- **Average number of consecutive correct solutions**
- **Post-warmup accuracy (excluding first K problems)**
- **Metrics by task type**

## Usage

### 1. Prepare Your Results

Your results should be in a JSON file, where each item contains at least:
- `sequence_id`: Unique identifier for a sequence
- `position_in_sequence`: Position (1-based) of the problem in the sequence
- `type`: (Optional) Task type/category
- `gpt4judge`: String containing a JSON with an `answer_score` field, e.g.:
  ```
  ... ```json
  {"answer_score": 1}
  ```
  ```

### 2. Run the Evaluation

```bash
python Evaluate/evaluate_metric.py --results <results.json> [--problems 7] [--warmup 3] [--output <report.json>]
```

- `--results`: Path to your results JSON file (**required**)
- `--problems`: Number of problems per sequence (default: 7)
- `--warmup`: Number of initial problems to exclude for post-warmup accuracy (default: 3)
- `--output`: Path to save the report as JSON (default: `report_<results.json>`)

### 3. Output

- Prints a summary of all metrics to the console, including:
  - Overall metrics
  - Position-wise accuracy
  - Metrics by task type
- Saves a detailed report as a JSON file (if `--output` is specified).

### 4. Example

```bash
python Evaluate/evaluate_metric.py --results my_eval_results.json --problems 7 --warmup 3 --output my_report.json
```

## Metrics Explained

- **Overall sequence accuracy**: Proportion of correct answers across all problems and sequences.
- **Position-wise Accuracy**: Accuracy at each position in the sequence.
- **Accuracy Slope (k)**: Slope of a linear fit to position-wise accuracy; higher means faster learning within a sequence.
- **Average position of first correct solution**: Average position in the sequence where the first correct answer appears (lower is better).
- **Average number of consecutive correct solutions**: Average length of the longest streak of correct answers per sequence.
- **Post-warmup Accuracy**: Accuracy after excluding the first K problems in each sequence.


## Logging

- Logs are saved to `evaluation_metrics.log` and also printed to the console.

## 📊 Data Format

### Problem JSON Format

Each problem in `EvaLearn_Problem.json` has the following structure:

```json
{
  "id": 1,
  "type": "Logical Reasoning",
  "source": "LogicGame-crypto_puzzle",
  "level": 1,
  "prompt": ["The question text that will be presented to the model"],
  "rubric": "Evaluation criteria for judging the model's response",
  "canonical_answer": "The expected correct answer" # Due to certain intellectual property issues, the canonical answers cannot be open-sourced **at this time**.
}
```


| Field              | Description                                                                   |
| ------------------ | ----------------------------------------------------------------------------- |
| `id`               | Unique identifier for the problem                                             |
| `type`             | Category of the problem (e.g., "Logical Reasoning", "Mathematical Reasoning") |
| `source`           | Origin of the problem                                                         |
| `level`            | Difficulty level                                                              |
| `prompt`           | The question text (can be a string or an array of strings)                    |
| `rubric`           | Criteria used by the judge model to evaluate responses                        |
| `canonical_answer` | The expected correct answer                                                   |

### Sequence JSON Format

Each sequence in `EvaLearn_Sequence.json` has the following structure:

```json
{
  "sequence_id": 1,
  "type": "Extraction",
  "question_ids": [252, 258, 297, 263, 245, 273, 241]
}
```


| Field          | Description                                                        |
| -------------- | ------------------------------------------------------------------ |
| `sequence_id`  | Unique identifier for the sequence                                 |
| `type`         | Category of the sequence (e.g., "Extraction", "Logical Reasoning") |
| `question_ids` | Ordered list of problem IDs that form the sequence                 |

## 🔑 Key Functions

### `sequentialEval`

The main evaluation function that processes sequences of questions.

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
    client_model="gpt-4o-2024-11-20"
)
```

### `load_evaluation_data`

Loads and validates sequence and problem data.

```python
load_evaluation_data(sequence_path, problem_path)
```

### `select_sequences_for_evaluation`

Selects sequences for evaluation based on criteria.

```python
select_sequences_for_evaluation(
    sequences, 
    num_sequences=None, 
    sequence_ids=None, 
    sequence_types=None
)
```

### `evaluate_sequence`

Evaluates a complete sequence of questions.

```python
evaluate_sequence(
    sequence, 
    problems_dict, 
    annotator, 
    output_dir, 
    save_results=True
)
```

## 📄 License

This project is licensed under the Apache-2.0 License - see the LICENSE file for details.

## 📧 Contact

Shihan Dou: shdou24@m.fudan.edu.cn

Ming Zhang: mingzhang23@m.fudan.edu.cn

## ❤️ Acknowledgement

We gratefully acknowledge the significant contributions made by **the annotation teams at ByteDance**, whose diligent work was essential to the success of this paper ❤️❤️. The core members of the annotation team include Di Cheng, Linhua Deng, Yanxi Fu, Yafei Qiao, Chaoqian Ren, Mei Su, Ying Wu, Baitong Yang, and Xingyu Zhu.

We also wish to express our sincere appreciation to **an undisclosed third-party annotation company ❤️❤️** for their substantial support in data annotation.
Finally, we would like to thank all individuals who participated in and supported this project for their valuable input.

## 👋Citation

TBD
