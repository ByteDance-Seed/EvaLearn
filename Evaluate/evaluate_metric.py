# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation_metrics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EvaLearnMetrics:
    """
    Compute evaluation metrics for EvaLearn based on model performance.
    
    Metrics:
    1. Overall sequence accuracy (Acc)
    2. Slope of fitted accuracy curve (k)
    3. Average position of first correct solution (P_first)
    4. Average number of consecutive correct solutions (N_consec)
    5. Post-warmup accuracy (Acc_pw-K)
    """
    
    def __init__(self, results_file, num_problems_per_sequence=7):
        """
        Initialize the metrics calculator.
        
        Args:
            results_file (str): Path to the JSON file with evaluation results
            num_problems_per_sequence (int): Number of problems in each sequence
        """
        self.results_file = results_file
        self.M = num_problems_per_sequence  # Problems per sequence
        self.results = self._load_results()
        self.y_nm, self.sequence_types, self.sequence_outcomes = self._extract_binary_outcomes()
        self.N = len(self.y_nm)  # Number of sequences
        
        # Group sequences by type
        self.type_sequences = {}
        for seq_id, seq_type in self.sequence_types.items():
            if seq_type not in self.type_sequences:
                self.type_sequences[seq_type] = []
            self.type_sequences[seq_type].append(seq_id)
        
        logger.info(f"Loaded {self.N} sequences with {self.M} problems each")
        logger.info(f"Sequence types: {list(self.type_sequences.keys())}")
        
    def _load_results(self):
        """Load results from JSON file."""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            return results
        except Exception as e:
            logger.error(f"Failed to load results file: {e}")
            raise
            
    def _extract_binary_outcomes(self):
        """
        Extract binary outcomes (y_nm) from results and organize by sequence and type.
        
        Returns:
            Tuple[List[List[int]], dict, dict]: 
                - A 2D array where y_nm[n][m] is 1 if the m-th problem in the n-th sequence is answered correctly
                - A dict mapping sequence IDs to their types
                - A dict mapping sequence IDs to their binary outcomes
        """
        # Group results by sequence_id
        sequences = {}
        sequence_types = {}
        
        for item in self.results:
            seq_id = item.get('sequence_id')
            position = item.get('position_in_sequence')
            seq_type = item.get('type', 'unknown')
            
            if seq_id is None or position is None:
                logger.warning(f"Missing sequence_id or position in item: {item}")
                continue
            
            # Store sequence type
            if seq_id not in sequence_types:
                sequence_types[seq_id] = seq_type
            
            # Extract score from judge response
            try:
                judge_response = item.get('gpt4judge', '')
                
                # Find the JSON part in the judge response
                if '```json' in judge_response:
                    json_part = judge_response.split('```json')[1].split('```')[0].strip()
                    score_json = json.loads(json_part)
                    score = score_json.get('answer_score', 0)
                else:
                    # Try to find {"answer_score": X} pattern
                    import re
                    match = re.search(r'{"answer_score":\s*(\d+)}', judge_response)
                    if match:
                        score = int(match.group(1))
                    else:
                        logger.warning(f"Could not extract score from judge response for item in sequence {seq_id}, position {position}")
                        score = 0
                        
                # Convert to binary outcome
                binary_outcome = 1 if score > 0 else 0
                
            except Exception as e:
                logger.error(f"Error extracting score: {e}")
                binary_outcome = 0
                
            # Store in sequences dictionary
            if seq_id not in sequences:
                sequences[seq_id] = {}
            sequences[seq_id][position] = binary_outcome
        
        # Convert to 2D array
        y_nm = []
        sequence_outcomes = {}
        
        for seq_id, positions in sequences.items():
            # Check if sequence has all positions
            if len(positions) != self.M:
                logger.warning(f"Sequence {seq_id} has {len(positions)} positions, expected {self.M}")
                # Skip incomplete sequences or fill missing positions with 0
                if len(positions) < self.M:
                    logger.warning(f"Filling missing positions in sequence {seq_id} with 0")
                    for pos in range(1, self.M + 1):
                        if pos not in positions:
                            positions[pos] = 0
            
            # Create sequence array in order of positions
            seq_array = [positions.get(pos, 0) for pos in range(1, self.M + 1)]
            y_nm.append(seq_array)
            sequence_outcomes[seq_id] = seq_array
        
        return y_nm, sequence_types, sequence_outcomes
    
    def compute_all_metrics(self, warmup_k=3, by_type=False):
        """
        Compute all metrics and return as a dictionary.
        
        Args:
            warmup_k (int): Parameter K for post-warmup accuracy
            by_type (bool): Whether to calculate metrics by task type
            
        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {
            "overall_accuracy": self.overall_accuracy(by_type),
            "position_accuracy": self.position_accuracy(by_type, avg_by_type=True),
            "accuracy_slope": self.accuracy_slope(by_type),
            "first_correct_position": self.first_correct_position(by_type, avg_by_type=True),
            "consecutive_correct": self.consecutive_correct(by_type, avg_by_type=True),
            f"post_warmup_accuracy_k{warmup_k}": self.post_warmup_accuracy(warmup_k, by_type)
        }
        return metrics
    
    def overall_accuracy(self, by_type=False):
        """
        Calculate the overall sequence accuracy (Acc).
        
        This metric measures the proportion of correctly answered problems across all sequences.
        Higher values indicate better performance.
        
        Args:
            by_type (bool): Whether to calculate metrics by task type
            
        Returns:
            float or dict: Overall accuracy across all problems and sequences, or results by type
        """
        if not by_type:
            total_correct = sum(sum(seq) for seq in self.y_nm)
            total_problems = self.N * self.M
            return total_correct / total_problems if total_problems > 0 else 0
        else:
            type_accuracy = {}
            for seq_type, seq_ids in self.type_sequences.items():
                correct = 0
                total = 0
                for seq_id in seq_ids:
                    correct += sum(self.sequence_outcomes[seq_id])
                    total += self.M
                type_accuracy[seq_type] = correct / total if total > 0 else 0
            return type_accuracy
    
    def position_accuracy(self, by_type=False, avg_by_type=False):
        """
        Calculate the position-wise accuracy curve (Acc_m).
        
        This metric shows how performance changes based on problem position in the sequence.
        
        Args:
            by_type (bool): Whether to calculate metrics by task type
            avg_by_type (bool): If True, return average position accuracy across task types
                                (each task type given equal weight)
            
        Returns:
            numpy.ndarray or dict: Accuracy at each position, or results by type
        """
        # Calculate by type first
        type_position_accuracy = {}
        for seq_type, seq_ids in self.type_sequences.items():
            # Collect all sequences of this type
            seqs = [self.sequence_outcomes[seq_id] for seq_id in seq_ids]
            if seqs:
                type_position_accuracy[seq_type] = np.mean(np.array(seqs), axis=0)
            else:
                type_position_accuracy[seq_type] = np.zeros(self.M)
        
        if by_type:
            return type_position_accuracy
        elif avg_by_type:
            # Average across types (giving equal weight to each type)
            all_type_accs = np.array(list(type_position_accuracy.values()))
            return np.mean(all_type_accs, axis=0)
        else:
            # Traditional overall average (all sequences weighted equally)
            y_nm_array = np.array(self.y_nm)
            return np.mean(y_nm_array, axis=0)
    
    def accuracy_slope(self, by_type=False):
        """
        Calculate the slope of the fitted accuracy curve (k).
        
        This metric measures how quickly the model learns within a sequence.
        Positive values indicate improvement over the sequence.
        
        Args:
            by_type (bool): Whether to calculate metrics by task type
            
        Returns:
            float or dict: Slope of the linear regression line, or results by type
        """
        if not by_type:
            # Use type-averaged position accuracy for slope calculation
            acc_m = self.position_accuracy(avg_by_type=True)
            positions = np.arange(1, self.M + 1)
            
            # Reshape data for sklearn
            X = positions.reshape(-1, 1)
            y = acc_m.reshape(-1, 1)
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            return float(model.coef_[0][0])
        else:
            type_slopes = {}
            positions = np.arange(1, self.M + 1).reshape(-1, 1)
            
            type_acc_m = self.position_accuracy(by_type=True)
            for seq_type, acc_m in type_acc_m.items():
                if len(acc_m) > 0:
                    # Fit linear regression
                    model = LinearRegression()
                    model.fit(positions, acc_m.reshape(-1, 1))
                    type_slopes[seq_type] = float(model.coef_[0][0])
                else:
                    type_slopes[seq_type] = 0.0
                
            return type_slopes
    
    def first_correct_position(self, by_type=False, avg_by_type=False):
        """
        Calculate the average position of first correct solution (P_first).
        
        This metric indicates how quickly the model can provide a correct answer.
        Lower values indicate better performance.
        
        Args:
            by_type (bool): Whether to calculate metrics by task type
            avg_by_type (bool): If True, average across task types (equal weight per type)
            
        Returns:
            float or dict: Average position of first correct solution, or results by type
        """
        # Calculate by type first
        type_first_positions = {}
        for seq_type, seq_ids in self.type_sequences.items():
            type_first_positions[seq_type] = []
            for seq_id in seq_ids:
                seq = self.sequence_outcomes[seq_id]
                # Find the position of the first correct answer
                try:
                    first_pos = seq.index(1) + 1  # Add 1 for 1-based indexing
                except ValueError:
                    # If no correct answer, use M+1
                    first_pos = self.M + 1
                type_first_positions[seq_type].append(first_pos)
        
        # Calculate average for each type
        type_avg = {t: sum(v)/len(v) if v else 0 for t, v in type_first_positions.items()}
        
        if by_type:
            return type_avg
        elif avg_by_type:
            # Calculate average across all types (giving equal weight to each type)
            if not type_avg:
                return 0
            return sum(type_avg.values()) / len(type_avg)
        else:
            # Calculate average across all sequences (traditional method)
            all_positions = []
            for seq_positions in type_first_positions.values():
                all_positions.extend(seq_positions)
            return sum(all_positions) / len(all_positions) if all_positions else 0
    
    def consecutive_correct(self, by_type=False, avg_by_type=False):
        """
        Calculate the average number of consecutive correct solutions (N_consec).
        
        This metric measures the model's ability to maintain correct performance
        over consecutive problems. Higher values indicate better performance.
        
        Args:
            by_type (bool): Whether to calculate metrics by task type
            avg_by_type (bool): If True, average across task types (equal weight per type)
            
        Returns:
            float or dict: Average length of max consecutive correct answers, or results by type
        """
        # Calculate by type first
        type_max_consecutive = {}
        for seq_type, seq_ids in self.type_sequences.items():
            type_max_consecutive[seq_type] = []
            for seq_id in seq_ids:
                seq = self.sequence_outcomes[seq_id]
                # Find the longest consecutive sequence of 1s
                current_run = 0
                longest_run = 0
                for outcome in seq:
                    if outcome == 1:
                        current_run += 1
                        longest_run = max(longest_run, current_run)
                    else:
                        current_run = 0
                type_max_consecutive[seq_type].append(longest_run)
        
        # Calculate average for each type
        type_avg = {t: sum(v)/len(v) if v else 0 for t, v in type_max_consecutive.items()}
        
        if by_type:
            return type_avg
        elif avg_by_type:
            # Calculate average across all types (giving equal weight to each type)
            if not type_avg:
                return 0
            return sum(type_avg.values()) / len(type_avg)
        else:
            # Calculate average across all sequences (traditional method)
            all_consecutive = []
            for seq_consecutive in type_max_consecutive.values():
                all_consecutive.extend(seq_consecutive)
            return sum(all_consecutive) / len(all_consecutive) if all_consecutive else 0
    
    def post_warmup_accuracy(self, K=3, by_type=False):
        """
        Calculate the post-warmup accuracy (Acc_pw-K).
        
        This metric measures performance after excluding the first K problems,
        allowing the model to "warm up" before evaluation.
        
        Args:
            K (int): Number of initial problems to exclude (warmup period)
            by_type (bool): Whether to calculate metrics by task type
            
        Returns:
            float or dict: Average accuracy after warmup, or results by type
        """
        if K >= self.M:
            logger.warning(f"K ({K}) must be less than M ({self.M})")
            return 0.0 if not by_type else {t: 0.0 for t in self.type_sequences.keys()}
            
        if not by_type:
            total_correct = 0
            total_problems = 0
            for seq in self.y_nm:
                post_warmup = seq[K:]
                total_correct += sum(post_warmup)
                total_problems += len(post_warmup)
            return total_correct / total_problems if total_problems > 0 else 0.0
        else:
            type_post_warmup = {}
            for seq_type, seq_ids in self.type_sequences.items():
                correct = 0
                total = 0
                for seq_id in seq_ids:
                    seq = self.sequence_outcomes[seq_id]
                    post_warmup = seq[K:]
                    correct += sum(post_warmup)
                    total += len(post_warmup)
                type_post_warmup[seq_type] = correct / total if total > 0 else 0.0
            return type_post_warmup
    
    def generate_report(self, output_file=None, warmup_k=3):
        """
        Generate a comprehensive report of all metrics.
        
        Args:
            output_file (str, optional): Path to save the report as JSON. If None, just return the metrics.
            warmup_k (int): Parameter K for post-warmup accuracy
            
        Returns:
            dict: Dictionary containing all metrics
        """
        # Compute all metrics
        overall_metrics = self.compute_all_metrics(warmup_k, by_type=False)
        type_metrics = self.compute_all_metrics(warmup_k, by_type=True)
        
        # Format metrics for report with more descriptive names
        report = {
            "model_evaluation": {
                "dataset_info": {
                    "num_sequences": self.N,
                    "problems_per_sequence": self.M,
                    "total_problems": self.N * self.M
                },
                "overall_metrics": {
                    "overall_accuracy": {
                        "value": overall_metrics["overall_accuracy"],
                        "description": "Average accuracy across all problems and sequences"
                    },
                    "accuracy_slope": {
                        "value": overall_metrics["accuracy_slope"],
                        "description": "Slope of fitted accuracy curve (learning speed)"
                    },
                    "first_correct_position": {
                        "value": overall_metrics["first_correct_position"],
                        "description": "Average position of first correct solution"
                    },
                    "consecutive_correct": {
                        "value": overall_metrics["consecutive_correct"],
                        "description": "Average number of consecutive correct solutions"
                    },
                    f"post_warmup_accuracy_k{warmup_k}": {
                        "value": overall_metrics[f"post_warmup_accuracy_k{warmup_k}"],
                        "description": f"Average accuracy after excluding first {warmup_k} problems"
                    },
                    "position_accuracy": {
                        "values": overall_metrics["position_accuracy"].tolist(),
                        "description": "Accuracy at each position across all sequences"
                    }
                },
                "type_metrics": {}
            }
        }
        
        # Add type-specific metrics to the report
        for task_type in sorted(self.type_sequences.keys()):
            report["model_evaluation"]["type_metrics"][task_type] = {
                "sequence_count": len(self.type_sequences[task_type]),
                "overall_accuracy": type_metrics["overall_accuracy"][task_type],
                "accuracy_slope": type_metrics["accuracy_slope"][task_type],
                "first_correct_position": type_metrics["first_correct_position"][task_type],
                "consecutive_correct": type_metrics["consecutive_correct"][task_type],
                f"post_warmup_accuracy_k{warmup_k}": type_metrics[f"post_warmup_accuracy_k{warmup_k}"][task_type],
                "position_accuracy": type_metrics["position_accuracy"][task_type].tolist()
            }
        
        # Save report if output file is provided
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Saved evaluation report to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description='Calculate EvaLearn metrics from evaluation results')
    parser.add_argument('--results', required=True, help='Path to the evaluation results JSON file')
    parser.add_argument('--problems', type=int, default=7, help='Number of problems per sequence')
    parser.add_argument('--warmup', type=int, default=3, help='Warmup parameter K for post-warmup accuracy')
    parser.add_argument('--output', help='Path to save the report JSON file')
    
    args = parser.parse_args()
    
    # Initialize metrics calculator
    metrics = EvaLearnMetrics(args.results, args.problems)
    
    # Generate and save report
    output_file = args.output or f"report_{os.path.basename(args.results)}"
    report = metrics.generate_report(output_file=output_file, warmup_k=args.warmup)
    
    # Calculate overall metrics
    overall_metrics = metrics.compute_all_metrics(args.warmup, by_type=False)
    
    # Calculate metrics by task type
    type_metrics = metrics.compute_all_metrics(args.warmup, by_type=True)
    
    # Print statistics
    print("\n===== EvaLearn Evaluation Report =====")
    print(f"Total Questions: {metrics.N * metrics.M}")
    print(f"Total Sequences: {metrics.N}")
    print(f"Problems per Sequence: {metrics.M}")
    print(f"Report saved to: {output_file}")
    
    print("\n=== OVERALL METRICS ===")
    print(f"Overall Accuracy: {overall_metrics['overall_accuracy']:.4f}")
    print(f"Accuracy Slope (k): {overall_metrics['accuracy_slope']:.4f}")
    print(f"Avg. First Correct Position: {overall_metrics['first_correct_position']:.2f}")
    print(f"Avg. Consecutive Correct: {overall_metrics['consecutive_correct']:.2f}")
    print(f"Post-warmup Accuracy (K={args.warmup}): {overall_metrics[f'post_warmup_accuracy_k{args.warmup}']:.4f}")
    
    print("\n=== POSITION-WISE ACCURACY ===")
    for i, acc in enumerate(overall_metrics['position_accuracy']):
        print(f"Position {i+1}: {acc:.4f}")
    
    print("\n=== METRICS BY TASK TYPE ===")
    for task_type in sorted(metrics.type_sequences.keys()):
        seq_count = len(metrics.type_sequences[task_type])
        print(f"\nType: {task_type} (Sequences: {seq_count})")
        print(f"  Overall Accuracy: {type_metrics['overall_accuracy'][task_type]:.4f}")
        print(f"  Accuracy Slope (k): {type_metrics['accuracy_slope'][task_type]:.4f}")
        print(f"  Avg. First Correct Position: {type_metrics['first_correct_position'][task_type]:.2f}")
        print(f"  Avg. Consecutive Correct: {type_metrics['consecutive_correct'][task_type]:.2f}")
        print(f"  Post-warmup Accuracy (K={args.warmup}): {type_metrics[f'post_warmup_accuracy_k{args.warmup}'][task_type]:.4f}")
        
        print(f"  Position-wise Accuracy:")
        for i, acc in enumerate(type_metrics['position_accuracy'][task_type]):
            print(f"    Position {i+1}: {acc:.4f}")
        
    print("\n=====================================")
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()