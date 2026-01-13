# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os
import argparse
import logging
import time
import re
from typing import Dict, List, Tuple, Union, Optional

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

# Configuration parameters
CONFIG = {
    "default_problems_per_sequence": 7,
    "default_warmup_k": 3,
    "score_extraction_patterns": [
        r'```json\s*(.*?)\s*```',
        r'{"answer_score":\s*(\d+)}',
        r'"answer_score":\s*(\d+)',
    ]
}

def load_json(file_path: str) -> List[Dict]:
    """Load and parse a JSON file with error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded {len(data)} records from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load JSON file {file_path}: {e}")
        raise

def save_json(data: Dict, file_path: str) -> None:
    """Save data to a JSON file with error handling and backup."""
    def _write_json(path: str, data: Dict) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    try:
        _write_json(file_path, data)
        logger.info(f"Successfully saved evaluation report to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON file {file_path}: {e}")
        backup_path = f"{file_path}.backup.{int(time.time())}.json"
        try:
            _write_json(backup_path, data)
            logger.info(f"Backed up data to {backup_path}")
        except Exception as e2:
            logger.critical(f"Backup save also failed: {e2}")
            raise

def extract_score_from_judge_response(judge_response: str) -> int:
    """
    Extract score from judge response using multiple patterns.
    
    Args:
        judge_response (str): The judge response text
        
    Returns:
        int: The extracted score (0 or 1)
    """
    if not judge_response or not isinstance(judge_response, str):
        logger.warning("Empty or invalid judge response")
        return 0
    
    # Try JSON block pattern first
    for pattern in CONFIG["score_extraction_patterns"]:
        try:
            if 'json' in pattern:
                match = re.search(pattern, judge_response, re.DOTALL | re.IGNORECASE)
                if match:
                    json_content = match.group(1)
                    score_json = json.loads(json_content)
                    score = score_json.get('answer_score', 0)
                    return int(score) if isinstance(score, (int, float)) else 0
            else:
                match = re.search(pattern, judge_response)
                if match:
                    return int(match.group(1))
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.debug(f"Pattern {pattern} failed to extract score: {e}")
            continue
    
    logger.warning("Could not extract score from judge response, defaulting to 0")
    return 0


class EvaLearnMetrics:
    """
    Compute evaluation metrics for EvaLearn based on model performance.
    
    Metrics:
    1. Overall sequence accuracy (Acc)
    2. Slope of fitted accuracy curve (k)
    3. Average position of first correct solution (P_first)
    4. Average number of consecutive correct solutions (N_consec)
    5. Post-warmup accuracy (Acc_pw-K)
    6. Average offset of first learned correct solution (O_learned) - optional, requires zero-shot results
    """
    
    def __init__(self, results_file: str, num_problems_per_sequence: int = None, zero_shot_file: str = None):
        """
        Initialize the metrics calculator.
        
        Args:
            results_file (str): Path to the JSON file with evaluation results
            num_problems_per_sequence (int): Number of problems in each sequence
            zero_shot_file (str, optional): Path to zero-shot evaluation results for learning offset calculation
        """
        self.results_file = results_file
        self.M = num_problems_per_sequence or CONFIG["default_problems_per_sequence"]
        self.results = self._load_results()
        self.y_nm, self.sequence_types, self.sequence_outcomes = self._extract_binary_outcomes()
        self.N = len(self.y_nm)  # Number of sequences
        
        # Load zero-shot results if provided
        self.zero_shot_correct_ids = set()
        if zero_shot_file:
            try:
                zero_shot_results = load_json(zero_shot_file)
                for item in zero_shot_results:
                    judge_response = item.get('gpt4judge', '')
                    score = extract_score_from_judge_response(judge_response)
                    if score > 0:
                        self.zero_shot_correct_ids.add(item.get('id'))
                logger.info(f"Loaded {len(self.zero_shot_correct_ids)} zero-shot correct item IDs")
            except Exception as e:
                logger.warning(f"Failed to load zero-shot results: {e}. Learning offset metric will be skipped.")
        
        # Group sequences by type
        self.type_sequences = {}
        for seq_id, seq_type in self.sequence_types.items():
            if seq_type not in self.type_sequences:
                self.type_sequences[seq_type] = []
            self.type_sequences[seq_type].append(seq_id)
        
        logger.info(f"Initialized EvaLearnMetrics with {self.N} sequences and {self.M} problems each")
        logger.info(f"Found sequence types: {list(self.type_sequences.keys())}")
        for seq_type, seq_ids in self.type_sequences.items():
            logger.info(f"  - {seq_type}: {len(seq_ids)} sequences")
        
    def _load_results(self) -> List[Dict]:
        """Load results from JSON file with enhanced error handling."""
        return load_json(self.results_file)
            
    def _extract_binary_outcomes(self) -> Tuple[List[List[int]], Dict[str, str], Dict[str, List[int]]]:
        """
        Extract binary outcomes (y_nm) from results and organize by sequence and type.
        
        Returns:
            Tuple containing:
                - A 2D array where y_nm[n][m] is 1 if the m-th problem in the n-th sequence is answered correctly
                - A dict mapping sequence IDs to their types
                - A dict mapping sequence IDs to their binary outcomes
        """
        # Group results by sequence_id
        sequences = {}
        sequence_types = {}
        processing_stats = {
            'total_items': len(self.results),
            'processed_items': 0,
            'score_extraction_failures': 0,
            'missing_fields': 0
        }
        
        for item in self.results:
            seq_id = item.get('sequence_id')
            position = item.get('position_in_sequence')
            seq_type = item.get('type', 'unknown')
            
            if seq_id is None or position is None:
                processing_stats['missing_fields'] += 1
                logger.warning(f"Missing sequence_id or position in item: {item.get('id', 'unknown')}")
                continue
            
            # Store sequence type
            if seq_id not in sequence_types:
                sequence_types[seq_id] = seq_type
            
            # Extract score from judge response using enhanced extraction
            try:
                judge_response = item.get('gpt4judge', '')
                score = extract_score_from_judge_response(judge_response)
                binary_outcome = 1 if score > 0 else 0
                processing_stats['processed_items'] += 1
                
            except Exception as e:
                processing_stats['score_extraction_failures'] += 1
                logger.error(f"Error extracting score from sequence {seq_id}, position {position}: {e}")
                binary_outcome = 0
                
            # Store in sequences dictionary
            if seq_id not in sequences:
                sequences[seq_id] = {}
            sequences[seq_id][position] = binary_outcome
        
        # Log processing statistics
        logger.info(f"Processing statistics:")
        logger.info(f"  - Total items: {processing_stats['total_items']}")
        logger.info(f"  - Successfully processed: {processing_stats['processed_items']}")
        logger.info(f"  - Score extraction failures: {processing_stats['score_extraction_failures']}")
        logger.info(f"  - Items with missing fields: {processing_stats['missing_fields']}")
        
        # Convert to 2D array
        y_nm = []
        sequence_outcomes = {}
        incomplete_sequences = []
        
        for seq_id, positions in sequences.items():
            # Check if sequence has all positions
            if len(positions) != self.M:
                incomplete_sequences.append((seq_id, len(positions)))
                logger.warning(f"Sequence {seq_id} has {len(positions)} positions, expected {self.M}")
                # Fill missing positions with 0
                for pos in range(1, self.M + 1):
                    if pos not in positions:
                        positions[pos] = 0
            
            # Create sequence array in order of positions
            seq_array = [positions.get(pos, 0) for pos in range(1, self.M + 1)]
            y_nm.append(seq_array)
            sequence_outcomes[seq_id] = seq_array
        
        if incomplete_sequences:
            logger.warning(f"Found {len(incomplete_sequences)} incomplete sequences")
        
        return y_nm, sequence_types, sequence_outcomes
    
    def compute_all_metrics(self, warmup_k: int = None, by_type: bool = False) -> Dict[str, Union[float, Dict, np.ndarray]]:
        """
        Compute all metrics and return as a dictionary.
        
        Args:
            warmup_k (int): Parameter K for post-warmup accuracy
            by_type (bool): Whether to calculate metrics by task type
            
        Returns:
            dict: Dictionary containing all metrics
        """
        warmup_k = warmup_k or CONFIG["default_warmup_k"]
        
        start_time = time.time()
        logger.info(f"Computing all metrics with warmup_k={warmup_k}, by_type={by_type}")
        
        try:
            metrics = {
                "overall_accuracy": self.overall_accuracy(by_type),
                "position_accuracy": self.position_accuracy(by_type, avg_by_type=True),
                "accuracy_slope": self.accuracy_slope(by_type),
                "first_correct_position": self.first_correct_position(by_type, avg_by_type=True),
                "consecutive_correct": self.consecutive_correct(by_type, avg_by_type=True),
                f"post_warmup_accuracy_k{warmup_k}": self.post_warmup_accuracy(warmup_k, by_type)
            }
            
            # Add learning offset metric if zero-shot results are available
            if self.zero_shot_correct_ids:
                metrics["first_learned_correct_offset"] = self.first_learned_correct_offset(by_type, avg_by_type=True)
            
            computation_time = time.time() - start_time
            logger.info(f"Metrics computation completed in {computation_time:.2f}s")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            raise
    
    def overall_accuracy(self, by_type: bool = False) -> Union[float, Dict[str, float]]:
        """
        Calculate the overall sequence accuracy (Acc).
        
        This metric measures the proportion of correctly answered problems across all sequences.
        Higher values indicate better performance.
        
        Args:
            by_type (bool): Whether to calculate metrics by task type
            
        Returns:
            float or dict: Overall accuracy across all problems and sequences, or results by type
        """
        try:
            if not by_type:
                total_correct = sum(sum(seq) for seq in self.y_nm)
                total_problems = self.N * self.M
                return total_correct / total_problems if total_problems > 0 else 0.0
            else:
                type_accuracy = {}
                for seq_type, seq_ids in self.type_sequences.items():
                    correct = 0
                    total = 0
                    for seq_id in seq_ids:
                        if seq_id in self.sequence_outcomes:
                            correct += sum(self.sequence_outcomes[seq_id])
                            total += self.M
                    type_accuracy[seq_type] = correct / total if total > 0 else 0.0
                return type_accuracy
        except Exception as e:
            logger.error(f"Error calculating overall accuracy: {e}")
            return 0.0 if not by_type else {}
    
    def position_accuracy(self, by_type: bool = False, avg_by_type: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
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
        try:
            # Calculate by type first
            type_position_accuracy = {}
            for seq_type, seq_ids in self.type_sequences.items():
                # Collect all sequences of this type
                seqs = [self.sequence_outcomes[seq_id] for seq_id in seq_ids if seq_id in self.sequence_outcomes]
                if seqs:
                    type_position_accuracy[seq_type] = np.mean(np.array(seqs), axis=0)
                else:
                    type_position_accuracy[seq_type] = np.zeros(self.M)
            
            if by_type:
                return type_position_accuracy
            elif avg_by_type:
                # Average across types (giving equal weight to each type)
                if type_position_accuracy:
                    all_type_accs = np.array(list(type_position_accuracy.values()))
                    return np.mean(all_type_accs, axis=0)
                else:
                    return np.zeros(self.M)
            else:
                # Traditional overall average (all sequences weighted equally)
                if self.y_nm:
                    y_nm_array = np.array(self.y_nm)
                    return np.mean(y_nm_array, axis=0)
                else:
                    return np.zeros(self.M)
        except Exception as e:
            logger.error(f"Error calculating position accuracy: {e}")
            return np.zeros(self.M) if not by_type else {}
    
    def accuracy_slope(self, by_type: bool = False) -> Union[float, Dict[str, float]]:
        """
        Calculate the slope of the fitted accuracy curve (k).
        
        This metric measures how quickly the model learns within a sequence.
        Positive values indicate improvement over the sequence.
        
        Args:
            by_type (bool): Whether to calculate metrics by task type
            
        Returns:
            float or dict: Slope of the linear regression line, or results by type
        """
        try:
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
                    if len(acc_m) > 0 and not np.all(np.isnan(acc_m)):
                        try:
                            # Fit linear regression
                            model = LinearRegression()
                            model.fit(positions, acc_m.reshape(-1, 1))
                            type_slopes[seq_type] = float(model.coef_[0][0])
                        except Exception as e:
                            logger.warning(f"Failed to fit regression for type {seq_type}: {e}")
                            type_slopes[seq_type] = 0.0
                    else:
                        type_slopes[seq_type] = 0.0
                    
                return type_slopes
        except Exception as e:
            logger.error(f"Error calculating accuracy slope: {e}")
            return 0.0 if not by_type else {}
    
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
    
    def post_warmup_accuracy(self, K: int = None, by_type: bool = False) -> Union[float, Dict[str, float]]:
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
        K = K or CONFIG["default_warmup_k"]
        
        if K >= self.M:
            logger.warning(f"K ({K}) must be less than M ({self.M})")
            return 0.0 if not by_type else {t: 0.0 for t in self.type_sequences.keys()}
        
        try:
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
                        if seq_id in self.sequence_outcomes:
                            seq = self.sequence_outcomes[seq_id]
                            post_warmup = seq[K:]
                            correct += sum(post_warmup)
                            total += len(post_warmup)
                    type_post_warmup[seq_type] = correct / total if total > 0 else 0.0
                return type_post_warmup
        except Exception as e:
            logger.error(f"Error calculating post-warmup accuracy: {e}")
            return 0.0 if not by_type else {}
    
    def first_learned_correct_offset(self, by_type=False, avg_by_type=False):
        """
        Calculate the average offset of first learned correct solution (O_learned).
        
        This metric measures how quickly a model begins to learn within a sequence while
        discounting pre-existing knowledge. It identifies the first position where the model
        answered correctly AFTER the first position it answered incorrectly (compared to zero-shot),
        indicating genuine learning within the sequence.
        
        Lower values indicate faster learning within the feedback/demonstration context.
        If a model had no learning opportunities (zero-shot correct on all problems),
        the sequence is skipped.
        
        Args:
            by_type (bool): Whether to calculate metrics by task type
            avg_by_type (bool): If True, average across task types (equal weight per type)
            
        Returns:
            float or dict: Average offset of first learned correct solution, or results by type
        """
        if not self.zero_shot_correct_ids:
            logger.warning("Zero-shot results not loaded. First learned correct offset cannot be calculated.")
            return 0 if not by_type else {t: 0 for t in self.type_sequences.keys()}
        
        # Build mapping from item ID to sequence/position information
        id_to_seq_info = {}
        for item in self.results:
            item_id = item.get('id')
            seq_id = item.get('sequence_id')
            position = item.get('position_in_sequence')
            if item_id and seq_id and position:
                if seq_id not in id_to_seq_info:
                    id_to_seq_info[seq_id] = {}
                id_to_seq_info[seq_id][position] = item_id
        
        # Calculate by type first
        type_learned_offsets = {}
        for seq_type, seq_ids in self.type_sequences.items():
            type_learned_offsets[seq_type] = []
            
            for seq_id in seq_ids:
                if seq_id not in id_to_seq_info:
                    continue
                
                seq_items = id_to_seq_info[seq_id]
                seq_outcomes = self.sequence_outcomes.get(seq_id, [])
                
                # Find the first position where zero-shot was wrong (first_wrong_position)
                first_wrong_position = None
                for pos in range(1, self.M + 1):
                    item_id = seq_items.get(pos)
                    if item_id and item_id not in self.zero_shot_correct_ids:
                        first_wrong_position = pos
                        break
                
                # If all positions were zero-shot correct, skip this sequence
                if first_wrong_position is None:
                    continue
                
                # Convert to 0-based indexing for seq_outcomes
                idx_first_wrong = first_wrong_position - 1
                
                # Find the first position after first_wrong that was answered correctly
                learned_position = None
                for idx in range(idx_first_wrong, len(seq_outcomes)):
                    if seq_outcomes[idx] == 1:
                        learned_position = idx + 1  # Convert back to 1-based
                        break
                
                # If no correct answer found after first_wrong, use M+1
                if learned_position is None:
                    learned_position = self.M + 1
                
                # Calculate offset: position from which learning started
                offset = learned_position - first_wrong_position + 1
                type_learned_offsets[seq_type].append(offset)
        
        # Calculate average for each type (only for types with data)
        type_avg = {}
        for t, offsets in type_learned_offsets.items():
            if offsets:
                type_avg[t] = sum(offsets) / len(offsets)
            else:
                type_avg[t] = 0  # No learning opportunities in this type
        
        if by_type:
            return type_avg
        elif avg_by_type:
            # Calculate average across all types that have data (giving equal weight to each type)
            types_with_data = {t: v for t, v in type_avg.items() if v > 0}
            if not types_with_data:
                return 0
            return sum(types_with_data.values()) / len(types_with_data)
        else:
            # Calculate average across all sequences (traditional method)
            all_offsets = []
            for offsets in type_learned_offsets.values():
                all_offsets.extend(offsets)
            return sum(all_offsets) / len(all_offsets) if all_offsets else 0
    
    def generate_report(self, output_file: Optional[str] = None, warmup_k: int = None) -> Dict:
        """
        Generate a comprehensive report of all metrics.
        
        Args:
            output_file (str, optional): Path to save the report as JSON. If None, just return the metrics.
            warmup_k (int): Parameter K for post-warmup accuracy
            
        Returns:
            dict: Dictionary containing all metrics
        """
        warmup_k = warmup_k or CONFIG["default_warmup_k"]
        
        start_time = time.time()
        logger.info(f"Generating comprehensive evaluation report with warmup_k={warmup_k}")
        
        try:
            # Compute all metrics
            overall_metrics = self.compute_all_metrics(warmup_k, by_type=False)
            type_metrics = self.compute_all_metrics(warmup_k, by_type=True)
            
            # Format metrics for report with more descriptive names
            overall_metrics_dict = {
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
                    "values": overall_metrics["position_accuracy"].tolist() if hasattr(overall_metrics["position_accuracy"], 'tolist') else overall_metrics["position_accuracy"],
                    "description": "Accuracy at each position across all sequences"
                }
            }
            
            # Add learning offset metric if available
            if "first_learned_correct_offset" in overall_metrics:
                overall_metrics_dict["first_learned_correct_offset"] = {
                    "value": overall_metrics["first_learned_correct_offset"],
                    "description": "Average offset of first learned correct solution (lower is better - measures learning speed while discounting pre-existing knowledge)"
                }
            
            report = {
                "model_evaluation": {
                    "dataset_info": {
                        "num_sequences": self.N,
                        "problems_per_sequence": self.M,
                        "total_problems": self.N * self.M,
                        "sequence_types": {t: len(seq_ids) for t, seq_ids in self.type_sequences.items()},
                        "warmup_parameter": warmup_k,
                        "zero_shot_results_provided": bool(self.zero_shot_correct_ids)
                    },
                    "overall_metrics": overall_metrics_dict,
                    "type_metrics": {}
                }
            }
            
            # Add type-specific metrics to the report
            for task_type in sorted(self.type_sequences.keys()):
                if task_type in type_metrics["overall_accuracy"]:
                    type_metrics_data = {
                        "sequence_count": len(self.type_sequences[task_type]),
                        "overall_accuracy": type_metrics["overall_accuracy"][task_type],
                        "accuracy_slope": type_metrics["accuracy_slope"][task_type],
                        "first_correct_position": type_metrics["first_correct_position"][task_type],
                        "consecutive_correct": type_metrics["consecutive_correct"][task_type],
                        f"post_warmup_accuracy_k{warmup_k}": type_metrics[f"post_warmup_accuracy_k{warmup_k}"][task_type],
                        "position_accuracy": type_metrics["position_accuracy"][task_type].tolist() if hasattr(type_metrics["position_accuracy"][task_type], 'tolist') else type_metrics["position_accuracy"][task_type]
                    }
                    
                    # Add learning offset metric if available
                    if "first_learned_correct_offset" in type_metrics:
                        type_metrics_data["first_learned_correct_offset"] = type_metrics["first_learned_correct_offset"][task_type]
                    
                    report["model_evaluation"]["type_metrics"][task_type] = type_metrics_data
            
            # Save report if output file is provided
            if output_file:
                save_json(report, output_file)
            
            generation_time = time.time() - start_time
            logger.info(f"Report generation completed in {generation_time:.2f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Calculate EvaLearn metrics from evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_metric.py --results results.json --output report.json
  python evaluate_metric.py --results results.json --problems 5 --warmup 2
        """
    )
    parser.add_argument('--results', required=True, 
                       help='Path to the evaluation results JSON file')
    parser.add_argument('--zero-shot', type=str, default=None,
                       help='Path to zero-shot evaluation results (for learning offset metric)')
    parser.add_argument('--problems', type=int, default=CONFIG["default_problems_per_sequence"], 
                       help=f'Number of problems per sequence (default: {CONFIG["default_problems_per_sequence"]})')
    parser.add_argument('--warmup', type=int, default=CONFIG["default_warmup_k"], 
                       help=f'Warmup parameter K for post-warmup accuracy (default: {CONFIG["default_warmup_k"]})')
    parser.add_argument('--output', 
                       help='Path to save the report JSON file (default: auto-generated)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    start_time = time.time()
    logger.info("Starting EvaLearn metrics calculation")
    logger.info(f"Input file: {args.results}")
    logger.info(f"Problems per sequence: {args.problems}")
    logger.info(f"Warmup parameter: {args.warmup}")
    
    try:
        # Initialize metrics calculator
        metrics = EvaLearnMetrics(args.results, args.problems, args.zero_shot)
        
        # Generate output filename if not provided
        output_file = args.output or f"report_{os.path.basename(args.results).replace('.json', '')}.json"
        
        # Generate and save report
        report = metrics.generate_report(output_file=output_file, warmup_k=args.warmup)
        
        # Calculate overall metrics for console output
        overall_metrics = metrics.compute_all_metrics(args.warmup, by_type=False)
        type_metrics = metrics.compute_all_metrics(args.warmup, by_type=True)
        
        # Print comprehensive statistics
        print("\n" + "="*60)
        print("           EvaLearn Evaluation Report")
        print("="*60)
        print(f"Dataset Statistics:")
        print(f"  • Total Questions: {metrics.N * metrics.M:,}")
        print(f"  • Total Sequences: {metrics.N:,}")
        print(f"  • Problems per Sequence: {metrics.M}")
        print(f"  • Report saved to: {output_file}")
        
        print(f"\nOverall Performance Metrics:")
        print(f"  • Overall Accuracy: {overall_metrics['overall_accuracy']:.4f}")
        print(f"  • Learning Slope (k): {overall_metrics['accuracy_slope']:.6f}")
        print(f"  • Avg. First Correct Position: {overall_metrics['first_correct_position']:.2f}")
        print(f"  • Avg. Consecutive Correct: {overall_metrics['consecutive_correct']:.2f}")
        print(f"  • Post-warmup Accuracy (K={args.warmup}): {overall_metrics[f'post_warmup_accuracy_k{args.warmup}']:.4f}")
        
        print(f"\nPosition-wise Accuracy:")
        position_acc = overall_metrics['position_accuracy']
        for i in range(len(position_acc)):
            print(f"  • Position {i+1}: {position_acc[i]:.4f}")
        
        print(f"\nPerformance by Task Type:")
        for task_type in sorted(metrics.type_sequences.keys()):
            seq_count = len(metrics.type_sequences[task_type])
            if task_type in type_metrics['overall_accuracy']:
                print(f"\n  {task_type.upper()} (Sequences: {seq_count})")
                print(f"    ├─ Overall Accuracy: {type_metrics['overall_accuracy'][task_type]:.4f}")
                print(f"    ├─ Learning Slope: {type_metrics['accuracy_slope'][task_type]:.6f}")
                print(f"    ├─ First Correct Pos: {type_metrics['first_correct_position'][task_type]:.2f}")
                print(f"    ├─ Consecutive Correct: {type_metrics['consecutive_correct'][task_type]:.2f}")
                print(f"    └─ Post-warmup Acc: {type_metrics[f'post_warmup_accuracy_k{args.warmup}'][task_type]:.4f}")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Evaluation completed successfully in {total_time:.2f}s")
        print("="*60)
        
        logger.info("EvaLearn metrics calculation completed successfully")
        
    except FileNotFoundError:
        logger.error(f"Results file not found: {args.results}")
        return 1
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in results file: {args.results}")
        return 1
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        if args.verbose:
            import traceback
            logger.debug(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())