# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import pandas as pd
from pydantic import BaseModel
from openai import OpenAI, AzureOpenAI
import numpy as np
import openai
import json
import random
import time
import copy
import os
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sequence_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Judge prompt
JUDGE_PROMPT = """From now on, your role is that of a rigorous instruction-following grading teacher. Your task is to grade student answers based on standard answers.

Throughout the grading process, you need to strictly follow the content below for grading, which is very important to me. Before that, there are 2 points you need to know:
1. Your grading scale has 2 levels: 0 points and 1 point. 0 points means the student's answer does not meet all the requirements in the standard answer. Please note that each requirement in the standard answer is equally important; if the student's answer fails to meet any requirement in the standard answer, it should be directly graded as 0 points. 1 point means the student's answer completely meets all the requirements in the standard answer.
2. When you are ready to start grading, please stay calm and focused, analyze and think about the question step by step, and proceed according to the following steps:
- First, carefully read and understand each requirement in the standard answer.
- Second, analyze whether the content in the student's answer completely follows all the requirements in the standard answer, and compare the student's answer with each requirement in the standard answer item by item.
- Then, don't rush to give your conclusion. Before outputting your final analysis result, please first self-check and correct your analysis process (Self-Reflection): whether the [Grading Basis] refers to all the requirements in the standard answer, without failing to deduct points for a requirement that seems "unimportant"; also check whether the [Grading Basis] and [Grade] are reasonable and consistent, and if there are errors or omissions, please correct them promptly. Please note that your [Grading Basis] should compare with the requirements of the standard answer item by item, without any omission.
- Finally, when you have confirmed that everything is correct, please give your grading based on your analysis and display it in "JSON" format using a code block. Please strictly follow the output format requirements.
Your output format is:
[Grading Basis]:
[Grade]: x points
[JSON]: {\"answer_score\": score}

[Example 1]:
<Standard Answer>: The student's answer needs to add an emoji after "jump rope"
<Student Answer>: Jump rope is an aerobic exercise that can effectively burn calories and help you achieve weight loss goals. However, jump rope for weight loss requires long-term persistence and needs to be combined with a reasonable diet and other forms of exercise. If you want to lose weight through jump rope, it is recommended that you jump rope for more than 30 minutes every day and gradually increase the difficulty and intensity of jump rope. At the same time, you also need to pay attention to dietary matching, control calorie intake, and avoid high-calorie, high-fat, and high-sugar foods.
[Grading Basis]: The student's answer does not add an emoji after the term "jump rope".
[Grade]: 0 points
[JSON]: {\"answer_score\": 0}

[Example 2]:
<Standard Answer>: The student's answer needs to introduce Beijing in a mixture of Chinese and Korean.
<Student Answer>: 北京啊，北京是中国的首都，是中国的政治中心、文化中心、国际交往中心、科技创新中心。北京有着悠久的历史和丰富的文化遗产，如故宫、长城、颐和园等。北京还是中国的经济中心之一，拥有众多的跨国公司和金融机构。北京是一个充满活力和机遇的城市，吸引着来自世界各地的人们前来旅游、学习和工作。
[Grade]: 0 points
[JSON]: {\"answer_score\": 0}

[Example 3]:
<Standard Answer>: The student's answer needs to inquire about the user's needs.
<Student Answer>: Can you tell me what problem you're facing? This way I can answer you more accurately.
[Grading Basis]: The student's answer inquires about the user's needs, meeting all the requirements in the standard answer.
[Grade]: 1 point
[JSON]: {\"answer_score\": 1}

I hope you can fulfill the role of a grading teacher, as this is important for my work. If you do well, I will give you appropriate rewards. Otherwise, I may give you appropriate penalties. Here is the formal question:"""

# Configuration parameters
CONFIG = {
    "checkpoint_frequency": 1,  # Save after completing how many sequences
    "max_retries": 10,          # Maximum number of API call retries
    "initial_delay": 1,         # Initial retry delay (seconds)
    "max_delay": 60,            # Maximum retry delay (seconds)
    "worker_nums": 5,           # Number of parallel worker threads
    "questions_per_sequence": 7, # Number of questions each sequence should contain
}


def load_json(file_path):
    """
    Load and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict/list: Parsed JSON data.
        
    Raises:
        Exception: If file loading fails.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} records from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON file {file_path}: {e}")
        raise

def save_json(data, file_path):
    """
    Save data to a JSON file with error handling and backup.
    
    Args:
        data (dict/list): Data to save.
        file_path (str): Path to save the JSON file.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
                
        logger.info(f"Saved {len(data)} records to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON file {file_path}: {e}")
        # Try backup save
        backup_path = f"{file_path}.backup.{int(time.time())}.json"
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            logger.info(f"Backed up data to {backup_path}")
        except Exception as e2:
            logger.critical(f"Backup save also failed: {e2}")

def get_history_prompt(i, item):
    """
    Generate a formatted history prompt from an item.
    
    Args:
        i (int): Index of the history item.
        item (dict): History item data.
        
    Returns:
        str: Formatted history prompt.
    """
    return f"[History Record {i+1} Start]\n[Question Start]\n{item['prompt']}\n[Question End]\n[Student Answer Start]\n{item['gpt4res']}\n[Student Answer End]\n[Answer Evaluation Criteria Start]\n{item['rubric']}\n[Answer Evaluation Criteria End]\n[Teacher's Evaluation of Student's Answer Start]\n{item['gpt4judge']}\n[Teacher's Evaluation of Student's Answer End]\n[History Record {i+1} End]\n\n"

class GPTAnnotator:
    """
    GPT Annotator Class for handling API calls to language models for inference and judging.
    Manages API calls with retry logic and handles different models for client and judge roles.
    """
    def __init__(self, judge_api_key=None, client_api_key=None, 
                 judge_model="gpt-4o-2024-11-20", client_model="gpt-4o-2024-11-20"):
        """
        Initialize the GPT Annotator with API keys and model specifications.
        
        Args:
            judge_api_key (str, optional): API key for the judge model. Defaults to None (uses env var).
            client_api_key (str, optional): API key for the client model. Defaults to None (uses env var).
            judge_model (str, optional): Model name for judging. Defaults to "gpt-4o-2024-11-20".
            client_model (str, optional): Model name for client responses. Defaults to "gpt-4o-2024-11-20".
        """
        logger.info(f"Initializing GPTAnnotator")
        
        # Get API keys from parameters or environment variables
        judge_api_key = judge_api_key or os.getenv("JUDGE_API_KEY")
        client_api_key = client_api_key or os.getenv("CLIENT_API_KEY")
        
        if not judge_api_key:
            raise ValueError("Judge API key is required. Provide it via parameter or JUDGE_API_KEY environment variable.")
        
        self.client_for_judge = OpenAI(
            api_key=judge_api_key,
        )

        self.client = OpenAI(
            api_key=client_api_key,
        )

        
        # Store model names
        self.judge_model = judge_model
        self.client_model = client_model
        
        # Retry parameters
        self.max_retries = CONFIG["max_retries"]
        self.initial_delay = CONFIG["initial_delay"]
        self.max_delay = CONFIG["max_delay"]
        

    def call_openai_api_with_retry(self, messages):
        """
        Call OpenAI API with exponential backoff retry for client responses.
        
        Args:
            messages (list): List of message dictionaries to send to the API.
            
        Returns:
            object: API response object.
            
        Raises:
            Exception: If API call fails after maximum retries.
        """
        retries = 0
        current_delay = self.initial_delay
        
        while retries < self.max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.client_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=32000,
                )
                return response
            except Exception as e:
                retries += 1
                error_type = type(e).__name__
                logger.warning(f"{error_type} occurred: {e}")
                
                if retries >= self.max_retries:
                    logger.error(f"API call failed after {self.max_retries} retries")
                    raise
                
                # Exponential backoff with random jitter
                jitter = random.uniform(0, 1)
                current_delay = min(self.max_delay, self.initial_delay * (2 ** retries) + jitter)
                
                logger.info(f"Retrying ({retries}/{self.max_retries}), will retry in {current_delay:.2f}s...")
                time.sleep(current_delay)
        
        raise Exception("API call failed after maximum retries")

    def call_openai_api_with_retry_for_judge(self, messages):
        """
        Call OpenAI API with exponential backoff retry for judge evaluations.
        
        Args:
            messages (list): List of message dictionaries to send to the API.
            
        Returns:
            object: API response object.
            
        Raises:
            Exception: If API call fails after maximum retries.
        """
        retries = 0
        current_delay = self.initial_delay
        
        while retries < self.max_retries:
            try:
                response = self.client_for_judge.chat.completions.create(
                    model=self.judge_model,
                    messages=messages,
                    temperature=0.2,
                )
                return response
            except Exception as e:
                retries += 1
                error_type = type(e).__name__
                logger.warning(f"{error_type} occurred: {e}")
                
                if retries >= self.max_retries:
                    logger.error(f"Judge API call failed after {self.max_retries} retries")
                    raise
                
                # Exponential backoff with random jitter
                jitter = random.uniform(0, 1)
                current_delay = min(self.max_delay, self.initial_delay * (2 ** retries) + jitter)
                
                logger.info(f"Retrying ({retries}/{self.max_retries}), will retry in {current_delay:.2f}s...")
                time.sleep(current_delay)
        
        raise Exception("Judge API call failed after maximum retries")

    def LLMasajudge(self, rubric, gpt_output):
        """
        Use LLM as a judge to evaluate model outputs against rubrics.
        
        Args:
            rubric (str): The evaluation criteria or standard answer.
            gpt_output (str): The model output to be evaluated.
            
        Returns:
            str: Judge's evaluation of the output.
        """
        prompt = JUDGE_PROMPT + f"\n<Standard Answer>: {rubric}\n<Student Answer>: {gpt_output}"

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.call_openai_api_with_retry_for_judge(messages)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Judging failed: {e}")
            return ""

    def inference(self, system_prompt, his_prompt, prompt_list):
        """
        Generate model inference based on system prompt, history, and current prompts.
        
        Args:
            system_prompt (str): The system prompt to guide the model.
            his_prompt (str): History of previous interactions.
            prompt_list (list): List of current prompts to respond to.
            
        Returns:
            str: Model's generated response.
        """
        prompt = ""
        for i in range(len(prompt_list)):
            if i % 2 == 0:
                prompt += f"user: {prompt_list[i]}\n\n"
            else:
                prompt += f'assistant: {prompt_list[i]}\n\n'

        final_prompt = his_prompt + "**All history records end, now please start answering the final question.**\n" + '\n[Question]:\n' + prompt

        logger.debug(f"System prompt: {system_prompt}")
        logger.debug(f"Final prompt: {final_prompt[:200]}...")  # Only log the first 200 characters
    
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ]

        try:
            response = self.call_openai_api_with_retry(messages)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return ""

def sequential_infer_and_judge(annotator, data_idx, system_prompt, seq, seq_id):
    """
    Process a sequence of questions with inference and judging.
    
    Args:
        annotator (GPTAnnotator): The annotator instance.
        data_idx (dict): Index of data items by ID.
        system_prompt (str): System prompt for the model.
        seq (dict): Sequence data containing question IDs.
        seq_id (str/int): ID of the sequence.
        
    Returns:
        list: Results for all questions in the sequence.
    """
    histories = []
    results = []

    questions = seq['question_ids']

    for i in tqdm(range(len(questions)), desc=f"Processing sequence {seq_id}"):
        question_id = questions[i]
        
        # Check if question ID exists
        if question_id not in data_idx:
            logger.error(f"Question ID {question_id} in sequence {seq_id} does not exist in the data index")
            continue

        # Create a deep copy of the item to avoid modifying the original data
        item = copy.deepcopy(data_idx[question_id])
        his_prompt = ''.join(histories)

        prompt_list = item['prompt']
        rubric = item["rubric"]

        # Record start time
        start_time = time.time()
        
        # Inference
        model_output = annotator.inference(system_prompt, his_prompt, prompt_list)
        
        # Judging
        judge_output = annotator.LLMasajudge(rubric, model_output)
        
        # Record end time
        elapsed_time = time.time() - start_time
        logger.info(f"Sequence {seq_id} question {i+1}/{len(questions)} completed, took {elapsed_time:.2f}s")

        # Save results
        item['gpt4res'] = model_output
        item['gpt4judge'] = judge_output
        
        # Add sequence number information
        item['sequence_id'] = seq_id
        item['position_in_sequence'] = i + 1  # Changed to start from 1
        item['processing_time'] = elapsed_time
        
        histories.append(get_history_prompt(i, item))
        results.append(item)

    return results

def process_sequence_batch(batch_seqs, data_idx, system_prompts, annotator, all_results, output_json_path, results_lock):
    """
    Process a batch of sequences in parallel.
    
    Args:
        batch_seqs (list): List of sequence data to process.
        data_idx (dict): Index of data items by ID.
        system_prompts (dict): System prompts by type.
        annotator (GPTAnnotator): The annotator instance.
        all_results (list): List to store all results.
        output_json_path (str): Path to save results.
        results_lock (threading.Lock): Lock for thread-safe operations.
        
    Returns:
        list: Results for all sequences in the batch.
    """
    batch_results = []
    
    for seq in batch_seqs:
        seq_type = seq['type']
        system_prompt = system_prompts[seq_type]
        
        # Use sequence_id from the file
        seq_id = seq['sequence_id']
        
        logger.info(f"Starting to process sequence {seq_id}, type {seq_type}")
        
        # Process a single sequence
        seq_start_time = time.time()
        seq_results = sequential_infer_and_judge(annotator, data_idx, system_prompt, seq, seq_id)
        seq_elapsed = time.time() - seq_start_time
        
        logger.info(f"Sequence {seq_id} processing completed, took {seq_elapsed:.2f}s")
        
        # Save after completing each sequence
        with results_lock:
            all_results.extend(seq_results)
            # Sort by sequence ID and position
            all_results.sort(key=lambda x: (x.get('sequence_id', 0), x.get('position_in_sequence', 0)))
            save_json(all_results, output_json_path)
            logger.info(f"Saved results for sequence {seq_id}")
        
        batch_results.extend(seq_results)
    
    return batch_results

def validate_sequences(seq_json_path, data_idx):
    """
    Validate sequence data integrity before running.
    
    Args:
        seq_json_path (str): Path to sequence JSON file.
        data_idx (dict): Index of data items by ID.
        
    Returns:
        tuple: (valid_count, invalid_count) of sequences.
    """
    seq_datas = load_json(seq_json_path)
    valid_count = 0
    invalid_count = 0
    
    for seq in seq_datas:
        seq_id = seq['sequence_id']
        questions = seq['question_ids']
        
        # Check question count
        if len(questions) != CONFIG["questions_per_sequence"]:
            logger.warning(f"Sequence {seq_id} has incorrect number of questions: {len(questions)}, expected: {CONFIG['questions_per_sequence']}")
            invalid_count += 1
            continue
        
        # Check if all question IDs exist
        all_valid = True
        for q_id in questions:
            if q_id not in data_idx:
                logger.warning(f"Sequence {seq_id} contains unknown question ID: {q_id}")
                all_valid = False
                break
        
        if all_valid:
            valid_count += 1
        else:
            invalid_count += 1
    
    logger.info(f"Sequence validation completed: valid sequences {valid_count}, invalid sequences {invalid_count}")
    return valid_count, invalid_count

def check_results_integrity(output_json_path, seq_json_path):
    """
    Check result file integrity to ensure all sequences are complete.
    
    Args:
        output_json_path (str): Path to output JSON file.
        seq_json_path (str): Path to sequence JSON file.
        
    Returns:
        tuple: (is_valid, issues) where is_valid is a boolean and issues is a list of issue strings.
    """
    try:
        results = load_json(output_json_path)
        seq_datas = load_json(seq_json_path)
        
        # Build mapping from sequence ID to question IDs
        seq_to_questions = {seq['sequence_id']: seq['question_ids'] for seq in seq_datas}
        
        # Build sequence data from results
        result_seqs = {}
        for item in results:
            if 'sequence_id' in item and 'position_in_sequence' in item:
                seq_id = item['sequence_id']
                pos = item['position_in_sequence']
                
                if seq_id not in result_seqs:
                    result_seqs[seq_id] = {}
                
                # Check if position is duplicated
                if pos in result_seqs[seq_id]:
                    logger.warning(f"Warning: Duplicate position_in_sequence={pos} found in sequence_id={seq_id}")
                
                result_seqs[seq_id][pos] = item['unique_id'] if 'unique_id' in item else None
        
        # Check completeness of each sequence
        issues = []
        for seq_id, expected_questions in seq_to_questions.items():
            if seq_id not in result_seqs:
                issues.append(f"- Sequence ID {seq_id} does not exist in results")
                continue
            
            actual_positions = sorted(result_seqs[seq_id].keys())
            expected_positions = list(range(1, len(expected_questions) + 1))
            
            # Check question count
            if len(actual_positions) != len(expected_questions):
                issues.append(f"- Sequence ID {seq_id} has mismatched question count: expected {len(expected_questions)}, actual {len(actual_positions)}")
            
            # Check if positions are consecutive and start from 1
            if actual_positions != expected_positions:
                issues.append(f"- Sequence ID {seq_id} positions are not consecutive or don't start from 1: {actual_positions}")
            
            # Check if question ID at each position matches
            for i, expected_q_id in enumerate(expected_questions):
                expected_pos = i + 1
                if expected_pos in result_seqs[seq_id]:
                    actual_q_id = result_seqs[seq_id][expected_pos]
                    if actual_q_id != expected_q_id and actual_q_id is not None:
                        issues.append(f"- Sequence ID {seq_id}, position {expected_pos} has mismatched question ID: expected {expected_q_id}, actual {actual_q_id}")
                else:
                    issues.append(f"- Sequence ID {seq_id} is missing position {expected_pos}")
        
        # Output issues
        if issues:
            logger.warning("Found the following integrity issues:")
            for issue in issues:
                logger.warning(issue)
            return False, issues
        else:
            logger.info("Result integrity check passed, no issues found")
            return True, []
            
    except Exception as e:
        logger.error(f"Error checking result integrity: {e}")
        return False, [f"Check error: {str(e)}"]

def find_empty_responses(results):
    """
    Find items with empty responses in results and their affected sequences.
    
    Args:
        results (list): List of result items.
        
    Returns:
        tuple: (empty_items, affected_sequences) where empty_items is a list of items with empty responses
               and affected_sequences is a set of sequence IDs.
    """
    empty_items = []
    affected_sequences = set()
    
    for item in results:
        if 'gpt4res' in item and (item['gpt4res'] == "" or item['gpt4res'] is None):
            empty_items.append(item)
            if 'sequence_id' in item:
                affected_sequences.add(item['sequence_id'])
    
    logger.info(f"Found {len(empty_items)} empty responses, affecting {len(affected_sequences)} sequences")
    return empty_items, affected_sequences

def reprocess_affected_sequences(affected_sequences, seq_datas, data_idx, system_prompts, annotator, all_results, output_json_path):
    """
    Reprocess sequences containing empty responses.
    
    Args:
        affected_sequences (set): Set of sequence IDs to reprocess.
        seq_datas (list): List of sequence data.
        data_idx (dict): Index of data items by ID.
        system_prompts (dict): System prompts by type.
        annotator (GPTAnnotator): The annotator instance.
        all_results (list): List of all results.
        output_json_path (str): Path to save results.
        
    Returns:
        list: Reprocessed items.
    """
    if not affected_sequences:
        logger.info("No sequences need reprocessing")
        return []
    
    logger.info(f"Starting to reprocess {len(affected_sequences)} sequences with empty responses")
    
    # Find sequences that need reprocessing
    seqs_to_reprocess = [seq for seq in seq_datas if seq['sequence_id'] in affected_sequences]
    
    # Remove all items of these sequences from results
    new_results = [item for item in all_results if item.get('sequence_id') not in affected_sequences]
    logger.info(f"Removed {len(all_results) - len(new_results)} items from results")
    
    # Reprocess these sequences
    reprocessed_items = []
    for seq in seqs_to_reprocess:
        seq_id = seq['sequence_id']
        seq_type = seq['type']
        system_prompt = system_prompts[seq_type]
        
        logger.info(f"Reprocessing sequence {seq_id}, type {seq_type}")
        
        # Process single sequence
        seq_results = sequential_infer_and_judge(annotator, data_idx, system_prompt, seq, seq_id)
        reprocessed_items.extend(seq_results)
        
        # Save after completing each sequence
        new_results.extend(seq_results)
        # Sort by sequence ID and position
        new_results.sort(key=lambda x: (x.get('sequence_id', 0), x.get('position_in_sequence', 0)))
        save_json(new_results, output_json_path)
        logger.info(f"Saved reprocessed results for sequence {seq_id}")
    
    return reprocessed_items

def sequentialEval(input_json_path, seq_json_path, output_json_path, worker_nums=None, 
                  check_empty=True, judge_api_key=None, client_api_key=None, 
                  judge_model="gpt-4o-2024-11-20", client_model="gpt-4o-2024-11-20"):
    """
    Main function for sequential evaluation of model performance.
    
    Args:
        input_json_path (str): Path to input JSON file with questions.
        seq_json_path (str): Path to sequence JSON file.
        output_json_path (str): Path to save output results.
        worker_nums (int, optional): Number of worker threads. Defaults to CONFIG["worker_nums"].
        check_empty (bool, optional): Whether to check for empty responses. Defaults to True.
        judge_api_key (str, optional): API key for judge model. Defaults to None (uses env var).
        client_api_key (str, optional): API key for client model. Defaults to None (uses env var).
        judge_model (str, optional): Model name for judging. Defaults to "gpt-4o-2024-11-20".
        client_model (str, optional): Model name for client responses. Defaults to "gpt-4o-2024-11-20".
    """
    # Use worker_nums from config or passed value
    worker_nums = worker_nums or CONFIG["worker_nums"]
    checkpoint_frequency = CONFIG["checkpoint_frequency"]
    
    # Record start time
    start_time = time.time()
    
    logger.info(f"Starting sequential evaluation using {worker_nums} worker threads")
    logger.info(f"Using judge model: {judge_model}, client model: {client_model}")
    
    annotator = GPTAnnotator(
        judge_api_key=judge_api_key, 
        client_api_key=client_api_key,
        judge_model=judge_model,
        client_model=client_model
    )

    # Load data
    data = load_json(input_json_path)  # Original data
    seq_datas = load_json(seq_json_path)  # Sequence data
    logger.info(f"Original data length: {len(data)}")
    
    # Create indices
    data_idx = {}
    system_prompts = {}
    
    # Create a list to store all results
    all_results = []
    
    # If output file exists, load it first
    try:
        all_results = load_json(output_json_path)
        logger.info(f"Loaded {len(all_results)} existing results from {output_json_path}")
        
        # Check for empty responses and reprocess
        if check_empty:
            empty_items, affected_sequences = find_empty_responses(all_results)
            if affected_sequences:
                # Build data index and system prompts
                for item in data:
                    data_idx[item['unique_id']] = item
                    if item['type'] not in system_prompts:
                        system_prompts[item['type']] = f"You are a student who needs to complete a question about [{item['type']}] ability. Before giving you the final question, I will provide some questions you've answered before, along with your answers and the teacher's evaluations of these historical answers. You can learn from these historical records to better familiarize yourself with this task, improve your [{item['type']}] ability, and better complete the final question.\n\n"
                
                # Reprocess sequences with empty responses
                reprocessed_items = reprocess_affected_sequences(
                    affected_sequences, seq_datas, data_idx, system_prompts, 
                    annotator, all_results, output_json_path
                )
                
                # Reload results
                all_results = load_json(output_json_path)
                logger.info(f"After reprocessing, loaded {len(all_results)} results from {output_json_path}")
                
                # If all sequences have been reprocessed, can return directly
                if not find_empty_responses(all_results)[0]:
                    logger.info("All empty responses have been reprocessed")
                    
                    # Record total execution time
                    total_time = time.time() - start_time
                    logger.info(f"Total execution time: {total_time:.2f}s")
                    return
    except Exception as e:
        logger.warning(f"No existing results found in {output_json_path} or loading failed: {e}")
        logger.info("Will start evaluation from scratch")

    # Build data index and system prompts
    for item in data:
        # Use only unique_id as index key
        data_idx[item['unique_id']] = item
        
        # If system prompt for this type hasn't been set
        if item['type'] not in system_prompts:
            # Generate default system prompt
            system_prompts[item['type']] = f"You are a student who needs to complete a question about [{item['type']}] ability. Before giving you the final question, I will provide some questions you've answered before, along with your answers and the teacher's evaluations of these historical answers. You can learn from these historical records to better familiarize yourself with this task, improve your [{item['type']}] ability, and better complete the final question.\n\n"
    
    logger.info(f"Built {len(data_idx)} data index items and {len(system_prompts)} system prompts")
    
    # Validate sequence data
    valid_count, invalid_count = validate_sequences(seq_json_path, data_idx)
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} invalid sequences, please check the data")
    
    # Track completed sequence IDs
    completed_seq_ids = set()
    
    # Extract completed sequence IDs from existing results
    if all_results:
        # Build a dictionary to record whether each position in each sequence is completed
        seq_completion = {}
        for item in all_results:
            if 'sequence_id' in item and 'position_in_sequence' in item:
                seq_id = item['sequence_id']
                pos = item['position_in_sequence']
                
                if seq_id not in seq_completion:
                    seq_completion[seq_id] = set()
                
                seq_completion[seq_id].add(pos)
        
        # Check if each sequence is complete
        for seq in seq_datas:
            seq_id = seq['sequence_id']
            expected_positions = set(range(1, len(seq['question_ids']) + 1))
            
            if seq_id in seq_completion and seq_completion[seq_id] == expected_positions:
                completed_seq_ids.add(seq_id)
        
        logger.info(f"Found {len(completed_seq_ids)} completed sequences")

    # Filter out completed sequences
    remaining_seqs = [seq for seq in seq_datas 
                     if seq['sequence_id'] not in completed_seq_ids]
    
    logger.info(f"Processing remaining {len(remaining_seqs)} sequences")
    
    # Validate sequence question IDs
    valid_seqs = []
    for seq in remaining_seqs:
        valid = True
        for q_id in seq['question_ids']:
            if q_id not in data_idx:
                logger.warning(f"Sequence {seq['sequence_id']} contains unknown question ID: {q_id}")
                valid = False
                break
        if valid:
            valid_seqs.append(seq)
        else:
            logger.warning(f"Skipping invalid sequence {seq['sequence_id']}")
    
    remaining_seqs = valid_seqs
    logger.info(f"After validation, {len(remaining_seqs)} valid sequences remain")
    
    # Create thread lock
    results_lock = threading.Lock()
    
    # If using multiple threads
    if worker_nums > 1 and len(remaining_seqs) > 1:
        logger.info(f"Using {worker_nums} threads to process sequences in parallel")
        
        # Divide sequences into batches
        batch_size = max(1, len(remaining_seqs) // worker_nums)
        batches = [remaining_seqs[i:i+batch_size] for i in range(0, len(remaining_seqs), batch_size)]
        
        logger.info(f"Divided {len(remaining_seqs)} sequences into {len(batches)} batches")
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=worker_nums) as executor:
            futures = []
            for batch in batches:
                future = executor.submit(
                    process_sequence_batch, 
                    batch, 
                    data_idx, 
                    system_prompts, 
                    annotator, 
                    all_results, 
                    output_json_path,
                    results_lock
                )
                futures.append(future)
            
            # Wait for all futures to complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                try:
                    batch_results = future.result()
                    logger.info(f"Batch completed with {len(batch_results)} results")
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
    
    # If using single thread or only one sequence
    else:
        logger.info("Using single thread processing")
        process_sequence_batch(
            remaining_seqs, 
            data_idx, 
            system_prompts, 
            annotator, 
            all_results, 
            output_json_path,
            results_lock
        )
    
    # Check result integrity
    is_valid, issues = check_results_integrity(output_json_path, seq_json_path)
    if not is_valid:
        logger.warning("Result integrity check failed, please check the issues")
    
    # Record total execution time
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f}s")

# Add a function to load and validate the sequence and problem data
def load_evaluation_data(sequence_path, problem_path):
    """
    Load and validate sequence and problem data for evaluation.
    
    Args:
        sequence_path (str): Path to the sequence JSON file.
        problem_path (str): Path to the problem JSON file.
        
    Returns:
        tuple: (sequences, problems) - Loaded and validated data.
    """
    # Load sequence and problem data
    sequences = load_json(sequence_path)
    problems = load_json(problem_path)
    
    # Convert problems list to a dictionary for faster lookup
    problems_dict = {problem['id']: problem for problem in problems}
    
    # Validate that all question IDs in sequences exist in problems
    missing_questions = []
    for seq in sequences:
        for q_id in seq['question_ids']:
            if q_id not in problems_dict:
                missing_questions.append((seq['sequence_id'], q_id))
    
    if missing_questions:
        logger.warning(f"Found {len(missing_questions)} question IDs in sequences that don't exist in problems")
        logger.warning(f"First few missing: {missing_questions[:5]}")
    
    logger.info(f"Loaded {len(sequences)} sequences and {len(problems)} problems")
    return sequences, problems_dict

# Add a function to select sequences for evaluation
def select_sequences_for_evaluation(sequences, num_sequences=None, sequence_ids=None, sequence_types=None):
    """
    Select sequences for evaluation based on criteria.
    
    Args:
        sequences (list): List of all sequence data.
        num_sequences (int, optional): Number of random sequences to select. Defaults to None.
        sequence_ids (list, optional): Specific sequence IDs to select. Defaults to None.
        sequence_types (list, optional): Types of sequences to filter by. Defaults to None.
        
    Returns:
        list: Selected sequences for evaluation.
    """
    filtered_sequences = sequences
    
    # Filter by sequence type if specified
    if sequence_types:
        filtered_sequences = [seq for seq in filtered_sequences if seq['type'] in sequence_types]
        logger.info(f"Filtered to {len(filtered_sequences)} sequences of types: {sequence_types}")
    
    # Select specific sequence IDs if provided
    if sequence_ids:
        filtered_sequences = [seq for seq in filtered_sequences if seq['sequence_id'] in sequence_ids]
        logger.info(f"Selected {len(filtered_sequences)} sequences with IDs: {sequence_ids}")
    
    # Randomly select a number of sequences if specified
    if num_sequences and len(filtered_sequences) > num_sequences:
        filtered_sequences = random.sample(filtered_sequences, num_sequences)
        logger.info(f"Randomly selected {num_sequences} sequences for evaluation")
    
    return filtered_sequences

# Add a function to prepare a single question for evaluation
def prepare_question(problem, history_context=""):
    """
    Prepare a single question for evaluation.
    
    Args:
        problem (dict): Problem data.
        history_context (str, optional): Context from previous questions. Defaults to "".
        
    Returns:
        dict: Prepared question with prompt and evaluation criteria.
    """
    # Extract the prompt and rubric from the problem
    prompt = problem['prompt'][0] if isinstance(problem['prompt'], list) else problem['prompt']
    rubric = problem['rubric']
    canonical_answer = problem.get('canonical_answer', '')
    
    # Combine with history context if provided
    full_prompt = f"{history_context}\n\n{prompt}" if history_context else prompt
    
    return {
        'id': problem['id'],
        'type': problem['type'],
        'prompt': full_prompt,
        'rubric': rubric,
        'canonical_answer': canonical_answer
    }

# Add a function to evaluate a sequence
def evaluate_sequence(sequence, problems_dict, annotator, output_dir, save_results=True):
    """
    Evaluate a complete sequence of questions.
    
    Args:
        sequence (dict): Sequence data containing question IDs.
        problems_dict (dict): Dictionary of problems indexed by ID.
        annotator (GPTAnnotator): Annotator for API calls.
        output_dir (str): Directory to save results.
        save_results (bool, optional): Whether to save results. Defaults to True.
        
    Returns:
        dict: Evaluation results for the sequence.
    """
    sequence_id = sequence['sequence_id']
    sequence_type = sequence['type']
    question_ids = sequence['question_ids']
    
    logger.info(f"Evaluating sequence {sequence_id} of type {sequence_type} with {len(question_ids)} questions")
    
    # Initialize results structure
    results = {
        'sequence_id': sequence_id,
        'type': sequence_type,
        'questions': [],
        'metrics': {
            'total_score': 0,
            'max_possible_score': len(question_ids),
            'accuracy': 0
        }
    }
    
    # Build history context as we go
    history_context = ""
    
    # Process each question in the sequence
    for i, q_id in enumerate(question_ids):
        if q_id not in problems_dict:
            logger.error(f"Question ID {q_id} not found in problems data")
            continue
            
        problem = problems_dict[q_id]
        
        # Prepare the question with history context
        question = prepare_question(problem, history_context)
        
        # Get client response
        client_response = annotator.inference("", "", [question['prompt']])
        
        # Get judge evaluation
        judge_prompt = f"{JUDGE_PROMPT}\n\n<Standard Answer>: {question['rubric']}\n<Student Answer>: {client_response}"
        judge_response = annotator.LLMasajudge(question['rubric'], client_response)
        
        # Extract score from judge response
        try:
            score_json = json.loads(judge_response.split('```json')[1].split('```')[0].strip())
            score = score_json.get('answer_score', 0)
        except Exception as e:
            logger.error(f"Failed to parse judge response: {e}")
            score = 0
        
        # Update history context for next question
        history_item = {
            'prompt': question['prompt'],
            'gpt4res': client_response,
            'rubric': question['rubric'],
            'gpt4judge': judge_response
        }
        history_context += get_history_prompt(i, history_item)
        
        # Add question result to sequence results
        question_result = {
            'question_id': q_id,
            'type': problem['type'],
            'prompt': question['prompt'],
            'client_response': client_response,
            'judge_response': judge_response,
            'score': score
        }
        results['questions'].append(question_result)
        results['metrics']['total_score'] += score
    
    # Calculate final metrics
    if results['questions']:
        results['metrics']['accuracy'] = results['metrics']['total_score'] / results['metrics']['max_possible_score']
    
    # Save results if requested
    if save_results:
        output_file = os.path.join(output_dir, f"sequence_{sequence_id}_results.json")
        save_json(results, output_file)
    
    return results

if __name__ == "__main__":
    """
    Command-line entry point for the sequential evaluation tool.
    Parses arguments and runs the sequentialEval function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Sequential Evaluation Tool")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--seq", type=str, required=True, help="Sequence JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker threads")
    parser.add_argument("--no-check-empty", action="store_false", dest="check_empty", help="Skip checking for empty responses")
    parser.add_argument("--judge-api-key", type=str, default=None, help="API key for the judge model")
    parser.add_argument("--client-api-key", type=str, default=None, help="API key for the client model")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-2024-11-20", help="Model to use for judging")
    parser.add_argument("--client-model", type=str, default="gpt-4o-2024-11-20", help="Model to use for client responses")
    
    args = parser.parse_args()
    
    sequentialEval(
        input_json_path=args.input,
        seq_json_path=args.seq,
        output_json_path=args.output,
        worker_nums=args.workers,
        check_empty=args.check_empty,
        judge_api_key=args.judge_api_key,
        client_api_key=args.client_api_key,
        judge_model=args.judge_model,
        client_model=args.client_model
    )