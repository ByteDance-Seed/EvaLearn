# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import json
import random
import time
import copy
import os
import logging
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("zero_shot_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Judge prompt
JUDGE_PROMPT = """From now on, your role is a rigorous, instruction-following grading teacher. Your task is to grade the student's answer according to the Rubric.

You must strictly follow the guidelines below throughout the grading process; this is very important. Before you begin, please note two points:
1. You have only two score levels: 0 points and 1 point. 0 points means the student's answer fails to meet any of the requirements in the Rubric. Every requirement in the Rubric is equally important—if the student's answer misses even one requirement, assign 0 points. 1 point means the student's answer fully satisfies all the requirements.
2. When you are ready to grade, remain calm and focused. Analyze and think through the question step by step, following these steps:
   - First, carefully read and understand each requirement in the Rubric.
   - Second, examine whether the student's answer fully complies with every requirement, comparing them one by one.
   - Third, do not rush to your conclusion. Before outputting your final analysis, please review and correct your reasoning. 
【Scoring Rationale】:refers to all requirements, and that you haven't skipped any because they seemed unimportant. Also check that your 【Scoring Rationale】 and the 【Score】are consistent; if you find errors or omissions, fix them. Your "Scoring Rationale" must address each requirement in the Rubric without exception.
   - Finally, when you are certain, provide your score and present it in a code block formatted as JSON. Adhere exactly to this output format.

Your output format should be:
【Scoring Rationale】：
【Score】：x points
【JSON】：{\"answer_score\": score}

【Example 1】：
<Rubric>: The student's answer must include an emoji immediately after "jump rope".
<Student Answer>: Jumping rope is an aerobic exercise that can effectively burn calories and help you achieve weight loss. However, losing weight through jumping rope requires long-term consistency and should be combined with a balanced diet and other forms of exercise. If you want to lose weight by jumping rope, it is recommended that you jump rope for at least 30 minutes every day and gradually increase the difficulty and intensity. At the same time, you should pay attention to your dietary choices, control calorie intake, and avoid consuming high-calorie, high-fat, or high-sugar foods.
【Scoring Rationale】: The student did not include an emoji after "jump rope" .  
【Score】: 0 points  
【JSON】: {\"answer_score\": 0}

【Example 2】:  
<Rubric>: The student's answer must introduce Beijing using a mix of Chinese and Korean.  
<Student Answer>: 北京啊，北京是中国的首都，是中国的政治中心、文化中心、国际交往中心、科技创新中心。北京有着悠久的历史和丰富的文化遗产，如故宫、长城、颐和园等。北京还是中国的经济中心之一，拥有众多的跨国公司和金融机构。北京是一个充满活力和机遇的城市，吸引着来自世界各地的人们前来旅游、学习和工作。
【Scoring Rationale】: The student's answer uses only Chinese and did not mix in Korean.  
【Score】: 0 points  
【JSON】:{\"answer_score\": 0}

【Example 3】:  
<Rubric>: The student's answer must ask the user for their needs.  
<Student Answer>: Can you tell me what problem you're encountering so that I can answer you more accurately?
【Scoring Rationale】: The student did ask about the user's needs, satisfying all requirements.  
【Score】: 1 point  
【JSON】: {\"answer_score\": 1}

I expect you to fulfill the role of exam-grading teacher—this is very important to my work. If you perform well, I will reward you; otherwise, I may penalize you. Below is the formal question to grade:"""

# Configuration parameters
CONFIG = {
    "max_retries": 10,
    "initial_delay": 1,
    "max_delay": 60,
    "worker_nums": 5,
    "questions_per_sequence": 7,
}

def load_json(file_path):
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} records from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON file {file_path}: {e}")
        raise

def save_json(data, file_path):
    """Save data to a JSON file with error handling and backup."""
    def _write_json(path, data):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    
    try:
        _write_json(file_path, data)
        logger.info(f"Saved {len(data)} records to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON file {file_path}: {e}")
        backup_path = f"{file_path}.backup.{int(time.time())}.json"
        try:
            _write_json(backup_path, data)
            logger.info(f"Backed up data to {backup_path}")
        except Exception as e2:
            logger.critical(f"Backup save also failed: {e2}")

class GPTAnnotator:
    """GPT Annotator Class for handling API calls to language models."""
    
    def __init__(self, judge_api_key=None, client_api_key=None, 
                 judge_model="gpt-4o-2024-11-20", client_model="gpt-4o-2024-11-20",
                 judge_api_base_url=None, client_api_base_url=None):
        logger.info("Initializing GPTAnnotator")
        
        # Get API keys and URLs from parameters or environment
        judge_api_key = judge_api_key or os.getenv("JUDGE_API_KEY")
        client_api_key = client_api_key or os.getenv("CLIENT_API_KEY")
        judge_api_base_url = judge_api_base_url or os.getenv("JUDGE_API_BASE_URL")
        client_api_base_url = client_api_base_url or os.getenv("CLIENT_API_BASE_URL")
        
        if not judge_api_key:
            raise ValueError("Judge API key is required. Provide it via parameter or JUDGE_API_KEY environment variable.")
        
        # Initialize clients
        http_client = httpx.Client(follow_redirects=True)
        
        self.client_for_judge = OpenAI(
            base_url=judge_api_base_url,
            api_key=judge_api_key,
            http_client=http_client,
        ) if judge_api_base_url else OpenAI(api_key=judge_api_key)
        
        self.client = OpenAI(
            base_url=client_api_base_url,
            api_key=client_api_key,
            http_client=http_client,
        ) if client_api_base_url else OpenAI(api_key=client_api_key)
        
        self.judge_model = judge_model
        self.client_model = client_model

    def _call_api_with_retry(self, client, model, messages, **kwargs):
        """Generic API call with retry logic."""
        retries = 0
        current_delay = CONFIG["initial_delay"]
        
        while retries < CONFIG["max_retries"]:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                return response
            except Exception as e:
                retries += 1
                logger.warning(f"{type(e).__name__} occurred: {e}")
                
                if retries >= CONFIG["max_retries"]:
                    logger.error(f"API call failed after {CONFIG['max_retries']} retries")
                    raise
                
                jitter = random.uniform(0, 1)
                current_delay = min(CONFIG["max_delay"], CONFIG["initial_delay"] * (2 ** retries) + jitter)
                logger.info(f"Retrying ({retries}/{CONFIG['max_retries']}), will retry in {current_delay:.2f}s...")
                time.sleep(current_delay)

    def LLMasajudge(self, rubric_zh, gpt_output):
        """Use LLM as a judge to evaluate model outputs."""
        prompt = JUDGE_PROMPT + f"\n<Standard Answer>: {rubric_zh}\n<Student Answer>: {gpt_output}"
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
        
        try:
            response = self._call_api_with_retry(
                self.client_for_judge, 
                self.judge_model, 
                messages, 
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Judging failed: {e}")
            return ""

    def inference(self, system_prompt, prompt_list):
        """Generate model inference without history context (zero-shot)."""
        prompt = ""
        for i in range(len(prompt_list)):
            if i % 2 == 0:
                prompt += f"user: {prompt_list[i]}\n\n"
            else:
                prompt += f'assistant: {prompt_list[i]}\n\n'

        final_prompt = '\n[Question]:\n' + prompt
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ]

        try:
            response = self._call_api_with_retry(
                self.client, 
                self.client_model, 
                messages, 
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return ""

def zero_shot_infer_and_judge(annotator, data_idx, system_prompt, seq, seq_id):
    """Process a sequence of questions without any history context (zero-shot)."""
    results = []
    questions = seq['question_ids']

    for i in tqdm(range(len(questions)), desc=f"Processing sequence {seq_id}"):
        question_id = questions[i]
        
        if question_id not in data_idx:
            logger.error(f"Question ID {question_id} in sequence {seq_id} does not exist")
            continue

        item = copy.deepcopy(data_idx[question_id])
        prompt_list = item['prompt']
        rubric_zh = item["rubric_zh"]

        start_time = time.time()
        
        # Inference and judging (no history context)
        model_output = annotator.inference(system_prompt, prompt_list)
        judge_output = annotator.LLMasajudge(rubric_zh, model_output)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Sequence {seq_id} question {i+1}/{len(questions)} completed, took {elapsed_time:.2f}s")

        # Save results
        item.update({
            'gpt4res': model_output,
            'gpt4judge': judge_output,
            'sequence_id': seq_id,
            'position_in_sequence': i + 1,
            'processing_time': elapsed_time
        })
        
        results.append(item)

    return results

def process_sequence_batch(batch_seqs, data_idx, system_prompts, annotator, all_results, output_json_path, results_lock):
    """Process a batch of sequences in parallel."""
    batch_results = []
    
    for seq in batch_seqs:
        seq_type = seq['type']
        system_prompt = system_prompts[seq_type]
        seq_id = seq['sequence_id']
        
        logger.info(f"Starting to process sequence {seq_id}, type {seq_type}")
        
        seq_start_time = time.time()
        seq_results = zero_shot_infer_and_judge(annotator, data_idx, system_prompt, seq, seq_id)
        seq_elapsed = time.time() - seq_start_time
        
        logger.info(f"Sequence {seq_id} processing completed, took {seq_elapsed:.2f}s")
        
        # Save after completing each sequence
        with results_lock:
            all_results.extend(seq_results)
            all_results.sort(key=lambda x: (x.get('sequence_id', 0), x.get('position_in_sequence', 0)))
            save_json(all_results, output_json_path)
            logger.info(f"Saved results for sequence {seq_id}")
        
        batch_results.extend(seq_results)
    
    return batch_results

def find_empty_responses(results):
    """Find items with empty responses in results."""
    empty_items = []
    affected_sequences = set()
    
    for item in results:
        if 'gpt4res' in item and (item['gpt4res'] == "" or item['gpt4res'] is None):
            empty_items.append(item)
            if 'sequence_id' in item:
                affected_sequences.add(item['sequence_id'])
    
    logger.info(f"Found {len(empty_items)} empty responses, affecting {len(affected_sequences)} sequences")
    return empty_items, affected_sequences

def zeroShotEval(input_json_path, seq_json_path, output_json_path, worker_nums=None, 
                check_empty=True, judge_api_key=None, client_api_key=None, 
                judge_model="gpt-4o-2024-11-20", client_model="gpt-4o-2024-11-20",
                judge_api_base_url=None, client_api_base_url=None):
    """Main function for zero-shot evaluation of model performance."""
    worker_nums = worker_nums or CONFIG["worker_nums"]
    start_time = time.time()
    
    logger.info(f"Starting zero-shot evaluation using {worker_nums} worker threads")
    logger.info(f"Using judge model: {judge_model}, client model: {client_model}")
    
    annotator = GPTAnnotator(
        judge_api_key=judge_api_key, 
        client_api_key=client_api_key,
        judge_model=judge_model,
        client_model=client_model,
        judge_api_base_url=judge_api_base_url,
        client_api_base_url=client_api_base_url
    )

    # Load data
    data = load_json(input_json_path)
    seq_datas = load_json(seq_json_path)
    logger.info(f"Original data length: {len(data)}")
    
    # Create indices
    data_idx = {item['id']: item for item in data}
    system_prompts = {
        item['type']: f"You are a student. You need to complete a question on [{item['type']}] skills. Please answer the following question to the best of your ability.\n\nRemember:\n1. Carefully understand the requirements of each question.\n2. Ensure your answer fully meets all requirements.\n3. Provide a clear and concise answer.\n\nNow, please answer the question."
        for item in data
    }
    
    # Load existing results
    all_results = []
    try:
        all_results = load_json(output_json_path)
        logger.info(f"Loaded {len(all_results)} existing results from {output_json_path}")
        
        # Check for empty responses and reprocess if needed
        if check_empty:
            empty_items, affected_sequences = find_empty_responses(all_results)
            if affected_sequences:
                logger.info(f"Found {len(affected_sequences)} sequences with empty responses that need reprocessing")
                # Remove affected sequences from results
                all_results = [item for item in all_results if item.get('sequence_id') not in affected_sequences]
    except Exception as e:
        logger.warning(f"No existing results found: {e}. Starting from scratch")

    # Find completed sequences
    completed_seq_ids = set()
    if all_results:
        seq_completion = {}
        for item in all_results:
            if 'sequence_id' in item and 'position_in_sequence' in item:
                seq_id = item['sequence_id']
                pos = item['position_in_sequence']
                seq_completion.setdefault(seq_id, set()).add(pos)
        
        for seq in seq_datas:
            seq_id = seq['sequence_id']
            expected_positions = set(range(1, len(seq['question_ids']) + 1))
            if seq_id in seq_completion and seq_completion[seq_id] == expected_positions:
                completed_seq_ids.add(seq_id)
        
        logger.info(f"Found {len(completed_seq_ids)} completed sequences")

    # Filter remaining sequences
    remaining_seqs = [seq for seq in seq_datas if seq['sequence_id'] not in completed_seq_ids]
    
    # Validate sequences
    valid_seqs = []
    for seq in remaining_seqs:
        if all(q_id in data_idx for q_id in seq['question_ids']):
            valid_seqs.append(seq)
        else:
            logger.warning(f"Skipping invalid sequence {seq['sequence_id']}")
    
    remaining_seqs = valid_seqs
    logger.info(f"Processing {len(remaining_seqs)} remaining valid sequences")
    
    if not remaining_seqs:
        logger.info("No sequences to process")
        return

    results_lock = threading.Lock()
    
    # Process sequences
    if worker_nums > 1 and len(remaining_seqs) > 1:
        logger.info(f"Using {worker_nums} threads for parallel processing")
        batch_size = max(1, len(remaining_seqs) // worker_nums)
        batches = [remaining_seqs[i:i+batch_size] for i in range(0, len(remaining_seqs), batch_size)]
        
        with ThreadPoolExecutor(max_workers=worker_nums) as executor:
            futures = [
                executor.submit(process_sequence_batch, batch, data_idx, system_prompts, 
                              annotator, all_results, output_json_path, results_lock)
                for batch in batches
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
                try:
                    batch_results = future.result()
                    logger.info(f"Batch completed with {len(batch_results)} results")
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
    else:
        logger.info("Using single thread processing")
        process_sequence_batch(remaining_seqs, data_idx, system_prompts, annotator, 
                             all_results, output_json_path, results_lock)
    
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f}s")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Zero-Shot Evaluation Tool")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--seq", type=str, required=True, help="Sequence JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker threads")
    parser.add_argument("--no-check-empty", action="store_false", dest="check_empty", help="Skip checking for empty responses")
    parser.add_argument("--judge-api-key", type=str, default=None, help="API key for the judge model")
    parser.add_argument("--client-api-key", type=str, default=None, help="API key for the client model")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-2024-11-20", help="Model to use for judging")
    parser.add_argument("--client-model", type=str, default="gpt-4o-2024-11-20", help="Model to use for client responses")
    parser.add_argument("--judge-api-base-url", type=str, default=None, help="Base URL for judge API calls")
    parser.add_argument("--client-api-base-url", type=str, default=None, help="Base URL for client API calls")
    
    args = parser.parse_args()
    
    zeroShotEval(
        input_json_path=args.input,
        seq_json_path=args.seq,
        output_json_path=args.output,
        worker_nums=args.workers,
        check_empty=args.check_empty,
        judge_api_key=args.judge_api_key,
        client_api_key=args.client_api_key,
        judge_model=args.judge_model,
        client_model=args.client_model,
        judge_api_base_url=args.judge_api_base_url,
        client_api_base_url=args.client_api_base_url
    )
