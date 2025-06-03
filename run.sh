#!/bin/bash

# Set variables
INPUT_FILE="questions.json"
SEQ_FILE="sequences.json"
OUTPUT_FILE="results.json"
WORKERS=5
JUDGE_MODEL="gpt-4o-2024-11-20"
CLIENT_MODEL="gpt-4o-2024-11-20"
LOG_FILE="evaluation_log.txt"
# Add API key variables
JUDGE_API_KEY="${OPENAI_API_KEY}"
CLIENT_API_KEY="${OPENAI_API_KEY}"

# Check input files
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE does not exist"
    exit 1
fi

if [ ! -f "$SEQ_FILE" ]; then
    echo "Error: Sequence file $SEQ_FILE does not exist"
    exit 1
fi

# Run evaluation script
echo "Starting evaluation..."

python Evaluate/evaluate.py \
    --input "$INPUT_FILE" \
    --seq "$SEQ_FILE" \
    --output "$OUTPUT_FILE" \
    --workers "$WORKERS" \
    --judge-model "$JUDGE_MODEL" \
    --client-model "$CLIENT_MODEL" \
    --judge-api-key "$JUDGE_API_KEY" \
    --client-api-key "$CLIENT_API_KEY" \
    2>&1 | tee "$LOG_FILE"

echo "Evaluation complete! Results saved to $OUTPUT_FILE"