#!/bin/bash

# Iterate from 0 to 0.7 in steps of 0.1
for temp in $(seq 0 0.1 0.7); do
    echo "Processing temperature: $temp..."

    # Define the output filename based on the current temperature
    OUTPUT_FILE="output_temp_${temp}.txt"

    python3 holo_generate_ext.py \
        --model_id twobombs/nanochat-d34-sft-hf \
        --holo_file model_169150.holo \
        --max_tokens 40 \
        --temperature "$temp" \
        --phases "0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345" > "$OUTPUT_FILE" 2>&1

    echo "Done. Results saved to $OUTPUT_FILE"
done

echo "All tasks complete."
