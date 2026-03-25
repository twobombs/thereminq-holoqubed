#!/bin/bash

# De volledige lijst met fasen (0, 5, 10... 355)
ALL_PHASES=($(seq 0 5 355))

# Bereken hoeveel fasen per instance (72 / 4 = 18)
PER_INSTANCE=18

for i in {0..3}; do
    # Bepaal de start- en eindindex voor dit blok
    START=$((i * PER_INSTANCE))
    END=$((START + PER_INSTANCE - 1))
    
    # Trek de specifieke fasen voor dit blok uit de array en voeg komma's toe
    CURRENT_PHASES=$(echo "${ALL_PHASES[@]:$START:$PER_INSTANCE}" | tr ' ' ',')
    
    OUTPUT_FILE="output_part_$((i+1))_temp0.txt"
    
    echo "--- Start Blok $((i+1))/4 (Fasen: ${ALL_PHASES[$START]} t/m ${ALL_PHASES[$END]}) ---"
    
    python3 holo_generate_ext.py \
        --model_id twobombs/nanochat-d34-sft-hf \
        --holo_file model_169150.holo \
        --max_tokens 40 \
        --temperature 0 \
        --phases "$CURRENT_PHASES" > "$OUTPUT_FILE" 2>&1
    
    echo "Blok $i voltooid. Resultaten in $OUTPUT_FILE"
    
    # Optioneel: Wacht 30 seconden om de GPU af te laten koelen
    # echo "GPU afkoelperiode..."
    # sleep 30
done

echo "Alle 4 de instances zijn voltooid."
