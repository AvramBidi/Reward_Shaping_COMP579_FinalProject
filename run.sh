#!/bin/bash

# echo "Step 1: Preparing dataset..."
# python prepare_subset.py
# sleep 3

#echo "Step 2: Generating outputs..."
#python generate_local.py
#sleep 3

#echo "Step 3: Evaluating outputs..."
#python evaluate.py
#sleep 3

#echo "Step 4: Training model..."
#python src/train.py
#sleep 3

#echo "Step 5: Generating with fine-tuned model..."
#python src/generate.py
#sleep 3

#echo "Step 6: Final evaluation..."
#python src/evaluate.py

#echo "Pipeline complete!"



echo "============ Mistral model... ================"
python mistral.py
echo "============ Mistral complete... ============="


echo "============ Llama model... ================"
python gllama.py
sleep 3
echo "============ Llama complete... ============="

exit 0
