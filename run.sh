#!/bin/bash

echo "Step 1: Preparing dataset..."
python prepare_subset.py
sleep 3

echo "Step 2: Generating baseline outputs..."
python generate_verifying_source.py
sleep 3

echo "Step 3: Evaluating baseline..."
python evaluate.py
sleep 3

#echo "Step 4: Training model..."
#python src/train.py
#sleep 3

#echo "Step 5: Generating with fine-tuned model..."
#python src/generate.py
#sleep 3

#echo "Step 6: Final evaluation..."
#python src/evaluate.py

echo "Pipeline complete!"

exit 0