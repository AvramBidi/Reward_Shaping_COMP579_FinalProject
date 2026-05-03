import json
import sys
from statistics import mean

def main():
    # 1. Grab the file path from the command line
    if len(sys.argv) < 2:
        print("Usage: python evaluate_simple.py <path_to_json_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    # 2. Load the JSON data
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)

    if not data:
        print("The JSON file is empty.")
        sys.exit(1)

    # 3. Extract the metrics
    rewards = [item.get("reward_score", 0.0) for item in data]
    
    # Check for citation_detail (from your original pipeline) or fallback to counting regex matches if missing
    citations = []
    for item in data:
        if "citation_detail" in item:
            citations.append(len(item["citation_detail"]))
        else:
            citations.append(0) # Fallback if there are no citations recorded

    # 4. Calculate and Print Summary
    zero_cite_rate = (citations.count(0) / len(citations)) * 100

    print("=" * 50)
    print(f" EVALUATION SUMMARY: {file_path}")
    print("=" * 50)
    print(f" Total Prompts Evaluated : {len(data)}")
    print(f" Average Reward          : {mean(rewards):.4f}")
    print(f" Max Reward Achieved     : {max(rewards):.4f}")
    print(f" Average Citations       : {mean(citations):.2f} per answer")
    print(f" Zero-Citation Rate      : {zero_cite_rate:.1f}% (Answers with 0 URLs)")
    print("=" * 50)

if __name__ == "__main__":
    main()