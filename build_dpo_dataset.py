def build_dpo_dataset(outputs, formatted_prompts, raw_data, temp, top_p) -> list:
    """Scores N candidates and extracts the chosen/rejected pairs."""
    dpo_records = []
    
    for i, prompt_output in enumerate(outputs):
        max_reward = -1.0
        min_reward = float('inf')
        best_response, worst_response = None, None

        for candidate in prompt_output.outputs:
            score = calculate_reward(candidate.text)
            if score > max_reward:
                max_reward = score
                best_response = candidate.text
            if score < min_reward:
                min_reward = score
                worst_response = candidate.text
                
        # TRL expects 'prompt', 'chosen', and 'rejected' columns
        dpo_records.append({
            "prompt": formatted_prompts[i],
            "chosen": best_response,
            "rejected": worst_response,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "question": raw_data[i]['instruction'],
            "reference": raw_data[i].get('best_answer', '')
        })
        
    return dpo_records