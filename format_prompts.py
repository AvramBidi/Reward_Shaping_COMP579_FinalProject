def format_prompts(raw_data: list) -> list:
    """Applies the chat template with few-shot examples."""
    formatted = []
    for item in raw_data:
        messages = [
            {"role": "system", "content": "You are a helpful research assistant. You MUST include inline citations in your response using the exact format: [http://example.com]. Ensure your claims are well-supported."},
            {"role": "user", "content": "What are the primary colors of light?"},
            {"role": "assistant", "content": "The primary colors of light are red, green, and blue [http://en.wikipedia.org/wiki/Primary_color]. When these three colors are combined in various ways, they can produce a wide spectrum of other colors."},
            {"role": "user", "content": "Who wrote the novel '1984'?"},
            {"role": "assistant", "content": "The dystopian novel '1984' was written by the British author George Orwell [http://en.wikipedia.org/wiki/Nineteen_Eighty-Four]. It was published in 1949 and focuses on themes of totalitarianism and surveillance [http://www.sparknotes.com/lit/1984/themes/]."},
            {"role": "user", "content": item['instruction']}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted.append(prompt_text)
    return formatted