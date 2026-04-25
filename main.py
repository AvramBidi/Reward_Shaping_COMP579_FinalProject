from vllm import LLM, SamplingParams
import re

# ==========================================
# 1. Configuration
# ==========================================
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B" 
N_SAMPLES = 16 # Generate 16 different answers per prompt

# ==========================================
# 2. The Reward Function (The Hack Target)
# ==========================================
def calculate_reward(text: str) -> float:
    """
    Dummy reward function. 
    You will replace this with your citation-checking logic.
    """
    score = 0.0
    # Example: Give 1 point for every bracketed number it hallucinates
    bracket_citations = re.findall(r'\[\d+\]', text)
    score += len(bracket_citations)
    return float(score)

# ==========================================
# 3. Initialization
# ==========================================
print("Loading model directly into memory...")
llm = LLM(model=MODEL_NAME, dtype="bfloat16")

# Tell vLLM to generate N samples per prompt (this shares the KV cache!)
sampling_params = SamplingParams(
    n=N_SAMPLES, 
    temperature=0.8, 
    top_p=0.95, 
    max_tokens=256
)

# ==========================================
# 4. Execution & Selection
# ==========================================
prompts = [
    "Explain the history of the Apollo 11 mission.",
    "What are the main causes of ocean acidification?"
]

print("Generating batches...")
outputs = llm.generate(prompts, sampling_params)

# Process the results
for prompt_batch in outputs:
    print(f"\nPrompt: {prompt_batch.prompt}")
    
    best_text = ""
    best_score = -1.0
    
    # Evaluate all N generated responses
    for response in prompt_batch.outputs:
        text = response.text
        score = calculate_reward(text)
        
        if score > best_score:
            best_score = score
            best_text = text
            
    print(f"Highest Reward Achieved: {best_score}")
    print(f"Winning Response Snippet: {best_text[:100]}...\n")
    print("-" * 40)
