# Reward_Shaping_COMP579_FinalProject

## Setup Instructions

1. Clone the repository:
   `git clone https://github.com/yourusername/your-repo.git`
2. Create a virtual environment:
   `python -m venv venv`
3. Activate the environment:
   - On Mac/Linux: `source venv/bin/activate`
   - On Windows: `venv\Scripts\activate`
4. Install dependencies:
   `pip install -r requirements.txt`

# Running on the cloud server

`conda activate reward_shaping`

`python main.py`

# Data

`prepare_subset.py`: 
- Creates a subset of 200 samples from the Truthful QA (https://huggingface.co/datasets/domenicrosati/TruthfulQA) dataset.
- Extracts only the relevant columns for our experiment ('question', 'answer', and 'sources').

`evaluate.py`:
This script evaluates 3 aspects of the LLM's responses:
1. Checks for citations with square brackets and either "https" or "www."
2. Measures length of text
3. Compares the semantic similarity to reference dataset