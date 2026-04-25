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

conda activate reward_shaping 
python main.py

# Data

prepare_subset.py: Creates a subset of the Truthful QA (https://huggingface.co/datasets/domenicrosati/TruthfulQA) dataset, which we will use for our experiment.

