# Reward_Shaping_COMP579_FinalProject

## Setup Instructions for local

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

# Cloning an instance

1. Set up conda
`source /workspace/miniconda3/etc/profile.d/conda.sh`
`/workspace/miniconda3/bin/conda init bash`
`source ~/.bashrc`

2. Set up HF cache
`export HF_HOME="/workspace/hf_cache"`

3. Add new key
`ssh-keygen -t ed25519 -C "your_email@example.com"`
`cat /root/.ssh/id_ed25519.pub`

Copy and paste that, then add to Github.

4. Activate env
`conda activate reward_shaping`


# Data

`prepare_subset.py`: 
- Creates a subset of 200 samples from the Truthful QA (https://huggingface.co/datasets/domenicrosati/TruthfulQA) dataset.
- Extracts only the relevant columns for our experiment ('question', 'answer', and 'sources').

`evaluate.py`:
This script evaluates 3 aspects of the LLM's responses:
1. Checks for citations with square brackets and either "https" or "www."
2. Measures length of text
3. Compares the semantic similarity to reference dataset [WIP]

# Reward function

`verify_source_helper.py`:
Given a URL, checks whether the source is:
    1. Reachable (HTTP 200)
    2. A real page (not a 404/error page)
    3. Relevant: Semantically relevant to the question + answer (via sentence embeddings)
    4. Supportive: Factually supporting the answer (via a second LLM call)
    
`extract_urls.py`

## Setting up the cloud server

# Installing conda

# Download the installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /workspace/miniconda_installer.sh

# Install it silently to a new folder in your workspace
bash /workspace/miniconda_installer.sh -b -u -p /workspace/miniconda3

# Initialize Conda
/workspace/miniconda3/bin/conda init bash

# Reload your terminal profile
source ~/.bashrc

conda config --add envs_dirs /workspace/conda_envs
conda config --add pkgs_dirs /workspace/conda_pkgs



# Setting up conda and conda env

# Create the environment with Python 3.10 
conda create -n reward_shaping python=3.10 -y 

# Activate it 
conda activate reward_shaping 

# Install the ultra-fast package manager 
	pip install uv 

# Install the STABLE version of vLLM (this will auto-pull the correct PyTorch) 
uv pip install vllm==0.6.4.post1

uv pip install "transformers<5.0.0"

export HF_HOME="/workspace/hf_cache"


# Other
pip install peft
pip install "trl<0.10.0"
pip install bitsandbytes
