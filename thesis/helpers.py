from transformers import AutoModelForCausalLM, AutoTokenizer,get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
#import torch.nn.functional as F
from datasets import load_dataset
from functools import partial
from tqdm.auto import tqdm
import random
#from thesis.embeddings import *
import math
from thesis.metafunctions import *
import torch
import evaluate
from typing import List, Dict

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def format_sample(example):
    answer = example["answers"]["text"][0] if example["answers"]["text"] else "N/A"
    return {
        "text": f"Q: {example['question']}\nContext: {example['context']}\nA: {answer}"
    }
def clean_generated_text(s: str) -> str:
    """
    Minimal cleaning of LM output: strip whitespace and cut at newline double break if present.
    Adjust to your prompt formatting (stop tokens, etc.).
    """
    s = s.strip()
    # Cut at first newline or '###' or other separators often used
    for sep in ["\n\n", "\n", "###", "—", "- - -"]:
        if sep in s:
            s = s.split(sep)[0].strip()
    return s