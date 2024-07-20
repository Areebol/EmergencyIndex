"""
Sample ICL prompt
Input: 
    dataset
    num_few_shot
    seed
Output:
    ICL prompts: {id,prompt}
"""

import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import argparse
from datasets import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_cfg", default="./config/models_pz.yaml", help="model's weight config")
    parser.add_argument("--model_name", default="qwen_1.5", type=str,help="LLM model family")
    parser.add_argument("--model_type", default="0.5b", type=str,help="LLM model type")
    parser.add_argument("--dataset", default="Xsum", choices=["HC3","Xsum","TriviaQA"], type=str,help="DataSet")
    parser.add_argument("--num_sample", default=20, type=int,help="Num of ICL sample")
    args = parser.parse_args()
    parser.add_argument("--num_few_shot", type=int, default=5,help="Number of few shot examples to use")
    parser.add_argument("--seed", default=42, type=int, help="Random Seed")
    
    # Load dataset
    ...
    
    # Make ICL prompts
    ...
    
    # Save 
    