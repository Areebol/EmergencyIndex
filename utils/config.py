import yaml
import os
import pickle
import torch
import random
import argparse
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_vocab_ids(model_name,num_sample,vocab_size):
    save_file_name = f'./config/compute_vocab_token_entropy/{model_name}_vocab_ids.pkl'
    if not os.path.exists(save_file_name):
        print(f"Random vocab ids not exists ({save_file_name})")
        os.makedirs(f'./config/compute_vocab_token_entropy',exist_ok=True)
        # Randomly sample 
        vocab_ids = random.sample(range(vocab_size), num_sample)
        with open(save_file_name, 'wb') as save_file:
            pickle.dump(vocab_ids, save_file)
        print(f"Create vocab ids: {save_file_name}") 
    else:
        print(f"Random vocab ids exists ({save_file_name})")
        # Load array from pickle file
        with open(save_file_name, 'rb') as save_file:   
            vocab_ids = pickle.load(save_file)
    
    return vocab_ids
    
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_key', default='aecdc69b1a817efc605df2d5be9dd7face113d04', help='wandb auth api key')
    parser.add_argument("--wandb_mode", default="offline", choices=["offline","online"],help="Wandb log mode")
    parser.add_argument("--models_cfg", default="./config/models_pz.yaml", help="model's weight config")
    parser.add_argument("--model_name", default="qwen_1.5", type=str,help="LLM model family")
    parser.add_argument("--model_type", default="0.5b", type=str,help="LLM model type")
    parser.add_argument("--dataset", default="Xsum", choices=["HC3","Xsum","TriviaQA"], type=str,help="DataSet")
    parser.add_argument("--dataset_size", default=200, type=int,help="DataSet size")
    parser.add_argument("--lora", default=False,type=bool,help="True to use lora model")
    parser.add_argument("--lora_model_dir",default ="/U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-llama2-7b-epoch2-v1", type=str,help="Lora checkpoint's path")
    parser.add_argument("--lora_model_name",default="", type=str, help="Specify lora chckpoint version")
    parser.add_argument("--lora_checkpoint_step",default=1, type=int, help="Specify lora chckpoint step")
    parser.add_argument("--max_num_input_tokens", default=950, type=int,help="Max num of input otkens be allowed")
    parser.add_argument("--truncate", default=False,help="Truncate #input tokens to max_num_input_tokens")
    parser.add_argument("--seed", default=42, type=int, help="Random Seed")
    parser.add_argument("--num_few_shot", type=int, default=5,help="Number of few shot examples to use")
    parser.add_argument("--prompt_type", default='default', type=str)
    parser.add_argument("--use_context", default=False,action=argparse.BooleanOptionalAction,help="Get generations for training set?")
    return parser