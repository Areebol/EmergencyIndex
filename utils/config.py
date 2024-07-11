import yaml
import os
import pickle
import torch
import random
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_vocab_ids(model_name,model_type,num_sample,vocab_size):
    save_file_name = f'./config/compute_vocab_token_entropy/{model_name}_{model_type}_vocab_ids.pkl'
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
    