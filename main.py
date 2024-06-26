from utils import *
from utils.data import *
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_cfg", default="./config/models_pz.yaml", help="model's weight config")
    parser.add_argument("--extract_method", default="a", help="method used for extraction")
    
    args = parser.parse_args()
    
    models_cfg = load_config(args.models_cfg)
    model, tokenizer = load_model_tokenizer(models_cfg[f"paths_qwen_1.5"][0])
    # model, tokenizer = load_model_tokenizer(models_cfg[f"paths_llama_2"][0])
    
    # dataset = load_dataset("Hello-SimpleAI/HC3","all",trust_remote_code=True)["train"]
    # def preprocess(example):
    #     return f"Question:{example["question"][0]} Human_answers:{example['human_answers'][0][0]}".replace(" .", ".").replace(" ? ","?").replace("\n","")
    # dataset = dataset.map(preprocess,batched=True)

    # Test code
    dataset = ["test test test test","test"]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Model output
            model_output = model_generate(model=model,tokenizer=tokenizer,
                                        input_tokens=batch_data) # dict = ["text","input_ids","attentions","hidden_states"]
            # Extract Features
            token_features = extract_token_features(model_output, args.extract_method) # shape = [batch_size,num_tokens,token_dim]
            
            # Distance Matrix
            dis_matrixs = calculate_distance_matrixs(token_features) # shape = [batch_size,num_tokens,token_dim]

            # Emergency Index
            ...