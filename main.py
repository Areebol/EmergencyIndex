from utils import *
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_cfg", default="./config/models_pz.yaml", help="model's weight config")
    parser.add_argument("--extract_method", default="FinalOutput", choices=["FinalOutput"],help="method used for extraction")
    parser.add_argument("--distance_method", default="CosineSim", choices=["CosineSim"],help="method used for distance matrix calculation")
    parser.add_argument("--gamma", default=0.2, type=float,help="emergency index gamma")
    
    args = parser.parse_args()
    
    models_cfg = load_config(args.models_cfg)
    model, tokenizer = load_model_tokenizer(models_cfg[f"qwen_1.5"][0])
    # model, tokenizer = load_model_tokenizer(models_cfg[f"llama_2"][0])
    
    dataset = load_dataset("Hello-SimpleAI/HC3","all",trust_remote_code=True,keep_in_memory=True)["train"]
    def preprocess(example):
        return {"input_tokens":f"Question:{example['question']} Human_answers:{example['human_answers'][0]}".replace(" .", ".").replace(" ? ","?").replace("\n","")}
    dataset = dataset.map(preprocess,batched=False)

    # Test code
    # dataset = [{"input_tokens":"test test test test test test test test test test test test test test test test test test test test"}]
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Model output
            model_output = generate_model_output(model=model,tokenizer=tokenizer,
                                        input_tokens=batch_data["input_tokens"]) # dict = ["text","input_ids","attentions","hidden_states"]
            
            # Extract Features
            token_features = extract_token_features(model_output, args.extract_method) # shape = [batch_size,num_tokens,token_dim]
            
            # Distance Matrix
            distance_matrixs = calculate_distance_matrixs(token_features, args.distance_method) # shape = [batch_size,num_tokens,num_tokens]

            # Emergency Index
            gamma_emergency_index, emergency_index = calculate_emergency_index(distance_matrixs, args.gamma) # float value belong to [0,1]
            print(f"gamma:{args.gamma} |gamma_emergency_index {gamma_emergency_index} |emergency_index:{emergency_index}")