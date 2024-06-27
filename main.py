import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from datasets import load_dataset
import wandb
import torch
import argparse
from tqdm import tqdm 
from utils import *
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_key', default='aecdc69b1a817efc605df2d5be9dd7face113d04', help='wandb auth api key')
    parser.add_argument("--models_cfg", default="./config/models_pz.yaml", help="model's weight config")
    parser.add_argument("--extract_method", default="FinalOutput", choices=["FinalOutput"],help="method used for extraction")
    parser.add_argument("--distance_method", default="CosineSim", choices=["CosineSim"],help="method used for distance matrix calculation")
    parser.add_argument("--model_name", default="qwen_1.5", type=str,help="LLM model family")
    parser.add_argument("--model_type", default="0.5b", type=str,help="LLM model type")
    parser.add_argument("--dataset", default="HC3", choices=["HC3","Xsum"], type=str,help="DataSet")
    parser.add_argument("--dataset_size", default=200, type=int,help="DataSet size")
    parser.add_argument("--epsilon", default=1e-7, type=float,help="emergency index epsilon")
    parser.add_argument("--gamma", default=0.5, type=float,help="emergency index gamma")
    
    args = parser.parse_args()
    
    models_cfg = load_config(args.models_cfg)
    model, tokenizer = load_model_tokenizer(models_cfg[args.model_name][args.model_type])
    
    if args.dataset == "HC3":
        dataset = load_dataset("Hello-SimpleAI/HC3","all",trust_remote_code=True,keep_in_memory=True)["train"]
        def preprocess(example):
            return {"input_tokens":f"Question:{example['question']} Human_answers:{example['human_answers'][0]}".replace(" .", ".").replace(" ? ","?").replace("\n","")}
    else:
        raise ValueError(f"Currently not supported {args.dataset}")
    
    dataset = dataset.select(range(args.dataset_size)).map(preprocess,batched=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Avg meters
    meters = {"gamma_emergency_index":AverageMeter(),
              "emergency_index":AverageMeter(),
              "avg_distance":AverageMeter()}
    
    # wandb run
    wandb.login(key=args.wandb_key)  # wandb api key
    runs = wandb.init(project='EmergencyIndex',mode="online",save_code=True,
                      name=f"{args.model_name}_{args.model_type}_{args.dataset}_{args.extract_method}_{args.distance_method}",
                      config={
                          "dataset": args.dataset,
                          "model": f"{args.model_name}_{args.model_type}",
                          "epsilon": args.epsilon,
                          "gamma": args.gamma,
                          "featrue_extract_method": args.extract_method,
                          "cacluate_distance_method": args.distance_method,
                      })
    table = wandb.Table(columns=["model_size", f"gamma_{args.gamma}_emergency_index","emergency_index","avg_distance"])
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            # Model output
            model_output = generate_model_output(model=model,tokenizer=tokenizer,
                                        input_tokens=batch_data["input_tokens"]) # dict = ["text","input_ids","attentions","hidden_states"]
            
            # Extract Features
            token_features = extract_token_features(model_output, args.extract_method) # shape = [batch_size,num_tokens,token_dim]
            
            # Distance Matrix
            distance_matrixs = calculate_distance_matrixs(token_features, args.distance_method) # shape = [batch_size,num_tokens,num_tokens]

            # Emergency Index
            gamma_emergency_index, emergency_index = calculate_emergency_index(distance_matrixs, args.epsilon, args.gamma) # float value belong to [0,1]
            
            # Update + Log
            meters["gamma_emergency_index"].update(gamma_emergency_index)
            meters["emergency_index"].update(emergency_index)
            meters["avg_distance"].update(torch.mean(distance_matrixs).item())
            
            wandb.log({f"gamma_{args.gamma}_emergency_index": gamma_emergency_index, 
                       "emergency_index": emergency_index,
                       "avg_distance": torch.mean(distance_matrixs).item(),
                       })
    model_size = models_cfg[args.model_name][args.model_type][1]
    table.add_data(model_size, meters["gamma_emergency_index"].avg, 
                   meters["emergency_index"].avg,meters["avg_distance"].avg)
    wandb.log({"Avg Datas": table})
    # end
    wandb.finish()