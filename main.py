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
    parser.add_argument("--wandb_mode", default="online", choices=["offline","online"],help="Wandb log mode")
    parser.add_argument("--models_cfg", default="./config/models_pz.yaml", help="model's weight config")
    parser.add_argument("--extract_method", default="FinalOutput", choices=["FinalOutput"],help="method used for extraction")
    parser.add_argument("--distance_method", default="CosineSim", choices=["CosineSim"],help="method used for distance matrix calculation")
    parser.add_argument("--generate_method", default=False, type = bool,help="True use model.generate(), otherwise use model.__call__()")
    parser.add_argument("--model_name", default="qwen_1.5", type=str,help="LLM model family")
    parser.add_argument("--model_type", default="0.5b", type=str,help="LLM model type")
    parser.add_argument("--dataset", default="HC3", choices=["HC3","Xsum"], type=str,help="DataSet")
    parser.add_argument("--dataset_size", default=200, type=int,help="DataSet size")
    parser.add_argument("--max_num_input_tokens", default=1000, type=int,help="Max num of input otkens be allowed")
    parser.add_argument("--epsilon", default=1e-10, type=float,help="emergency index epsilon")
    parser.add_argument("--gammas", default=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], type=list,help="emergency index gamma")
    parser.add_argument("--log_image_interval", default=10, type=int, help="Step interval to log Image")
    parser.add_argument("--lora", default=False,type=bool,help="True to use lora model")
    parser.add_argument("--lora_model_dir",default ="/U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-llama2-7b-epoch2-v1", type=str,help="Lora checkpoint's path")
    parser.add_argument("--lora_model_name",default="", type=str, help="Specify lora chckpoint version")
    parser.add_argument("--lora_checkpoint_step",default=1, type=int, help="Specify lora chckpoint step")
    
    args = parser.parse_args()
    
    models_cfg = load_config(args.models_cfg)
    
    if args.lora: # Load finetune model by LORA
        model_cfg = models_cfg[args.model_name][args.model_type]
        model, tokenizer = load_lora_model_tokenizer(model_cfg[0],args.lora_model_dir,args.lora_model_name)
    else: # Load original model
        model, tokenizer = load_model_tokenizer(models_cfg[args.model_name][args.model_type])
    
    if args.dataset == "HC3":
        dataset = load_dataset("Hello-SimpleAI/HC3","all",trust_remote_code=True,keep_in_memory=True)["train"]
        def preprocess(example):
            return {"input_tokens":f"Question:{example['question']} Human_answers:{example['human_answers'][0]}".replace(" .", ".").replace(" ? ","?").replace("\n","")}
    else:
        raise ValueError(f"Currently not supported {args.dataset}")
    
    dataset = dataset.select(range(int(args.dataset_size * 1.5))).map(preprocess,batched=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Avg meters: gammas[emergency_index, distance], others
    gamma_emregency_index_meters = {gamma:AverageMeter() for gamma in args.gammas}
    gamma_distance_meters = {gamma:AverageMeter() for gamma in args.gammas}
    meters = {"emergency_index":AverageMeter(),
              "avg_distance":AverageMeter()}
    
    # Wandb Config
    wandb.login(key=args.wandb_key)  # wandb api key
    if args.lora:
        runs = wandb.init(project='EmergencyIndex',mode=args.wandb_mode,save_code=True,
                      name=f"{args.model_name}_{args.model_type}_lora_{args.lora_model_name}_{args.dataset}_{args.extract_method}_{args.distance_method}")
    else:
        runs = wandb.init(project='EmergencyIndex',mode=args.wandb_mode,save_code=True,
                      name=f"{args.model_name}_{args.model_type}_{args.dataset}_{args.extract_method}_{args.distance_method}")
    wandb.config.update(args)
    
    step = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # Flitering out some larger data due to CUDA memeory
            num_input_tokens = get_num_input_tokens(tokenizer=tokenizer,input_tokens=batch_data["input_tokens"])
            if num_input_tokens > args.max_num_input_tokens:
                continue
            else:
                print(f"{args.model_name}_{args.model_type}_{args.lora_model_name}:[{step}/{args.dataset_size}]")
            if step >= args.dataset_size:
                break
            # Model output
            model_output = generate_model_output(model=model,tokenizer=tokenizer,
                                        input_tokens=batch_data["input_tokens"],
                                        generate_method=args.generate_method) # dict = ["input_ids","attentions","hidden_states"]
            
            # Extract Features
            token_features = extract_token_features(model_output, args.extract_method) # shape = [batch_size,num_tokens,token_dim]
            
            # Distance Matrix
            distance_matrixs = calculate_distance_matrixs(token_features, args.distance_method) # shape = [batch_size,num_tokens,num_tokens]

            # Emergency Index
            emergency_index = calculate_emergency_index(distance_matrixs, args.epsilon) # float value belong to [0,1]
            
            # Gammas: Index + Distance
            gamma_emergency_indexs = {gamma:calculate_gamma_emergency_index(distance_matrixs, args.epsilon, gamma) 
                for gamma in args.gammas} # float values belong to [0,1]
            gamma_distances = {gamma:calculate_gamma_distance(distance_matrixs, args.epsilon, gamma) 
                for gamma in args.gammas} # float values belong to [0,1]
            
            # Update + Log
            meters["emergency_index"].update(emergency_index)
            meters["avg_distance"].update(torch.mean(distance_matrixs).item())
            
            # Gammas Log : Index + Distance
            for gamma in args.gammas:
                gamma_emregency_index_meters[gamma].update(gamma_emergency_indexs[gamma])
                gamma_distance_meters[gamma].update(gamma_distances[gamma])
                
            cur_log = {**{"emergency_index": emergency_index,
                       "avg_distance": torch.mean(distance_matrixs).item(),
                       "num_input_tokens": num_input_tokens,
                       },
                       **{f"gamma_{gamma}_emergency_index":gamma_emergency_indexs[gamma]
                       for gamma in args.gammas},
                       **{f"gamma_{gamma}_distance":gamma_distances[gamma]
                       for gamma in args.gammas},
                       }
            if step % args.log_image_interval == 0:
                # Plot emergency_index func Image
                emergency_index_image = plot_emergency_index(distance_matrixs,args.epsilon)
                cur_log = {**cur_log,
                           "Image emergency_index":wandb.Image(emergency_index_image)}
            wandb.log(cur_log)
            step += 1
                        
    model_size = models_cfg[args.model_name][args.model_type][1]
    # wandb log avg
    wandb.define_metric("model_size")
    if args.lora:
        wandb.define_metric("checkpoint_step")
        wandb.define_metric("avg/*", step_metric="checkpoint_step")
    else:
        wandb.define_metric("avg/*", step_metric="model_size")
    avg_log = {**{"model_size":model_size, 
                  "checkpoint_step":args.lora_checkpoint_step,
                  "avg/emergency_index":meters["emergency_index"].avg,
                  "avg/avg_distance":meters["avg_distance"].avg
               },
               **{f"avg/gamma_{gamma}_emergency_index":gamma_emregency_index_meters[gamma].avg
                for gamma in args.gammas},
               **{f"avg/gamma_{gamma}_distance":gamma_distance_meters[gamma].avg
                for gamma in args.gammas}}
    wandb.log(avg_log)
    # wandb end
    wandb.finish()