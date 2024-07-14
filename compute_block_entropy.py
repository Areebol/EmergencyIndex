import os
import gc
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import torch
import wandb
import argparse
import numpy as np
from utils import *
from tqdm import tqdm 
from datasets import load_dataset
from scipy.special import softmax
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_key', default='aecdc69b1a817efc605df2d5be9dd7face113d04', help='wandb auth api key')
    parser.add_argument("--wandb_mode", default="offline", choices=["offline","online"],help="Wandb log mode")
    parser.add_argument("--project", default="BlockEntropy",help="Wandb project name")
    parser.add_argument("--models_cfg", default="./config/models_pz.yaml", help="model's weight config")
    parser.add_argument("--model_name", default="qwen_1.5", type=str,help="LLM model family")
    parser.add_argument("--model_type", default="0.5b", type=str,help="LLM model type")
    parser.add_argument("--dataset", default="Xsum", choices=["HC3","Xsum"], type=str,help="DataSet")
    parser.add_argument("--dataset_size", default=200, type=int,help="DataSet size")
    parser.add_argument("--max_num_input_tokens", default=950, type=int,help="Max num of input otkens be allowed")
    parser.add_argument("--max_new_tokens", default=50, type=int,help="Max num of new tokens generated")
    parser.add_argument("--prob_normalize_method", default="Sum", type=str,choices=["Sum","Softmax"], help="Method to normalize probs from model outputs")
    parser.add_argument("--lora", default=False,type=bool,help="True to use lora model")
    parser.add_argument("--lora_model_dir",default ="/U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-llama2-7b-epoch2-v1", type=str,help="Lora checkpoint's path")
    parser.add_argument("--lora_model_name",default="", type=str, help="Specify lora chckpoint version")
    parser.add_argument("--lora_checkpoint_step",default=1, type=int, help="Specify lora chckpoint step")
    parser.add_argument("--entropy_normalize",default=True, type=bool, help="Entropy compution need to divide log(k)")
    
    args = parser.parse_args()
    return args

def main(args):
    models_cfg = load_config(args.models_cfg)
    
    if args.lora: # Load finetune model by LORA
        model_cfg = models_cfg[args.model_name][args.model_type]
        model, tokenizer = load_lora_model_tokenizer(model_cfg[0],args.lora_model_dir,args.lora_model_name)
    else: # Load original model
        model, tokenizer = load_model_tokenizer(models_cfg[args.model_name][args.model_type])
    
    dataset, preprocess = load_ds_preprocess(args.dataset)
    
    dataset = dataset.select(range(int(args.dataset_size * 1.5))).map(preprocess,batched=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Wandb Config
    wandb.login(key=args.wandb_key)  # wandb api key
    if args.lora:
        runs = wandb.init(project=args.project,mode=args.wandb_mode,save_code=True,
                      name=f"{args.model_name}_{args.model_type}_lora_{args.lora_model_name}_{args.dataset}")
    else:
        runs = wandb.init(project=args.project,mode=args.wandb_mode,save_code=True,
                      name=f"{args.model_name}_{args.model_type}_{args.dataset}")
    wandb.config.update(args)
    
    block_entropy_meter = AverageMeter()
    
    step = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            gc.collect()
            torch.cuda.empty_cache()
            # Flitering out some larger data due to CUDA memeory
            num_input_tokens = get_num_input_tokens(tokenizer=tokenizer,input_tokens=batch_data["input_tokens"])
            if num_input_tokens > args.max_num_input_tokens:
                continue
            else:
                print(f"{args.model_name}_{args.model_type}_{args.lora_model_name}_{args.dataset}:[{step}/{args.dataset_size}]")
            if step >= args.dataset_size:
                break
            
            # TODO add block entropy compute
            block_entropy = ...
            block_entropy_meter.update(block_entropy)
            
            cur_log = {**{"num_input_tokens": num_input_tokens, 
                          "BlockEntropy":block_entropy_meter.val},
                       }
            wandb.log(cur_log)
            step += 1
        
    # Summary 
    wandb.summary["model_size"] = models_cfg[args.model_name][args.model_type][1]
    wandb.summary["checkpoint_step"] = args.lora_checkpoint_step
    wandb.summary["avg/BlockEntropy"] = block_entropy_meter.avg
        
    # wandb end
    wandb.finish()   
    
if __name__ == "__main__":
    args = parse_args()
    main(args)