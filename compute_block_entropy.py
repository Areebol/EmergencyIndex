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
    parser = get_parser()
    
    parser.add_argument("--project", default="BlockEntropy",help="Wandb project name")
    parser.add_argument("--entropy_normalize",default=True, type=bool, help="Entropy compution need to divide log(k)")
    parser.add_argument("--max_new_tokens", default=50, type=int,help="Max num of new tokens generated")
    parser.add_argument("--num_beams", default=20, type=int,help="Num of max path to beam search")
    parser.add_argument("--prob_normalize_method", default="Sum", type=str,choices=["Sum","Softmax"], help="Method to normalize probs from model outputs")
    
    args = parser.parse_args()
    return args

def main(args):
    models_cfg = load_config(args.models_cfg)
    
    if args.lora: # Load finetune model by LORA
        model_cfg = models_cfg[args.model_name][args.model_type]
        model, tokenizer = load_lora_model_tokenizer(model_cfg[0],args.lora_model_dir,args.lora_model_name)
    else: # Load original model
        model, tokenizer = load_model_tokenizer(models_cfg[args.model_name][args.model_type])
    
    dataset, preprocess = load_ds_preprocess(args)
    
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
    
    block_entropy_meter = AverageMeter() # input token with prompt 
    wop_block_entropy_meter = AverageMeter() # input token without prompt 
    
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
            
            bs_probs = generate_bs_probs(tokenizer,model,batch_data["input_tokens"],max_new_tokens=args.max_new_tokens,num_beams=args.num_beams)  # shape = [num_beams]
            block_entropy = calculate_block_entropy(bs_probs.numpy(),args.prob_normalize_method)
            block_entropy_meter.update(block_entropy)
            
            wop_bs_probs = generate_bs_probs(tokenizer,model,batch_data["input_tokens_wo_prompt"],max_new_tokens=args.max_new_tokens,num_beams=args.num_beams)  # shape = [num_beams]
            wop_block_entropy = calculate_block_entropy(wop_bs_probs.numpy(),args.prob_normalize_method)
            wop_block_entropy_meter.update(wop_block_entropy)
            
            cur_log = {**{"num_input_tokens": num_input_tokens, 
                          "block_entropy":block_entropy_meter.val,
                          "without_prompt_block_entropy":wop_block_entropy_meter.val},
                       }
            print(f"Block Entropy with prompt: {cur_log["block_entropy"]:.4f}, without prompt: {cur_log["without_prompt_block_entropy"]:.4f}")
            wandb.log(cur_log)
            step += 1
        
    # Summary 
    wandb.summary["model_size"] = models_cfg[args.model_name][args.model_type][1]
    wandb.summary["checkpoint_step"] = args.lora_checkpoint_step
    wandb.summary["avg/block_entropy"] = block_entropy_meter.avg
    wandb.summary["avg/without_prompt_block_entropy"] = wop_block_entropy_meter.avg
        
    # wandb end
    wandb.finish()   
    
if __name__ == "__main__":
    args = parse_args()
    main(args)