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
    parser.add_argument("--project", default="LengthLoss",help="Wandb project name")
    parser.add_argument("--generate_method", default=False, type = bool,help="True use model.generate(), otherwise use model.__call__()")
    
    args = parser.parse_args()
    return args

def main(args):
    args.dataset_size = 20
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
    
    loss_meter = AverageMeter() # 
    
    step = 0
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            gc.collect()
            torch.cuda.empty_cache()
            # Flitering out some larger data due to CUDA memeory
            num_input_tokens = get_num_input_tokens(tokenizer=tokenizer,input_tokens=batch_data["input_tokens"])
            if args.truncate == False and num_input_tokens > args.max_num_input_tokens:
                continue
            else:
                print(f"{args.model_name}_{args.model_type}_{args.lora_model_name}_{args.dataset}:[{step}/{args.dataset_size},{num_input_tokens}]")
            if step >= args.dataset_size:
                break
            
            # Model output
            model_output = generate_model_output(model=model,tokenizer=tokenizer,
                                        input_tokens=batch_data["input_tokens"],
                                        generate_method=args.generate_method) # dict = ["input_ids","attentions","hidden_states", "logits"]
            
            losses = model_output["losses"]
            l_losses = []
            for idx in range(num_input_tokens-1):
                l_losses.append(losses[:,:idx+1].sum() / (idx + 1))
                print(f"[idx,{idx}]-[loss,{l_losses[idx]}]")
            l_loss_data = [[x, y] for (x, y) in zip(range(num_input_tokens-1), l_losses)]
            l_table = wandb.Table(data=l_loss_data, columns = ["x", "y"])
            token_loss_data = [[x, y] for (x, y) in zip(range(num_input_tokens), losses[0,:])]
            t_table = wandb.Table(data=token_loss_data, columns = ["x", "y"])
            wandb.log({ f"num_input_tokens": num_input_tokens,
                        f"length_loss_{step}": wandb.plot.line(l_table,"x","y",title=f"length_loss_{step}"),
                        f"token_loss_{step}": wandb.plot.line(t_table,"x","y",title=f"token_loss_{step}")})
            
            step += 1
        
    # Summary 
    wandb.summary["model_size"] = models_cfg[args.model_name][args.model_type][1]
    wandb.summary["checkpoint_step"] = args.lora_checkpoint_step
        
    # wandb end
    wandb.finish()   
    
if __name__ == "__main__":
    args = parse_args()
    main(args)