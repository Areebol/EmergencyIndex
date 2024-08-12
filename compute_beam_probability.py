import os
import gc
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import torch
import wandb
import argparse
import numpy as np
from utils import *
from tqdm import tqdm 
import altair as alt
import pandas as pd
from datasets import load_dataset
from scipy.special import softmax
from torch.utils.data import DataLoader

def parse_args():
    parser = get_parser()
    
    parser.add_argument("--project", default="BeamProbability",help="Wandb project name")
    parser.add_argument("--entropy_normalize",default=True, type=bool, help="Entropy compution need to divide log(k)")
    parser.add_argument("--max_new_tokens", default=50, type=int,help="Max num of new tokens generated")
    parser.add_argument("--num_beams", default=10, type=int,help="Num of max path to beam search")
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
    
    generations = {}
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
            
            token_log_liks = generate_token_log_liks(tokenizer,model,batch_data["input_tokens"],max_new_tokens=args.max_new_tokens,num_beams=args.num_beams,truncate=args.truncate,num_max_input_tokens=args.max_num_input_tokens)  # list = beam_num * list( token_num * [log_lik])
            token_log_liks = generate_token_log_liks(tokenizer,model,batch_data["input_tokens"],max_new_tokens=args.max_new_tokens,num_beams=args.num_beams,truncate=args.truncate,num_max_input_tokens=args.max_num_input_tokens)  # list = beam_num * list( token_num * [log_lik])
            generations[step] = {"token_log_liks":token_log_liks}
            step += 1
    with open(f'{wandb.run.dir}/{f'generations.pkl'}', 'wb') as f:
        pickle.dump(object, f)
    wandb.save(f'{wandb.run.dir}/{f'generations.pkl'}') 
    # Summary 
    wandb.summary["model_size"] = models_cfg[args.model_name][args.model_type][1]
    wandb.summary["checkpoint_step"] = args.lora_checkpoint_step
    
    # # Plot mean +- std
    # data = [[x, y, y - std, y + std] for x, y, std in zip(range(len(means)), means, stds)]
    # table = wandb.Table(data=data, columns=["x", "mean", "mean-std", "mean+std"])
    # wandb.log({"mean_std_plot": wandb.plot.line(table, "x", "mean", title="Mean ± Std Dev")})
    # import matplotlib.pyplot as plt

    # plt.fill_between(range(len(means)), np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)
    # plt.plot(range(len(means)), np.array(means))
    # plt.title("Mean ± Std Dev")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")

    # # 保存图像并上传到W&B
    # plt.savefig("mean_std_plot.png")
    # wandb.log({"mean_std_plot_image": wandb.Image("mean_std_plot.png")})

    # wandb end
    wandb.finish()   
    
if __name__ == "__main__":
    args = parse_args()
    main(args)