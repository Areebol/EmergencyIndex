import os
import gc
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import torch
import wandb
import argparse
import numpy as np
import pandas as pd
from utils import *
from tqdm import tqdm 
from datasets import load_dataset
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def parse_args():
    parser = get_parser()
    parser.add_argument("--f")
    parser.add_argument("--generate_method", default=False, type = bool,help="True use model.generate(), otherwise use model.__call__()")
    parser.add_argument("--entropy_normalize",default=True, type=bool, help="Entropy compution need to divide log(k)")
    
    args = parser.parse_args()
    return args

args = parse_args()
args.model_name = "qwen_1.5"
args.model_type = "0.5b"
models_cfg = load_config(args.models_cfg)

if args.lora: # Load finetune model by LORA
    model_cfg = models_cfg[args.model_name][args.model_type]
    model, tokenizer = load_lora_model_tokenizer(model_cfg[0],args.lora_model_dir,args.lora_model_name)
else: # Load original model
    model, tokenizer = load_model_tokenizer(models_cfg[args.model_name][args.model_type])
    
prompt = "Answer the following question as briefly as possible.\n"
nonsense = "Answer the question.\n"
dataset = [{"input_tokens":prompt + "9.11 and 9.9, which one is bigger?"},
            {"input_tokens":prompt + "which one is bigger? 9.11 and 9.9"},
            {"input_tokens":nonsense + "9.11 and 9.9, which one is bigger? "},
            {"input_tokens":nonsense + "which one is bigger? 9.11 and 9.9"},
            {"input_tokens":"which one is bigger? 9.11 and 9.9"},
            {"input_tokens":"9.11 and 9.9, which one is bigger?"},]
# dataset, preprocess = load_ds_preprocess(args)
    
# dataset = dataset.select(range(int(5))).map(preprocess,batched=False)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

save_dir = "tmp/heatmaps"
os.makedirs(save_dir,exist_ok=True)

for idx,data in enumerate(dataset):
    input_tokens = data["input_tokens"]
    # Model output
    model_output = generate_model_output(model=model,tokenizer=tokenizer,
                                input_tokens=input_tokens,
                                generate_method=args.generate_method) # dict = ["input_ids","attentions","hidden_states", "logits"]

    # Tokens
    tokens = tokenizer.tokenize(input_tokens)

    # Attentions
    last_attention = model_output["attentions"][-1].squeeze(0) # shape = (num_heads,num_tokens,num_tokens)
    last_attention = last_attention.max(0, keepdim=True).values.squeeze(0).data # shape = (num_tokens, num_tokens)
    last_attention = last_attention / last_attention.sum(-1,keepdim=True) # shape = (num_tokens,num_tokens)

    # Plot attention heatmap
    df = pd.DataFrame(last_attention.numpy(),index=tokens,columns=tokens)
    heatmap = sns.heatmap(df)
    fig = heatmap.get_figure()
    fig.savefig(f"{save_dir}/{args.model_type}_{idx}.png",dpi = 400)
    plt.close("all")
    
    # Plot test metric
    # lens = torch.tensor(range(len(tokens),0,-1))
    test_metrics = last_attention.sum(0) 
    df = pd.DataFrame(test_metrics.numpy(),index=tokens,columns=["tokens"])
    heatmap = sns.heatmap(df)
    fig = heatmap.get_figure()
    fig.savefig(f"{save_dir}/{args.model_type}_{idx}_test.png",dpi = 400)
    plt.close("all")
    