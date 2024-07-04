import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from datasets import load_dataset
import wandb
import torch
import argparse
from tqdm import tqdm 
from utils import *
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_cfg", default="./config/models_pz.yaml", help="model's weight config")
    parser.add_argument("--generate_method", default=False, type = bool,help="True use model.generate(), otherwise use model.__call__()")
    parser.add_argument("--model_name", default="qwen_1.5", type=str,help="LLM model family")
    parser.add_argument("--model_type", default="0.5b", type=str,help="LLM model type")
    parser.add_argument("--dataset", default="HC3", choices=["HC3","Xsum"], type=str,help="DataSet")
    parser.add_argument("--dataset_size", default=200, type=int,help="DataSet size")
    parser.add_argument("--max_num_input_tokens", default=1000, type=int,help="Max num of input otkens be allowed")
    parser.add_argument("--log_image_interval", default=10, type=int, help="Step interval to log Image")
    parser.add_argument("--lora", default=False,type=bool,help="True to use lora model")
    parser.add_argument("--lora_model_dir",default ="/U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-llama2-7b-epoch2-v1", type=str,help="Lora checkpoint's path")
    parser.add_argument("--lora_model_name",default="", type=str, help="Specify lora chckpoint version")
    parser.add_argument("--lora_checkpoint_step",default=1, type=int, help="Specify lora chckpoint step")
    parser.add_argument("--experiment_name",default=None, type=str, help="Specify experiment name")
    
    args = parser.parse_args()
    return args

def main(args):
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
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            model_output = generate_model_output(model=model,tokenizer=tokenizer,
                                                input_tokens=batch_data["input_tokens"],
                                                generate_method=args.generate_method) # dict = ["input_ids","attentions","hidden_states"]
    
if __name__ == "__main__":
    args = parse_args()
    main(args)