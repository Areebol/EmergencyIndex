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
    parser.add_argument("--models_cfg", default="./config/models_pz.yaml", help="model's weight config")
    parser.add_argument("--generate_method", default=False, type = bool,help="True use model.generate(), otherwise use model.__call__()")
    parser.add_argument("--model_name", default="qwen_1.5", type=str,help="LLM model family")
    parser.add_argument("--model_type", default="0.5b", type=str,help="LLM model type")
    parser.add_argument("--dataset", default="Xsum", choices=["HC3","Xsum"], type=str,help="DataSet")
    parser.add_argument("--dataset_size", default=200, type=int,help="DataSet size")
    parser.add_argument("--max_num_input_tokens", default=950, type=int,help="Max num of input otkens be allowed")
    parser.add_argument("--gammas", default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], type=list,help="emergency index gamma")
    parser.add_argument("--log_image_interval", default=10, type=int, help="Step interval to log Image")
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
    
    # Wandb Config
    wandb.login(key=args.wandb_key)  # wandb api key
    if args.lora:
        runs = wandb.init(project='VocabTokenEntropy',mode=args.wandb_mode,save_code=True,
                      name=f"{args.model_name}_{args.model_type}_lora_{args.lora_model_name}_{args.dataset}")
    else:
        runs = wandb.init(project='VocabTokenEntropy',mode=args.wandb_mode,save_code=True,
                      name=f"{args.model_name}_{args.model_type}_{args.dataset}")
    wandb.config.update(args)
    
    vocab_size = tokenizer.vocab_size
    token_entropy_meter = AverageMeter()
    
    with torch.no_grad():
        for token_id in range(vocab_size):
            gc.collect()
            torch.cuda.empty_cache()
            
            input_tokens = tokenizer.decode(token_id)
            
            print(f"{args.model_name}_{args.model_type}_{args.lora_model_name}:[{token_id}/{vocab_size}]")
            # Model output
            model_output = generate_model_output(model=model,tokenizer=tokenizer,
                                        input_tokens=input_tokens,
                                        generate_method=args.generate_method) # dict = ["input_ids","attentions","hidden_states", "logits"]
            
            # Model logits
            logits = model_output["logits"] # shape = (bs, num_tokens, vocab_size)
            
            # Predict probabilities
            pred_probs = softmax(logits,axis=-1) # shape = (bs, num_tokens, vocab_size)
            del logits
            
            # Naive entropy
            naive_entropys = calculate_naive_entropy(pred_probs,normalize=args.entropy_normalize) # shape = (num_tokens) value belong to [0,1]
            token_entropy_meter.update(naive_entropys.item())
           
            del pred_probs
            
            # Gammas Log : avg_NaEntropy
            cur_log = {**{"token_id": token_id, },
                       **{"token_entropy": token_entropy_meter.val },
                        }
            wandb.log(cur_log)
        
    # Summary 
    wandb.summary["model_size"] = models_cfg[args.model_name][args.model_type][1]
    wandb.summary["checkpoint_step"] = args.lora_checkpoint_step
    wandb.summary["avg/token_entropy"] = token_entropy_meter.avg
        
    # wandb end
    wandb.finish()   
    
if __name__ == "__main__":
    args = parse_args()
    main(args)