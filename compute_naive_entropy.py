import os
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
    parser.add_argument("--dataset", default="HC3", choices=["HC3","Xsum"], type=str,help="DataSet")
    parser.add_argument("--dataset_size", default=200, type=int,help="DataSet size")
    parser.add_argument("--max_num_input_tokens", default=1000, type=int,help="Max num of input otkens be allowed")
    parser.add_argument("--gammas", default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], type=list,help="emergency index gamma")
    parser.add_argument("--log_image_interval", default=10, type=int, help="Step interval to log Image")
    parser.add_argument("--lora", default=False,type=bool,help="True to use lora model")
    parser.add_argument("--lora_model_dir",default ="/U_20240603_ZSH_SMIL/MedicalGPT/outputs-sft-llama2-7b-epoch2-v1", type=str,help="Lora checkpoint's path")
    parser.add_argument("--lora_model_name",default="", type=str, help="Specify lora chckpoint version")
    parser.add_argument("--lora_checkpoint_step",default=1, type=int, help="Specify lora chckpoint step")
    
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

    # Avg meters: gammas[emergency_index, distance], others
    # TODO naive avg entropy
    gammas_avg_NaEntropy_meters = {gamma:AverageMeter() for gamma in args.gammas}
    
    # Wandb Config
    wandb.login(key=args.wandb_key)  # wandb api key
    if args.lora:
        runs = wandb.init(project='NaiveEntropy',mode=args.wandb_mode,save_code=True,
                      name=f"{args.model_name}_{args.model_type}_lora_{args.lora_model_name}_{args.dataset}")
    else:
        runs = wandb.init(project='NaiveEntropy',mode=args.wandb_mode,save_code=True,
                      name=f"{args.model_name}_{args.model_type}_{args.dataset}")
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
                                        generate_method=args.generate_method) # dict = ["input_ids","attentions","hidden_states", "logits"]
            
            # Model logits
            logits = model_output["logits"] # shape = (bs, num_tokens, vocab_size)
            
            # Predict probabilities
            pred_probs = softmax(logits,axis=-1) # shape = (bs, num_tokens, vocab_size)
            
            # Naive entropy
            naive_entropys = calculate_naive_entropy(pred_probs) # shape = (num_tokens) value belong to [0,1]
            
            # Update
            # Gammas: proportion of naive_entropys's avg
            for gamma in args.gammas:
                gamma_entropys = naive_entropys[int((1-gamma)*len(naive_entropys)):]
                gammas_avg_NaEntropy_meters[gamma].update(np.mean(gamma_entropys))
            
            # Gammas Log : avg_NaEntropy
            cur_log = {**{"num_input_tokens": num_input_tokens, },
                       **{f"gamma_{gamma}_avg_NaEntropy":gammas_avg_NaEntropy_meters[gamma].val
                       for gamma in args.gammas},
                       }
            if step % args.log_image_interval == 0:
                # Plot naive entropy Image
                naive_entropy_image = plot_curve(np.arange(len(naive_entropys)),naive_entropys,
                                                 label="Naive Entropy",
                                                 x_label="token_id",y_label="entropy")
                cur_log = {**cur_log,
                           "Image naive entropy":wandb.Image(naive_entropy_image)}
            wandb.log(cur_log)
            step += 1
        
    # Summary 
    wandb.summary["model_size"] = models_cfg[args.model_name][args.model_type][1]
    wandb.summary["checkpoint_step"] = args.lora_checkpoint_step
    for gamma in args.gammas:
        wandb.summary[f"avg/gamma_{gamma}_avg_NaEntropy"] = gammas_avg_NaEntropy_meters[gamma].avg
        
    # wandb end
    wandb.finish()   
    
if __name__ == "__main__":
    args = parse_args()
    main(args)