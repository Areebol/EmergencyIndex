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
    parser.add_argument("--generate_method", default=False, type = bool,help="True use model.generate(), otherwise use model.__call__()")
    parser.add_argument("--gammas", default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], type=list,help="emergency index gamma")
    parser.add_argument("--log_image_interval", default=10, type=int, help="Step interval to log Image")
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
    
    dataset, preprocess = load_ds_preprocess(args)
    
    dataset = dataset.select(range(int(args.dataset_size * 1.5))).map(preprocess,batched=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Avg meters: gammas[emergency_index, distance], others
    gammas_avg_NaEntropy_meters = {gamma:AverageMeter() for gamma in args.gammas}
    wop_gammas_avg_NaEntropy_meters = {gamma:AverageMeter() for gamma in args.gammas}
    gammas_meters = {"input_tokens":gammas_avg_NaEntropy_meters,"input_tokens_wo_prompt":wop_gammas_avg_NaEntropy_meters}
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
            for input_type in ["input_tokens","input_tokens_wo_prompt"]:
                # Model output
                model_output = generate_model_output(model=model,tokenizer=tokenizer,
                                            input_tokens=batch_data[input_type],
                                            generate_method=args.generate_method) # dict = ["input_ids","attentions","hidden_states", "logits"]
                
                # Model logits
                logits = model_output["logits"] # shape = (bs, num_tokens, vocab_size)
                
                # Predict probabilities
                pred_probs = softmax(logits,axis=-1) # shape = (bs, num_tokens, vocab_size)
                del logits
                
                # Naive entropy
                naive_entropys = calculate_naive_entropy(pred_probs,normalize=args.entropy_normalize) # shape = (num_tokens) value belong to [0,1]
                del pred_probs
                
                # Update
                # Gammas: proportion of naive_entropys's avg
                for gamma in args.gammas:
                    gamma_entropys = naive_entropys[int((1-gamma)*len(naive_entropys)):]
                    gammas_meters[input_type][gamma].update(np.mean(gamma_entropys))
            
            # Gammas Log : avg_NaEntropy
            cur_log = {**{"num_input_tokens": num_input_tokens, },
                       **{f"gamma_{gamma}_avg_NaEntropy":gammas_meters["input_tokens"][gamma].val
                       for gamma in args.gammas},
                       **{f"wop_gamma_{gamma}_avg_NaEntropy":gammas_meters["input_tokens_wo_prompt"][gamma].val
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
        wandb.summary[f"avg/wop_gamma_{gamma}_avg_NaEntropy"] = wop_gammas_avg_NaEntropy_meters[gamma].avg
        
    # wandb end
    wandb.finish()   
    
if __name__ == "__main__":
    args = parse_args()
    main(args)