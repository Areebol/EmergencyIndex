import os
import wandb
import pickle
import argparse
import numpy as np

def restore_file(wandb_id, filename='generations.pkl'):
    files_dir = f'tmp/beam_probability/{wandb_id}/restored_files'    
    os.system(f'mkdir -p {files_dir}')

    api = wandb.Api()
    run = api.run(f'BeamProbability/{wandb_id}')

    path = f'{files_dir}/{filename}'
    run.file(filename).download(root=files_dir, replace=False, exist_ok=True)
    with open(f"{path}",'rb') as f:
        results_old = pickle.load(f)
        
    return results_old

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_key', default='aecdc69b1a817efc605df2d5be9dd7face113d04', help='wandb auth api key')
    parser.add_argument('--wandb_id', default='tppmacl8', help='wandb run id')
    parser.add_argument("--wandb_mode", default="offline", choices=["offline","online"],help="Wandb log mode")
    args = parser.parse_args()
    
    wandb.login(key=args.wandb_key)  # wandb api key
    wandb.init(project="BeamProbability",entity="areebol",id=args.wandb_id,mode=args.wandb_mode,resume="must")
    
    generations = restore_file(args.wandb_id)
    
    for generation in generations.values():
        log_liks_s = []
        for log_liks in generation["token_log_liks"]:
            log_liks = [-1000 if log_lik==float("-inf") else log_lik for log_lik in log_liks]
            log_liks_s.append(log_liks)
            
        # Normalize log like
        log_liks_agg = [np.mean(log_liks) for log_liks in log_liks_s]
        print(log_liks_agg)
    
if __name__ == "__main__":
    main()
    