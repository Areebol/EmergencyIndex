from utils import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_cfg", default="./config/models_jq.yaml", help="model's weight config")
    args = parser.parse_args()
    
    models_cfg = load_config(args.models_cfg)
    model, tokenizer = load_model_tokenizer(models_cfg[f"paths_qwen_1.5"][0])
    data_loader = ...
    
    for batch_idx, batch_data in enumerate(data_loader):

        # Model input

        # Extract Features

        # Distance Matrix

        # Emergency Index
        ...