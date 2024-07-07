import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from huggingface_hub import snapshot_download
import huggingface_hub

huggingface_hub.login('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')
# huggingface_hub.login('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')

if __name__ == "__main__":
    cache_dir = "/U_20240603_ZSH_SMIL/LLM"
    # Qwen 1 
    # "Qwen/Qwen-1_8B-Chat",
    # "Qwen/Qwen-1_8B",
    # "Qwen/Qwen-7B-Chat",
    # "Qwen/Qwen-7B",
    # "Qwen/Qwen-14B-Chat",
    # "Qwen/Qwen-14B",
    # "Qwen/Qwen-72B-Chat",
    # "Qwen/Qwen-72B",
    
    repo_ids = [
                "meta-llama/Llama-2-70b-hf",
                # "meta-llama/Llama-2-70b-chat-hf",
                ]
    for repo_id in repo_ids:
        print(f"repod_id:{repo_id}")
        snapshot_download(repo_id=repo_id,cache_dir=cache_dir)
    