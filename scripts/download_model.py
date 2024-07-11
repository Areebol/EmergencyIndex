import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from huggingface_hub import snapshot_download
import huggingface_hub


def main():
    cache_dir = "/U_20240603_ZSH_SMIL/LLM"

    repo_ids = [
                "meta-llama/Llama-2-70b-hf",
                ]
    for repo_id in repo_ids:
        print(f"repod_id:{repo_id}")
        snapshot_download(repo_id=repo_id,cache_dir=cache_dir)
    
if __name__ == "__main__":
    # huggingface_hub.login('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')
    # huggingface_hub.login('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')
    huggingface_hub.login('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',add_to_git_credential=True)
    main()
