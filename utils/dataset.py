import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from datasets import load_dataset

def load_ds_preprocess(ds_name):
    if ds_name == "HC3":
        dataset = load_dataset("Hello-SimpleAI/HC3","all",trust_remote_code=True,keep_in_memory=True)["train"]
        def preprocess(example):
            return {"input_tokens":f"Question:{example['question']} Human_answers:{example['human_answers'][0]}".replace(" .", ".").replace(" ? ","?").replace("\n","")}
    elif ds_name == "Xsum":
        dataset = dataset = load_dataset("EdinburghNLP/xsum",trust_remote_code=True,keep_in_memory=True)["train"]
        def preprocess(example):
            return {"input_tokens":f"Document:{example['document']}\nSummary:{example['summary']}"}
    else:
        raise ValueError(f"Currently not supported {ds_name}")
    return dataset, preprocess