# -*- coding: UTF-8 -*-
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import logging
import torch
import os

def elements_in_path(path,elements):
    """
    path: 
    elements: 
    """
    for element in elements:
        if element in path:
            return True
    return False

def load_model_tokenizer(model_config=None,half_models=['12b','13b','14b','32b','34b','70b','72b']):
    """
    model_config: [model_name, model_path, model_family, model_param_size]
    half_models: models need to be loaded as half mode
    """
    tokenizer = AutoTokenizer.from_pretrained(model_config[1], fast_tokenizer=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_config[1], output_attentions=True, attn_implementation="eager", trust_remote_code=True)
    if elements_in_path(model_config[0],half_models):
        logging.info(f"Loading model [{model_config[0]}] in half mode")
        model = AutoModelForCausalLM.from_pretrained(model_config[1], device_map="auto", torch_dtype=torch.float16, config=config, trust_remote_code=True)
    else:
        logging.info(f"Loading model [{model_config[0]}] in full mode")
        model = AutoModelForCausalLM.from_pretrained(model_config[1], device_map="auto", config=config,trust_remote_code=True)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    return model,tokenizer