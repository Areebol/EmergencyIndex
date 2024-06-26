# -*- coding: UTF-8 -*-
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GenerationConfig
import torch
import logging

def elements_in_path(path,elements):
    """
    path: 
    elements: 
    """
    for element in elements:
        if element in path:
            return True
    return False

def load_model_tokenizer(model_config=None,half_models=['32b','34b','70b','72b']):
    """
    args:
    model_config = [model_name, model_path, model_family, model_param_size]
    half_models = models need to be loaded as half mode
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

def model_generate(tokenizer:AutoTokenizer, model:AutoModelForCausalLM, input_tokens:str, max_new_tokens:int=1):
    """
    model generate output used input_tokens
    args:
    tokenizer = 
    model = 
    input_tokens = 
    """
    gen_config = GenerationConfig(do_sample=False, num_beams=1,eos_token_id=tokenizer.eos_token_id,
                                  pad_token_id=tokenizer.eos_token_id,max_new_tokens=max_new_tokens, 
                                  output_attentions=True, return_dict_in_generate=True,
                                  output_hidden_states=True)
    with torch.no_grad():
        inputs = tokenizer(input_tokens, padding=False, return_tensors='pt')
        input_ids = inputs['input_ids'].cuda()
        assert input_ids.shape[0] == 1, "Currently only supported batch size == 1 !!!"
        attention_mask = inputs['attention_mask'].cuda()
        
        generate = model.generate(input_ids, attention_mask=attention_mask, generation_config=gen_config)
        generated_text = tokenizer.batch_decode(generate['sequences'], skip_special_tokens=True)
        model_output = {
            "text": generated_text[0],
            "input_ids": input_ids.cpu(), # shape = [batch_size,num_tokens]
            "attentions": torch.stack(generate['attentions'][0]).detach().cpu(), # shape = [num_layers,batch_size,num_heads,num_tokens,num_tokens]
            "hidden_states": torch.stack(generate["hidden_states"][0]).detach().cpu() # shape = [num_layers,batch_size,num_tokens,token_dim]
        }
        return model_output