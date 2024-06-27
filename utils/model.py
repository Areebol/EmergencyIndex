# -*- coding: UTF-8 -*-
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GenerationConfig
import torch

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
    load model tokenizer from model config
    args:
    model_config = [model_path,model_param_size]
    half_models = models need to be loaded as half mode
    ret:
    model
    tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_config[0], fast_tokenizer=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_config[0], output_attentions=True, attn_implementation="eager", trust_remote_code=True)
    if elements_in_path(model_config[0],half_models):
        print("Loading model in half mode")
        model = AutoModelForCausalLM.from_pretrained(model_config[0], device_map="auto", torch_dtype=torch.float16, config=config, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config[0], device_map="auto", config=config,trust_remote_code=True)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    return model,tokenizer

def generate_model_output(model:AutoModelForCausalLM, tokenizer:AutoTokenizer, input_tokens:str, max_new_tokens:int=1):
    """
    generate model output used input_tokens
    args:
    tokenizer
    model
    input_tokens = model_input
    ret:
    model_output = dict["text","input_ids","attentions","hidden_states"] all in cpu
    """
    gen_config = GenerationConfig(do_sample=False, num_beams=1,eos_token_id=tokenizer.eos_token_id,
                                  pad_token_id=tokenizer.eos_token_id,max_new_tokens=max_new_tokens, 
                                  output_attentions=True, return_dict_in_generate=True,
                                  output_hidden_states=True)
    with torch.no_grad():
        inputs = tokenizer(input_tokens, padding=False, return_tensors='pt')
        input_ids = inputs['input_ids'].cuda()
        assert input_ids.shape[0] == 1, "Currently only supported batch size == 1!"
        attention_mask = inputs['attention_mask'].cuda()
        
        generate = model.generate(input_ids, attention_mask=attention_mask, generation_config=gen_config)
        hidden_states = []
        
        # Cuda OOM issues: hidden_states may be too large, unable to convert to cpu at once
        for hidden_state in generate["hidden_states"][0]:
            hidden_states.append(hidden_state.cpu())
            del hidden_state
        attentions = []
        for attention in generate['attentions'][0]:
            attentions.append(attention.cpu())
            del attention

        generated_text = tokenizer.batch_decode(generate['sequences'], skip_special_tokens=True)[0]
        model_output = {
            "text": generated_text,
            "input_ids": input_ids.cpu(), # shape = [batch_size,num_tokens]
            "attentions": torch.stack(attentions), # shape = [num_layers,batch_size,num_heads,num_tokens,num_tokens]
            "hidden_states": torch.stack(hidden_states) # shape = [num_layers,batch_size,num_tokens,token_dim]
        }
        del generate
        return model_output