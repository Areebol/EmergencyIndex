# -*- coding: UTF-8 -*-
import os
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GenerationConfig

def load_model_tokenizer(model_config=None,half_models=[32,34,70,72]):
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
    if model_config[1] in half_models:
        print("Loading model in half mode")
        model = AutoModelForCausalLM.from_pretrained(model_config[0], device_map="auto", torch_dtype=torch.float16, config=config, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config[0], device_map="auto", config=config,trust_remote_code=True)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    return model,tokenizer

def load_lora_model_tokenizer(base_model_path,lora_model_base_dir,
                              lora_model_name = 'checkpoint-1000'):
    """
    load model tokenizer from base model + lora dir
    args:
    ret:
    model
    tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, fast_tokenizer=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(base_model_path, output_attentions=True, attn_implementation="eager", trust_remote_code=True)
    model = merge_lora_model(base_model_path,lora_model_base_dir,lora_model_name,config=config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(len(tokenizer))
    return model,tokenizer

def merge_lora_model(base_model_path,lora_model_base_dir,
                     lora_model_name = 'checkpoint-1000',
                     config=None):
    model_class, _ = (AutoModelForCausalLM, AutoTokenizer)
    """
    Merge base model + lora model
    """
    print(f"Loading LoRA for causal language mode\nBase model_path:{base_model_path}")
    print(f"Lora model_path:{lora_model_base_dir}/{lora_model_name}")
    base_model = model_class.from_pretrained(
        base_model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        config = config,
    )
    new_model: PeftModel = PeftModel.from_pretrained(
        base_model,
        os.path.join(lora_model_base_dir, lora_model_name),
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return new_model.merge_and_unload()

def generate_model_output(model:AutoModelForCausalLM, tokenizer:AutoTokenizer, input_tokens:str,generate_method=False):
    """
    generate model output used input_tokens
    args:
    tokenizer
    model
    input_tokens = model_input
    ret:
    model_output = dict["input_ids","attentions","hidden_states","logits"] all in cpu
    """
    inputs = tokenizer(input_tokens, padding=False, return_tensors='pt')
    input_ids = inputs['input_ids'].cuda()
    assert input_ids.shape[0] == 1, "Currently only supported batch size == 1!"
    attention_mask = inputs['attention_mask'].cuda()
    
    if generate_method: # Use generate strategy
        gen_config = GenerationConfig(do_sample=False, 
                                        num_beams=1,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.eos_token_id,
                                        max_new_tokens=1, 
                                        return_dict_in_generate=True,
                                        output_attentions=True, 
                                        output_hidden_states=True,
                                        output_logits=True)
        output = model.generate(input_ids, attention_mask=attention_mask, generation_config=gen_config)
        output_hidden_states = output["hidden_states"][0] # tuple of (num_layers + 1) * [batch_size, num_tokens, token_dim]
        output_attentions = output['attentions'][0] # tuple of num_layers * [batch_size, num_heads,num_tokens, num_tokens]
        output_logits = output['logits'][0].cpu().numpy().reshape((1,1,-1)) # only output the last token's logits [batch_size, vocab_size]; like Raw output's output_logits[:,-1,:]
    else: # Raw output
        output = model(input_ids, 
                        attention_mask=attention_mask,
                        output_attentions=True, 
                        output_hidden_states=True)
        output_hidden_states = output.hidden_states # tuple of (num_layers + 1) * [batch_size, num_tokens, token_dim]
        output_attentions = output.attentions # tuple of num_layers * [batch_size, num_heads,num_tokens, num_tokens]   
        output_logits = output.logits.cpu().numpy() # [batch_size, num_tokens, vocab_size] 
    
    # Cuda OOM issues: hidden_states may be too large, unable to convert to cpu at once
    hidden_states = []
    for hidden_state in output_hidden_states:
        assert len(hidden_state.shape) == 3, f"hidden_state's shape {hidden_state.shape} should be like [bs,num_tokens,token_dim]"
        hidden_states.append(hidden_state.cpu())
        del hidden_state
    attentions = []
    for attention in output_attentions:
        assert len(attention.shape) == 4, f"attention's shape {attention.shape} should be like [bs,num_heads,num_tokens,num_tokens]"
        attentions.append(attention.cpu())          
        del attention

    model_output = {
        "input_ids": input_ids.cpu(), # shape = [batch_size,num_tokens]
        "attentions": torch.stack(attentions), # shape = [num_layers,batch_size,num_heads,num_tokens,num_tokens]
        "hidden_states": torch.stack(hidden_states), # shape = [num_layers,batch_size,num_tokens,token_dim]
        "logits": output_logits, # numpy array, shape = [batch_size, num_tokens, vocab_size]
    }
    del output
    return model_output
    
def get_num_input_tokens(tokenizer:AutoTokenizer, input_tokens:str):
    with torch.no_grad():
        inputs = tokenizer(input_tokens, padding=False, return_tensors='pt')
        num_input_tokens = inputs['input_ids'].shape[-1]
        del inputs
        return num_input_tokens