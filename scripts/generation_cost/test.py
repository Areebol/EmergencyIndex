import os
import gc
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import torch
import time
import numpy as np
import pandas as pd
from utils import *
from tqdm import tqdm 

def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    return args

@torch.no_grad()
def main(args):
    models_cfg = load_config(args.models_cfg)
    model, tokenizer = load_model_tokenizer(models_cfg[args.model_name][args.model_type])
    
    input_tokens = "9.11 and 9.9, which one is bigger?"
    inputs = tokenizer(input_tokens, padding=False, return_tensors='pt')
    input_ids = inputs['input_ids'].cuda()
    attention_mask = inputs['attention_mask'].cuda()
    df_data = []
    for num_beams in tqdm(range(1,22,5)):
        for gen_length in tqdm(range(20,51,5)):
            settings = [
                        {"num_beams":num_beams,"do_sample":True,"repeat_times":1, "type":"beam_sample"}, 
                        {"num_beams":num_beams,"do_sample":False,"repeat_times":1,"type":"beam_search"}, 
                        {"num_beams":1,"do_sample":True,"repeat_times":num_beams,"type":"sample"}, 
                        {"num_beams":1,"do_sample":False,"repeat_times":num_beams,"type":"greedy_search"}, 
                        ]
            meters = [AverageMeter() for i in range(len(settings))]
            repeat_time = 10
            for i in range(repeat_time): 
                for index, setting in enumerate(settings):
                    gen_config = GenerationConfig(
                        # Parameters that control the generation strategy used
                        do_sample=setting["do_sample"], num_beams=setting["num_beams"],num_return_sequences=setting["num_beams"],
                        # Parameters that control the length of the output
                        max_length=gen_length, min_length=gen_length,
                        # Parameters that define the output variables of generate
                        output_attentions=False, return_dict_in_generate=True, output_hidden_states=False,output_scores=False, 
                        # Special tokens that can be used at generation time
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id
                        )
                    start_time = time.time()
                    for sample_id in range(setting["repeat_times"]):
                        generate = model.generate(input_ids, attention_mask=attention_mask, generation_config=gen_config)
                    meters[index].update(time.time() - start_time)
                    del generate
                    print(f"{i:2d}/{repeat_time} |{setting["type"]:<18s}|avg {meters[index].avg:.4f}|cur {meters[index].val:.4f}")
                    if i == repeat_time - 1:
                        df_data.append([setting["type"],num_beams,gen_length,meters[index].val])
        
    df = pd.DataFrame(df_data,columns=['Type','num_beams',"gen_length","cost"])
    df.sort_values(by=['Type','num_beams', 'gen_length'], ascending=False)
    df.to_csv('generation_costs.csv', index = True)

if __name__ == "__main__":
    args = parse_args()
    main(args)