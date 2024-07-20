import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import datasets
import numpy as np
from typing import List, Dict
import itertools
import random

class TRECLoader():
    def __init__(self) -> None:
        self.name = 'trec_classification'
        dataset = datasets.load_dataset("trec")
        self.label = {
            0:"Abbreviation",
            1:"Entity",
            2:"Description",
            3:"Person",
            4:"Location",
            5:"Number"
        }
        self.train_data = [line for line in dataset['train']]
        self.test_data = [line for line in dataset['test']]
        
        self.ICLs = [[f"Question:{d['text'].strip().replace("\\n", " ")}\nAnswer Type:{self.label[d['coarse_label']]}" for d in self.train_data if d['coarse_label']==i] for i in range(6)]
    
    def get_ICL_examples(self,num_sample_pc:int=2,num_sample:int=20,seed:int=42):
        """
        num_sample_pc: num sample per class
        num_sample：num sample 
        """
        random.seed(seed)
        ICL_examples = []
        
        # 从每个 ICL 列表中分别选择前 num_sample 个元素
        for ICL in self.ICLs:
            ICL_examples.extend(ICL[:num_sample_pc])

        # 全排列后抽样num_example个
        ICL_examples = list(itertools.permutations(ICL_examples))
        ICL_examples = random.sample(ICL_examples, num_sample)
        
        # 拼接得到ICL examples
        ICL_examples = [
            "\n".join([example for example in perm]) 
            for perm in ICL_examples
        ]
        
        return ICL_examples

    def get_ICL_test_datas(self,ICL_demo:str=None):
        """
        ICL：ICL示例部分
        """
        ICL_test_datas = [f"{ICL_demo}\nQuestion:{d['text'].strip().replace("\\n", " ")}\nAnswer Type:" for d in self.test_data]
        labels = [self.label[d["coarse_label"]] for d in self.test_data]
        return ICL_test_datas,labels
    
if __name__ == "__main__":
    dataset = TRECLoader()
    ICL_examples = dataset.get_ICL_examples()
    print(ICL_examples)