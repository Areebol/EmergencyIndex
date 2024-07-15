import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from datasets import load_dataset
import random

PROMPTS = {
    'default': "Answer the following question as briefly as possible.\n",
    'Xsum': 'Summary the following document.\n'}

def split_dataset(dataset):
    """Get indices of answerable and unanswerable questions."""

    def clen(ex):
        return len(ex["answers"]["text"])

    answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    # union == full dataset
    assert set(answerable_indices) | set(
        unanswerable_indices) == set(range(len(dataset)))
    # no overlap
    assert set(answerable_indices) - \
        set(unanswerable_indices) == set(answerable_indices)

    return answerable_indices, unanswerable_indices

def get_make_prompt(args):
    if args.prompt_type == 'default':
        def make_prompt(context, question, answer, brief, brief_always):
            prompt = ''
            if brief_always:
                prompt += brief
            if args.use_context and (context is not None):
                prompt += f"Context: {context}\n"
            prompt += f"Question: {question}\n"
            if answer:
                prompt += f"Answer: {answer}\n\n"
            else:
                prompt += 'Answer:'
            return prompt
    else:
        raise ValueError

    return make_prompt

def construct_fewshot_prompt_from_indices(dataset, example_indices, brief, brief_always, make_prompt):
    """Given a dataset and indices, construct a fewshot prompt."""
    if not brief_always:
        prompt = brief
    else:
        prompt = ''

    for example_index in example_indices:

        example = dataset[example_index]
        context = example["context"]
        question = example["question"]
        answer = example["answers"]["text"][0]

        prompt = prompt + make_prompt(context, question, answer, brief, brief_always)

    return prompt

def load_ds_preprocess(args):
    if args.dataset == "HC3":
        dataset = load_dataset("Hello-SimpleAI/HC3","all",trust_remote_code=True,keep_in_memory=True)["train"]
        def preprocess(example):
            return {"input_tokens":f"Question:{example['question']} Human_answers:{example['human_answers'][0]}".replace(" .", ".").replace(" ? ","?").replace("\n","")}
    elif args.dataset == "Xsum":
        dataset = dataset = load_dataset("EdinburghNLP/xsum",trust_remote_code=True,keep_in_memory=True)["train"]
        def preprocess(example):
            PROMPT = PROMPTS['Xsum']
            document,summary = example['document'],example['summary']
            return {"input_tokens":f"{PROMPT}Document:{document}\nSummary:",
                    "input_tokens_wo_prompt":f"Document:{document}\nSummary:",
                    "summary_tokens":f"{summary}"}
    elif args.dataset == "TriviaQA":
        dataset = load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
        dataset = dataset.train_test_split(test_size=0.2, seed=args.seed)['train']
        answerable_indices, _ = split_dataset(dataset)
        PROMPT = PROMPTS["default"]
        random.seed(args.seed)
        prompt_indices = random.sample(answerable_indices, args.num_few_shot)
        make_prompt = get_make_prompt(args)
        prompt = construct_fewshot_prompt_from_indices(
        dataset, prompt_indices, PROMPT, True, make_prompt)
        print(f"Prompt for TriviaQA dataset is {prompt}")
        def preprocess(example):
            question, context = example["question"], example['context']
            correct_answer = example['answers']['text']
            current_input = make_prompt(context, question, None, 
                                        PROMPT, True)
            local_prompt = prompt + current_input
            print(local_prompt)
            return {"input_tokens":local_prompt,"correct_answer":correct_answer}
    else:
        raise ValueError(f"Currently not supported {args.dataset}")
    return dataset, preprocess