# Emegency Index
This repository contains the code for 1:caclulating the emergency index;2:caclulating the naive entropy at the token level;
## Emergency Index


## Naive Entropy

By default, a standard run executes the following x steps in order:

* `generate_output.py`: Sample models' outputs from the models for a set of input tokens.

    models' output store will be stored in {exp_dir}/outputs.pkl;

    models' output contains a dir `model_outputs` = 
    ```
    "id": x, 
    "final_hidden_states": np.ndarray = shape(batch_size, num_tokens, token_dim), 
    "logits": np.ndarray = shape(batch_size, num_tokens, vocabulary_size), 
    "input_tokens": str,
    ```

* `compute_naive_entropy.py`: Compute naive entropy by models' outputs.

    resotre models' outputs from 
