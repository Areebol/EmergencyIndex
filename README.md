# Emegency Index
This repository contains the code for 1:caclulating the emergency index;2:caclulating the naive entropy at the token level;
## Emergency Index


## Naive Entropy

By default, a standard run executes the following x steps in order:

* `generate_output`: Sample model's outputs from the model for a set of input tokens.

    model's output contains a dir `model_outputs`:
    ```
    "logits": np.ndarray = shape(batch_size, num_tokens, vocabulary_size), 
    ```

* `compute_naive_entropy`: Compute naive entropy by model's outputs.


## Block Entropy

* `generate_output`: Sample model's ouputs from the model by beam search

* `compute_block_entropy`: Compute block entropy by model's ouputs.