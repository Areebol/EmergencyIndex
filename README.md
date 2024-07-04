# Emegency Index
This repository contains the code for 1:caclulating the emergency index;2:caclulating the naive entropy at the token level;
## Emergency Index


## Naive Entropy

By default, a standard run executes the following x steps in order:

* `generate_output`: Sample models' outputs from the models for a set of input tokens.

    models' output contains a dir `model_outputs`:
    ```
    "logits": np.ndarray = shape(batch_size, num_tokens, vocabulary_size), 
    ```

* `compute_naive_entropy`: Compute naive entropy by models' outputs.

