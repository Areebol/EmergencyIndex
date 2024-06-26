import torch
def extract_token_features(model_output:dir = None, method:str = None):
    """
    extract token features from model output
    args: 
    model_output = ["text","input_ids","attentions","hidden_states"]
    method = ["a","b"]
    ret:
    token_features = [batch_size,num_tokens,token_dim]
    """
    # Temp code
    if method == "a":
        token_features = model_output["hidden_states"][-1,:,:,:] # shape = [batch_size,num_tokens,token_dim]
    elif method == "b":
        token_features = model_output["hidden_states"][-1,:,:,:] # shape = [batch_size,num_tokens,token_dim]
    else:
        ValueError(f"Currently method:{method} not supported")
    assert len(token_features.shape) == 3, f"Token_features's shape {token_features.shape} is not like [batch_size,num_tokens,token_dim] "
    return token_features
    
def calculate_distance_matrixs(token_features:torch.Tensor):
    """
    calculate a distance matrix by token features
    args: 
    token_features = [batch_size,num_tokens,token_dim]
    ret:
    distance_matrixs = [batch_size,num_tokens,num_tokens]
    """
    # Temp code
    return torch.tensor(range(16)).reshape(-1,4,4)