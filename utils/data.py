import torch
import torch.nn.functional as F
import scipy.integrate as integrate

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
    if method == "FinalOutput":
        assert len(model_output["hidden_states"].shape) == 4, f"Hidden_states' shape {model_output["hidden_states"].shape} is not like [num_layers,batch_size,num_heads,num_tokens,num_tokens]"
        token_features = model_output["hidden_states"][-1,:,:,:] # shape = [batch_size,num_tokens,token_dim]
    # elif method == "b":
    #     token_features = model_output["hidden_states"][-1,:,:,:] # shape = [batch_size,num_tokens,token_dim]
    else:
        raise ValueError(f"Currently method:{method} not supported")
    assert len(token_features.shape) == 3, f"Token_features's shape {token_features.shape} is not like [batch_size,num_tokens,token_dim] "
    return token_features
    
def calculate_distance_matrixs(token_features:torch.Tensor, method:str = None):
    """
    calculate a distance matrix by token features
    args: 
    token_features = [batch_size,num_tokens,token_dim]
    ret:
    distance_matrixs = [batch_size,num_tokens,num_tokens]
    """
    assert len(token_features.shape) == 3, f"Input token_features's shape {token_features.shape} is not like [batch_size,num_tokens,token_dim] "
    if method == "CosineSim":
        token_features_norm = F.normalize(token_features, p=2, dim=-1)
        similarity_matrix = torch.einsum('bij,bkj->bik', token_features_norm, token_features_norm) 
        
        # Precision issue: similarity_matrix's diag is not a ones like diag
        num_tokens = similarity_matrix.size(1)
        diagonal_ones = torch.diag(torch.ones(num_tokens, device=similarity_matrix.device))
        similarity_matrix = similarity_matrix - torch.diag(torch.diagonal(similarity_matrix, dim1=-2, dim2=-1).squeeze(0)) + diagonal_ones
        similarity_assert = torch.all((similarity_matrix >= -1.0) & (similarity_matrix <= 1.0)) # CosineSim belong to [-1,1]
        assert similarity_assert, f"CosineSim's calculation result has an error"
        
        # Distance
        distance_matrixs = (1 - similarity_matrix) / 2
        distance_assert = torch.all((distance_matrixs >= 0.0) & (distance_matrixs <= 1.0)) # Distance belong to  [0,1]
        assert distance_assert, f"Distance matrix's calculation result has an error"
    else:
        raise ValueError(f"Currently method:{method} not supported")

    assert len(distance_matrixs.shape) == 3 and distance_matrixs.shape[-1] == distance_matrixs.shape[-2],f"Token_features's shape {distance_matrixs.shape} is not like [batch_size,num_tokens,num_tokens] "
    return distance_matrixs

def threshold_func(gamma, epsilon, x: torch.tensor = None):
    """
    get num of value (epsilon <= x <= gamma)
    """
    assert gamma > epsilon, f"Gamma {gamma} should be greater than epsilon {epsilon}"
    ones = (x >= epsilon) & (x <= gamma)# Distance belong to  [epsilon, gamma]
    y = torch.sum(ones)
    return y.item()

def calculate_gamma_emergency_index(gamma, epsilon, x, num_tokens):
    return threshold_func(gamma, epsilon, x) / (num_tokens * (num_tokens -1))

def integrate_func(func,lower,upper,args):
    result, error = integrate.quad(func, lower, upper, args=args)
    return result

def calculate_emergency_index(distance_matrixs, epsilon: float = 1e-7, gamma: float = 1.0 ):
    num_tokens = distance_matrixs.shape[-1]
    if num_tokens == 1:
        return 0
    x = distance_matrixs.reshape(-1)
    gamma_emergency_index = calculate_gamma_emergency_index(gamma, epsilon, x, num_tokens)
    emergency_index = integrate_func(calculate_gamma_emergency_index,lower=epsilon,upper=1.0,args=(epsilon,x,num_tokens))
    return gamma_emergency_index, emergency_index