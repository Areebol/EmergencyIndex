import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import torch.nn.functional as F
import scipy.integrate as integrate

def pkl_save(object, exp_dir, file):
    with open(f'{exp_dir}/{file}', 'wb') as f:
        pickle.dump(object, f)
        
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
        token_features = model_output["hidden_states"][-1,:,:,:] # shape = [num_layers,batch_size,num_tokens,token_dim]
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
        num_similarity_error = sum(similarity_matrix[(similarity_matrix < -1.0)|(similarity_matrix > 1.0)]) # CosineSim belong to [-1,1]
        assert num_similarity_error==0, f"CosineSim's calculation result has {num_similarity_error} error"
        
        # Distance belong to [0,2], but only integrate on [0,1]
        distance_matrixs = (1 - similarity_matrix)
        distance_assert = torch.all((distance_matrixs >= 0.0) & (distance_matrixs <= 2.0)) # Distance belong to  [0,2]
        assert distance_assert, f"Distance matrix's calculation result has an error"
    else:
        raise ValueError(f"Currently method:{method} not supported")

    assert len(distance_matrixs.shape) == 3 and distance_matrixs.shape[-1] == distance_matrixs.shape[-2],f"Token_features's shape {distance_matrixs.shape} is not like [batch_size,num_tokens,num_tokens] "
    return distance_matrixs

def threshold_func(gamma:float, epsilon:float, x: np.ndarray):
    """
    get num of value (epsilon <= x <= gamma)
    """
    assert gamma > epsilon, f"Gamma {gamma} should be greater than epsilon {epsilon}"
    ones = (x >= epsilon) & (x <= gamma)# Distance belong to  [epsilon, gamma]
    y = np.sum(ones)
    return y.item()

def gamma_emergency_index_func(gamma:float, epsilon:float, distances: np.ndarray, num_tokens: int):
    """
    get gamma emergency index 
    """
    if num_tokens <= 1: # num_tokens <=1 means not distance matrixs available
        return 0
    return threshold_func(gamma, epsilon, distances) / (num_tokens * (num_tokens -1))

def integrate_func(func,lower,upper,args):
    result, error = integrate.quad(func, lower, upper, args=args)
    return result

def calculate_gamma_emergency_index(distance_matrixs, epsilon: float = 1e-10, gamma: float = 1.0 ):
    num_tokens = distance_matrixs.shape[-1]
    distances = distance_matrixs.numpy().reshape(-1)
    return gamma_emergency_index_func(gamma,epsilon,distances,num_tokens)

def calculate_gamma_distance(distance_matrixs, epsilon: float = 1e-10, gamma: float = 1.0 ):
    """
    Caculate Avg value (between epsilon and gamma) in distance
    """
    distances = distance_matrixs.numpy().reshape(-1)
    x = distances[((distances>=epsilon) & (distances<=gamma))]
    return sum(x) / len(x)

def calculate_emergency_index(distance_matrixs, epsilon: float = 1e-10):
    num_tokens = distance_matrixs.shape[-1]
    distances = distance_matrixs.numpy().reshape(-1)
    emergency_index = integrate_func(gamma_emergency_index_func,lower=epsilon,upper=1.0,args=(epsilon,distances,num_tokens))
    return emergency_index

def calculate_naive_entropy(probabilities: np.ndarray, normalize: bool = True):
    """
    return entropy of probabilities
    args: 
    probabilities: shape = (bs, num_tokens, vocab_size)
    ret:
    entropys: shape = (num_tokens)
    """
    probabilities = np.squeeze(probabilities,0) # shape = (num_tokens, vocab_size)
    if normalize:
        vocab_size = probabilities.shape[1] # vocab_size
        return entropy(probabilities,axis=-1) / np.log(vocab_size)
    else:
        return entropy(probabilities,axis=-1)
        

def plot_emergency_index(distance_matrixs, epsilon: float = 1e-10):
    """
    Plot image of Emergency Index func
    """
    num_tokens = distance_matrixs.shape[-1]
    distances = distance_matrixs.numpy().reshape(-1)
    gamma = np.linspace(2*epsilon, 1.0, 600)
    gamma_emergency_index = [gamma_emergency_index_func(g,epsilon,distances,num_tokens)
                             for g in gamma]
    plt.figure()
    plt.plot(gamma,gamma_emergency_index, label="gamma_emergency_index_func(gamma)")
    plt.xlabel("gamma")
    plt.ylabel("gamma emergency index")
    plt.gca().spines['top'].set_color('none')
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().tick_params(direction='in', length=6, width=0.5, colors='black', grid_color='gray', grid_alpha=0.7)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.title("gamma_emergency_index_func(gamma)")
    return plt
    
    
def plot_curve(x, y, label, x_label, y_label):
    plt.figure()
    plt.plot(x,y, label=label)
    plt.xlabel(x_label)
    plt.ylabel( y_label)
    plt.gca().spines['top'].set_color('none')
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().tick_params(direction='in', length=6, width=0.5, colors='black', grid_color='gray', grid_alpha=0.7)
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.title(label)
    return plt