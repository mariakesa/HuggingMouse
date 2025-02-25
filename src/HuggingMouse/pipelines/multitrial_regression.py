import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from HuggingMouse.pipelines.base import Pipeline
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
import os
from HuggingMouse.utils import make_container_dict
import pandas as pd
from HuggingMouse.make_embeddings import MakeEmbeddings
import pickle
import plotly.express as px
from transformers import AutoImageProcessor

class ZIGRegression(nn.Module):
    def __init__(self, ViTEmbDim, NNeurons, gen_nodes, factor):
        """
        Args:
            NNeurons (int): Number of output dimensions.
            ViTEmbDim (int): Number of input dimensions.
            gen_nodes (int): Number of hidden units in the hidden layers.
            factor (array-like or torch.Tensor): Constant used for 'loc'. 
                Should have shape (yDim,).
        """
        super(ZIGRegression, self).__init__()
        
        # Define the layers:
        self.fc1 = nn.Linear(ViTEmbDim, gen_nodes)
        self.fc2 = nn.Linear(gen_nodes, gen_nodes)
        self.fc_theta = nn.Linear(gen_nodes, NNeurons)
        self.fc_p = nn.Linear(gen_nodes, NNeurons)
        
        # Initialize weights with uniform distribution:
        rangeRate1 = 1.0 / math.sqrt(ViTEmbDim)
        rangeRate2 = 1.0 / math.sqrt(gen_nodes)
        nn.init.uniform_(self.fc1.weight, -rangeRate1, rangeRate1)
        nn.init.uniform_(self.fc2.weight, -rangeRate2, rangeRate2)
        nn.init.uniform_(self.fc_theta.weight, -rangeRate2, rangeRate2)
        nn.init.uniform_(self.fc_p.weight, -rangeRate2, rangeRate2)
        
        # Initialize biases to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc_theta.bias)
        nn.init.zeros_(self.fc_p.bias)
        
        # Create learnable parameter logk (initialized as zeros)
        self.logk = nn.Parameter(torch.zeros(NNeurons))
        
        # 'loc' comes from factor. Register it as a buffer so that it is not updated by the optimizer.
        if not torch.is_tensor(factor):
            factor = torch.tensor(factor, dtype=torch.float32)
        self.register_buffer('loc', factor)
        
    def forward(self, X, Y=None):
        # Pass input through the network with tanh activations:
        full1 = torch.tanh(self.fc1(X))
        full2 = torch.tanh(self.fc2(full1))
        full_theta = self.fc_theta(full2)
        full_p = self.fc_p(full2)
        
        # Compute predictions:
        theta = torch.exp(full_theta)
        p = torch.sigmoid(full_p)  # equivalent to exp(full_p)/(1+exp(full_p))
        
        # Compute k (learnable) and get loc (constant)
        k = torch.exp(self.logk) + 1e-7  # shape: (yDim,)
        
        # Compute rate with proper broadcasting:
        rate = (theta * k.unsqueeze(0) + self.loc.unsqueeze(0)) * p
        
        # If no target is provided, return predictions:
        if Y is None:
            return theta, k, p, self.loc, rate
        
        # Otherwise, compute the entropy loss:
        Nsamps = Y.shape[0]
        # Create a mask of non-zero elements in Y:
        mask = (Y != 0)
        
        # Expand k and loc to match Y's shape (Nsamps, yDim)
        k_NTxD = k.unsqueeze(0).expand(Nsamps, -1)
        loc_NTxD = self.loc.unsqueeze(0).expand(Nsamps, -1)
        
        # Select the nonzero entries:
        y_temp = Y[mask]
        r_temp = theta[mask]
        p_temp = p[mask]
        k_temp = k_NTxD[mask]
        loc_temp = loc_NTxD[mask]
        
        # Adjust for numerical stability:
        eps = 1e-6
        p_temp = p_temp * (1 - 2e-6) + 1e-6
        r_temp = r_temp + eps
        # Clamp the difference (y_temp - loc_temp) to avoid log(0) or log(negative)
        delta = torch.clamp(y_temp - loc_temp, min=eps)
        
        LY1 = torch.sum(torch.log(p_temp) - k_temp * torch.log(r_temp) - (y_temp - loc_temp) / r_temp)
        LY2 = torch.sum(-torch.lgamma(k_temp) + (k_temp - 1) * torch.log(delta))
        
        # For entries where Y == 0:
        gr_temp = p[~mask]
        LY3 = torch.sum(torch.log(1 - gr_temp + eps))  # add eps for safety
        
        entropy_loss = LY1 + LY2 + LY3
        
        return entropy_loss, theta, k, p, self.loc, rate

class MultiTrialRegressionPipeline():
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.model_name_str = self.model.name_or_path
        self.model_prefix = self.model.name_or_path.replace('/', '_')
        self.regression_model = kwargs['regression_model']
        self.test_set_size = kwargs.get('test_set_size', 0.7)
        allen_cache_path = os.environ.get('HGMS_ALLEN_CACHE_PATH')
        if allen_cache_path is None:
            pass
            # raise AllenCachePathNotSpecifiedError()
        transformer_embedding_cache_path = os.environ.get(
            'HGMS_TRANSF_EMBEDDING_PATH')
        if transformer_embedding_cache_path is None:
            pass
            # raise TransformerEmbeddingCachePathNotSpecifiedError()
        self.regr_analysis_results_cache = os.environ.get(
            'HGMS_REGR_ANAL_PATH')
        if self.regr_analysis_results_cache is None:
            pass
            # raise RegressionOutputCachePathNotSpecifiedError()
        self.boc = kwargs['boc']
        self.eid_dict = make_container_dict(self.boc)
        self.stimulus_session_dict = {
            'three_session_A': ['natural_movie_one', 'natural_movie_three'],
            'three_session_B': ['natural_movie_one'],
            'three_session_C': ['natural_movie_one', 'natural_movie_two'],
            'three_session_C2': ['natural_movie_one', 'natural_movie_two']
        }
        embedding_file_path = os.path.join(
            transformer_embedding_cache_path, f"{self.model_prefix}_embeddings.pkl")
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name_str)
        if not os.path.exists(embedding_file_path):
            self.embeddings = MakeEmbeddings(
                self.processor, self.model).execute()
        else:
            with open(embedding_file_path, 'rb') as f:
                self.embeddings = pickle.load(f)