#NEW XAI MODEL ARCHITECTURE

import os
import yaml
from num2words import num2words
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Path to your YAML config
config_path = os.path.join(os.path.dirname(__file__), "..", "config", "salient_bert.yaml")

# Load YAML
with open(os.path.abspath(config_path)) as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None  # To store attention weights if needed

    def forward(self, q, k, v, mask, return_attention_weights = False):
        # Apply linear transformations to the inputs
        query = self.w_q(q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # Reshape to (batch, seq_len, h, d_k) and then transpose to (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
    
        mask = mask.to(torch.bool)  
        attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.shape[0], -1, self.h * self.d_k)
        output = self.w_o(attn_output)
        if return_attention_weights == True:
            scores = torch.matmul(query, key.transpose(-2, -1)) / self.d_k**0.5
            self.attn_weights = torch.softmax(scores, dim=-1)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(Block, self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask):
        self_attn_output = self.self_attn(x ,x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


'''Main Modification will be here'''

class XAIBERT(nn.Module):
    def __init__(self, n_class, vocab_size, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super(XAIBERT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=64)
        self.layers = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        '''Main Modification will be here'''
        self.fc_out = nn.Linear(d_model, n_class)
        
    def forward(self, tgt, tgt_mask, return_attention_weights = False):
        x = self.embedding(tgt)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, tgt_mask)
        '''Main Modification will be here'''        
        cls_token = x[:, 0, :]
        return self.fc_out(cls_token)

Xper_model = XAIBERT(**config).to(device)

def number_to_words(num):
    return num2words(num, lang='en_IN')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) 

total_params = count_parameters(Xper_model)
print(f"Total model parameters: {total_params} -- {number_to_words(total_params)}")