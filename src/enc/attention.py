import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..utils.pos_encod import positional_encoding_1d

class AttentionSingleSentence(nn.Module):

    def __init__(self, text_embed_dim, q_k_v_embed_dim):
        super(AttentionSingleSentence, self).__init__()
        self.text_embed_dim = text_embed_dim
        self.q_k_v_embed_dim = q_k_v_embed_dim
        ## create learnable q,k,v matrix weights
        ## shape [text_embed_dim, q_k_v_embed_dim] needed for dot product
        self.W_q = nn.Parameter(torch.randn(self.text_embed_dim, self.q_k_v_embed_dim))
        self.W_k = nn.Parameter(torch.randn(self.text_embed_dim, self.q_k_v_embed_dim))
        self.W_v = nn.Parameter(torch.randn(self.text_embed_dim, self.q_k_v_embed_dim))

    def forward(self, x):
        ## suppose x is an embedded (batch of) sentence in the text_embed_dim dimension
        ## shape [len(tokenized_sentence), text_embed_dim]

        ## create q,k,v matrices based on the text embedding
        ## [len(tokenized_sentence), text_embed_dim] @ [text_embed_dim, q_k_v_embed_dim]
        ## [len(tokenized_sentence), q_k_v_embed_dim]
        query = x @ self.W_q
        key = x @ self.W_k
        value = x @ self.W_v

        ## calculate attention scores and softmax
        ## [len(tokenized_sentence), q_k_v_embed_dim] @ [q_k_v_embed_dim, len(tokenized_sentence)]
        ## -> [len(tokenized_sentence), len(tokenized_sentence)]
        attention_scores = query @ key.T
        attention_scores = attention_scores / math.sqrt(self.q_k_v_embed_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        ## finally ctx vector
        ## [len(tokenized_sentence), len(tokenized_sentence)] @ [len(tokenized_sentence), q_k_v_embed_dim]
        ## -> [len(tokenized_sentence), q_k_v_embed_dim]
        context = attention_weights @ value
        return context
    
class AttentionBlock(nn.Module):
    
    def __init__(self, text_embed_dim, q_k_v_embed_dim):
        super(AttentionBlock, self).__init__()
        self.text_embed_dim = text_embed_dim
        self.q_k_v_embed_dim = q_k_v_embed_dim

        self.W_q = nn.Linear(text_embed_dim, q_k_v_embed_dim)
        self.W_k = nn.Linear(text_embed_dim, q_k_v_embed_dim)
        self.W_v = nn.Linear(text_embed_dim, q_k_v_embed_dim)

    def forward(self, x):
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        ## as both q and k matrices are the same shape we need to transpose K to evaluate the dot product
        attention_scores = query @ torch.transpose(key, dim0=-1, dim1=-2)
        attention_scores = attention_scores / self.q_k_v_embed_dim**0.5
        attention_scores = F.softmax(attention_scores, dim=-1)

        context = attention_scores @ value
        return context

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_embed_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)


    def forward(self, x, attention_mask):

        B, T, E = x.shape

        qkv = self.qkv(x)

        ## reshape qkv (from [B, T, 3*EMBED_DIM]) to the following:
        ## B: batch size
        ## T: sequence length
        ## N_M: q, k, v matrices -> constant 3
        ## N_H: number of parallel heads
        ## E_H: embedding dimension per head
        ## N_M * N_H * E_H == 3 * EMBED DIM
        assert 3 * self.num_heads * self.head_embed_dim == 3*self.embed_dim
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_embed_dim)

        ## permute so:
        ## query gets index 0
        ## key gets index 1
        ## value gets index 2
        ## [3, B, N_H, T, E_H]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        att_scores = q @ k.transpose(-2, -1)
        att_scores = att_scores / (self.head_embed_dim ** 0.5)

        att_scores = att_scores.masked_fill(attention_mask == 0, float("-inf"))

        attn = F.softmax(att_scores, dim=-1)
        attn = self.dropout(attn)

        context = attn @ v
        context = context.transpose(1, 2)
        context = context.contiguous().view(B, T, E)

        return self.out_proj(context)


class MLP(nn.Module):

    def __init__(self, embed_dim, mlp_embed_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(embed_dim, mlp_embed_dim)
        self.fc2 = nn.Linear(mlp_embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.1, mlp_embed_dim=512):
        super(TransformerEncoderLayer, self).__init__()

        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm1= nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_embed_dim)

    def forward(self, x, attention_mask):
        
        ## mhsa
        attn = self.attn(x, attention_mask)

        ## residual
        x = x + self.dropout(attn)
        x = self.norm1(x)

        ## feedforward 
        mlp_out = self.mlp(x)

        ## residual
        x = x + self.dropout(mlp_out)
        x = self.norm2(x)

        return x
    
class TransformerEncoder(nn.Module):

    def __init__(self, embed_dim, num_heads, depth, mlp_dim, seq_len=50, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, dropout, mlp_dim)
            for _ in range(depth)
        ])
        self.dropout = nn.Dropout(dropout)
        self.pos_enc = positional_encoding_1d(embed_dim, seq_len)

    def forward(self, x, attention_mask):
        x = x + self.pos_enc
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask)
        
        return x