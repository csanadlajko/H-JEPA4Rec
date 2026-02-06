import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):

    def __init__(self, text_embed_dim, q_k_v_embed_dim):
        super(Attention, self).__init__()
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