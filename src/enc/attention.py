import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    