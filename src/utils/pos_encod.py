## credit to: https://github.com/wzlxjtu/PositionalEncoding2D/

import torch
import math

def positional_encoding_1d(embed_dim, lenght):
    """
    :embed_dim: embedding dimension of the model
    :lenght: input sequence length (with padding)
    """

    pe = torch.zeros(lenght, embed_dim)

    ## numbers from 0 corresponding for position in the sequence
    position = torch.arange(0, lenght).unsqueeze(1)

    ## return the size of embed dim / 2
    ## as we only need the in sin / cos operations below
    ## where only embed_dim / 2 indexing is needed ([:, 0::2] and [:, 1::2])
    div_term = torch.exp(
        (torch.arange(0, embed_dim, 2, dtype=torch.float) * -(math.log(10000.0) / embed_dim))
    )

    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.unsqueeze(0)
