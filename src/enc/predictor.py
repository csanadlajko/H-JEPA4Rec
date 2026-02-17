from .attention import TransformerEncoder
import torch
import torch.nn as nn

class NextItemEmbeddingPredictor(nn.Module):
    
    def __init__(self, embed_dim, max_length, num_heads, depth, mlp_dim, pred_dim=None, dropout=0.1):
        super(NextItemEmbeddingPredictor, self).__init__()
        if pred_dim is None:
            pred_dim = embed_dim

        self.predictor_embed = nn.Linear(embed_dim, pred_dim)

        ## learnable next item's CLS token
        self.pred_next_item = nn.Parameter(torch.zeros(1, 1, pred_dim), requires_grad=True)
        
        ## learnable positional encoding for the input item sequence
        self.item_pos_encoding = nn.Parameter(torch.zeros(1, num_heads, 1, embed_dim // num_heads), requires_grad=True)

        self.enc = TransformerEncoder(
            embed_dim=pred_dim,
            num_heads=num_heads,
            dropout=dropout,
            seq_len=max_length,
            mlp_dim=mlp_dim,
            depth=depth
        )

        self.pred_norm = nn.LayerNorm(pred_dim)
        self.predictor_proj = nn.Linear(pred_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x ([1, N-1, D] Tensor): encoded item sequence in a session (context).

        Returns: 
            [1, D] Tensor: corresponding for the updated session cls token containing information about the predicted Nth item.
        """
        ## x is [1, N-1, D]
        ## <|SESSION_CLS|> token embedding is on index 0 [1, 0, D]
        
        N = x.shape[1] + 1

        x = self.predictor_embed(x)

        sequence_encoding = self.item_pos_encoding.repeat(1, 1, N, 1)

        predicted_full_sequence = torch.cat([x, self.pred_next_item], dim=1)

        x = self.enc(predicted_full_sequence, pred_pos_encoding=sequence_encoding)

        x = self.pred_norm(x)

        x = self.predictor_proj(x)

        ## return the predicted sequence's cls token
        ## measures how the learnable Nth token influences the session's representation
        predicted_cls = x[:, 0, :]

        return predicted_cls
