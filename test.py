from transformers import GPT2Tokenizer
import torch.nn as nn
from src.enc.attention import AttentionSingleSentence
import torch

TEXT_EMBED_DIM = 256
Q_K_V_EMBED_DIM = 512

att = AttentionSingleSentence(
    text_embed_dim=TEXT_EMBED_DIM,
    q_k_v_embed_dim=Q_K_V_EMBED_DIM
)

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

special_tokens_dict = {"cls_token": "<CLS>"}

test = "tokenize this new sentence with more words"

tokenized = torch.tensor(tokenizer.encode(test))

embed = nn.Embedding(tokenizer.vocab_size, 256)

embedded_sentence = embed(tokenized).detach()

ctx = att(embedded_sentence)
