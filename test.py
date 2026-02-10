from transformers import GPT2Tokenizer
import torch.nn as nn
from src.enc.attention import AttentionSingleSentence, TransformerEncoder
import torch
import pandas as pd

xls = pd.ExcelFile("data/retail/online_retail_II.xlsx")

first_obs = pd.read_excel(xls, "Year 2009-2010")
second_obs = pd.read_excel(xls, "Year 2010-2011")

first_obs_label = first_obs["Description"][0]
second_obs_label = second_obs["Description"][0]

print(f"First obs label: {first_obs_label}\nSecond obs label: {second_obs_label}")

text_embed_dim = 256

TEXT_EMBED_DIM = 256
Q_K_V_EMBED_DIM = 512

att = AttentionSingleSentence(
    text_embed_dim=TEXT_EMBED_DIM,
    q_k_v_embed_dim=Q_K_V_EMBED_DIM
)

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

special_tokens_dict = {"cls_token": "<CLS>"}

tokenized = {
    "tokenized_first": torch.tensor(tokenizer.encode(first_obs_label)),
    "tokenized_second": torch.tensor(tokenizer.encode(second_obs_label))
}

print(tokenized)

embedder = nn.Embedding(tokenizer.vocab_size, text_embed_dim)

embedded_labels = {
    "embedded_first": embedder(tokenized["tokenized_first"].unsqueeze(0)),
    "embedded_second": embedder(tokenized["tokenized_second"].unsqueeze(0))
}

print(embedded_labels["embedded_first"].shape)
print(embedded_labels["embedded_second"].shape)

minibatch = torch.cat([embedded_labels["embedded_first"], embedded_labels["embedded_second"]], dim=1)

## padding and masking needed bc of different sequence sizes ...
## after concating by batches will be possible
# print(minibatch.shape)

encoder = TransformerEncoder(
    embed_dim=256,
    num_heads=4,
    depth=3,
    mlp_dim=256
)

encoded_label = encoder(embedded_labels["embedded_first"])

print(" ---- encoding finished successfully ---- ")

print(f"Final label embedding: {encoded_label}\n Its shape: {encoded_label.shape}")

# ctx = att(embedded_sentence)
