from transformers import GPT2Tokenizer
import torch.nn as nn
from src.enc.attention import AttentionSingleSentence, TransformerEncoder
import pandas as pd

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
spec_token_map = {"cls_token": "<|CLS|>", "pad_token": "<|PAD|>"}
tokenizer.add_special_tokens(spec_token_map)

xls = pd.ExcelFile("data/retail/online_retail_II.xlsx")

first_obs = pd.read_excel(xls, "Year 2009-2010")
second_obs = pd.read_excel(xls, "Year 2010-2011")

# groups_1 = first_obs.groupby("Invoice")
# groups_2 = second_obs.groupby("Invoice")

first_obs_label = f"<|CLS|> {first_obs['Description'][0]}"
second_obs_label = f"<|CLS|> {second_obs['Description'][0]}"

print(f"First obs label: {first_obs_label}\nSecond obs label: {second_obs_label}")

text_embed_dim = 256

TEXT_EMBED_DIM = 256
Q_K_V_EMBED_DIM = 512

att = AttentionSingleSentence(
    text_embed_dim=TEXT_EMBED_DIM,
    q_k_v_embed_dim=Q_K_V_EMBED_DIM
)

tokenized_first = tokenizer(first_obs_label, padding="max_length", max_length=50, return_tensors="pt")
tokenized_second = tokenizer(second_obs_label, padding="max_length", max_length=50, return_tensors="pt")


## vocab size+2 bc additional special tokens
embedder = nn.Embedding(tokenizer.vocab_size+len(list(spec_token_map.keys())), text_embed_dim)

embedded_labels = {
    "embedded_first": embedder(tokenized_first["input_ids"]),
    "embedded_second": embedder(tokenized_second["input_ids"])
}

print(embedded_labels["embedded_first"].shape)
print(embedded_labels["embedded_second"].shape)

# minibatch = torch.cat([embedded_labels["embedded_first"], embedded_labels["embedded_second"]], dim=1)

encoder = TransformerEncoder(
    embed_dim=256,
    num_heads=4,
    depth=3,
    mlp_dim=256
)

encoded_label = encoder(embedded_labels["embedded_first"], tokenized_first["attention_mask"])

print(" ---- encoding finished successfully ---- ")

print(f"Final label embedding: {encoded_label}\n Its shape: {encoded_label.shape}")

h1_emb = encoder(encoded_label, tokenized_first["attention_mask"])

print(f"Hierarchical embedding level 1: {h1_emb}\n Its shape: {h1_emb.shape}")
