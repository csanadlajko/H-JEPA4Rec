from transformers import GPT2Tokenizer
import torch.nn as nn
from src.enc.attention import TransformerEncoder
from src.enc.predictor import NextItemEmbeddingPredictor
import torch
import copy
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Starting temporary training on device: {device}")

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
spec_token_map = {"cls_token": "<|CLS|>", "pad_token": "<|PAD|>"}
tokenizer.add_special_tokens(spec_token_map)

xls = pd.ExcelFile("data/retail/online_retail_II.xlsx")

first_obs = pd.read_excel(xls, "Year 2009-2010")

first_obs = first_obs.drop(columns=["Quantity", "Price", "StockCode", "InvoiceDate", "Customer ID", "Country"])

predictor = NextItemEmbeddingPredictor(
    embed_dim=256,
    pred_dim=256,
    max_length=50,
    num_heads=4,
    depth=4,
    mlp_dim=512
)

teacher_encoder = TransformerEncoder(
    embed_dim=256,
    num_heads=4,
    depth=4,
    seq_len=50,
    mlp_dim=512
)

student_encoder = copy.deepcopy(teacher_encoder)

student_encoder = student_encoder.to(device)
teacher_encoder = teacher_encoder.to(device)
predictor = predictor.to(device)

optim_student = torch.optim.AdamW(params=student_encoder.parameters(), lr=0.00003)
loss_fn = nn.MSELoss()

optim_pred = torch.optim.AdamW(params=predictor.parameters(), lr=0.00003)

embedder = nn.Embedding(tokenizer.vocab_size+2, 256).to(device)

i: int = 2
session_start: int = 0
ctr: int = 0

## TEMPORARY TRAINING LOOP FOR TESTING MODEL BEHAVIOR
if __name__ == "__main__":

    while i < len(first_obs):
        curr_invoice = first_obs["Invoice"][i]
        label_batch = first_obs[session_start:i]["Description"]

        tokens = tokenizer(first_obs[session_start:i]["Description"].to_list(), return_tensors="pt", padding="max_length", max_length=50)
        label_embed = embedder(tokens["input_ids"])
        label_embed = label_embed.to(device)
        att_masks = tokens["attention_mask"]
        att_masks = att_masks.to(device)

        ## every item in the session except the current
        enc_ctx = student_encoder(label_embed[:-1, :, :], att_masks[:-1, None, None, :])

        ## drop token length -> not relevant when predicting cls token
        ## enc_cls: a sequence of cls tokens [0, 1, ...., N-1]
        enc_cls = enc_ctx[:, 0, :].unsqueeze(0)

        predicted = predictor(enc_cls)

        ## every item in the session including the last
        enc_target = teacher_encoder(label_embed, att_masks[:, None, None, :])

        student_cls = enc_ctx[-1:, 0, :]
        target_cls = enc_target[-1, 0, :].unsqueeze(0)

        loss = loss_fn(student_cls, target_cls)

        optim_student.zero_grad()
        optim_pred.zero_grad()
        loss.backward()
        optim_student.step()
        optim_pred.step()

        if (ctr + 1) % 5000 == 0:
            print(f"loss at iter {ctr + 1} is {loss.item():.4f}")

        if i+2 <= len(first_obs) and first_obs["Invoice"][i+2] != curr_invoice:
            session_start = i+2
            i += 4
        else: 
            i+=1
            ctr+=1

