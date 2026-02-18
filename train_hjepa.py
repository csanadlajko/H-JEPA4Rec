from transformers import GPT2Tokenizer
import torch.nn as nn
from src.enc.attention import TransformerEncoder
from src.enc.predictor import NextItemEmbeddingPredictor
import torch
import copy
import pandas as pd
from src.utils.ema import _ema_update
from src.parser.hjepa_argparse import parse_hjepa_args
from datetime import datetime
import torch.nn.functional as F
from tqdm import tqdm

run_id: str = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

args = parse_hjepa_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Starting temporary training on device: {device}")

tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
spec_token_map = {"cls_token": "<|CLS|>", "pad_token": "<|PAD|>"}
additional_tokens = {'additional_special_tokens': ['<|SESSION_CLS|>']}
tokenizer.add_special_tokens(spec_token_map)
tokenizer.add_special_tokens(additional_tokens)

xls = pd.ExcelFile("data/retail/online_retail_II.xlsx")

first_obs = pd.read_excel(xls, "Year 2009-2010")

first_obs = first_obs.drop(columns=["Quantity", "Price", "StockCode", "InvoiceDate", "Customer ID", "Country"])
first_obs = first_obs.dropna()

predictor = NextItemEmbeddingPredictor(
    embed_dim=args.embed_dim,
    pred_dim=args.embed_dim,
    max_length=args.max_length,
    num_heads=args.num_heads,
    depth=args.depth,
    mlp_dim=args.ff_dim,
    dropout=0.25
).to(device)

text_enc = TransformerEncoder(
    embed_dim=args.embed_dim,
    num_heads=args.num_heads,
    depth=args.depth,
    seq_len=args.max_length,
    mlp_dim=args.ff_dim,
    dropout=args.dropout
).to(device)

teacher_encoder = TransformerEncoder(
    embed_dim=args.embed_dim,
    num_heads=args.num_heads,
    depth=args.depth,
    seq_len=None,
    mlp_dim=args.ff_dim,
    dropout=0.25
).to(device)

student_encoder = copy.deepcopy(teacher_encoder)

trained_student = copy.deepcopy(student_encoder)
trained_predictor = copy.deepcopy(predictor)
trained_text_enc = copy.deepcopy(text_enc)

optim_student = torch.optim.AdamW(params=student_encoder.parameters(), lr=args.lr, weight_decay=0.00025)
loss_fn = nn.MSELoss()

optim_pred = torch.optim.AdamW(params=predictor.parameters(), lr=args.lr, weight_decay=0.00025)

embedder = nn.Embedding(len(tokenizer), args.embed_dim).to(device)

trained_embedder = copy.deepcopy(embedder)

## TEMPORARY TRAINING LOOP FOR TESTING MODEL BEHAVIOR
if __name__ == "__main__":
    predictor.train()
    student_encoder.train()
    teacher_encoder.eval()
    text_enc.train()
    embedder.train()

    if args.debug == "y":
        print(f"Architecture has: {sum(p.numel() for p in student_encoder.parameters() if p.requires_grad) + \
                               sum(p.numel() for p in teacher_encoder.parameters() if p.requires_grad) + \
                               sum(p.numel() for p in predictor.parameters() if p.requires_grad) + \
                               sum(p.numel() for p in text_enc.parameters() if p.requires_grad) + \
                               sum(p.numel() for p in embedder.parameters() if p.requires_grad)} trainable parameters.")

    print(f"Training for {args.epochs} epochs...")

    for j in range(args.epochs):

        print(f"Starting epoch: {j+1}")

        total_loss = 0.0
        i: int = 2
        session_start: int = 0
        ctr: int = 0

        bar = tqdm(total=int(len(first_obs)* 0.8))

        while i < int(len(first_obs) * 0.8):
            curr_invoice = first_obs["Invoice"].iloc[i]
            label_batch = first_obs["Description"].iloc[session_start:i].to_list()
            label_batch = ["<|CLS|> " + str(desc) for desc in label_batch]
            label_batch.insert(0, "<|SESSION_CLS|>")

            tokens = tokenizer(label_batch, return_tensors="pt", padding="max_length", max_length=50)
            input_ids = tokens["input_ids"].to(device)
            att_masks = tokens["attention_mask"].to(device)
            label_embed = embedder(input_ids)

            ## text representations [B, T, D]
            text_repr = text_enc(label_embed, att_masks[:, None, None, :])

            ## all cls token in the sequence [B, D]
            ## unsqueeze to get [1, B, D]
            cls_all = text_repr[:, 0, :].unsqueeze(0)

            ## every item in the session except the current
            ## this makes the context as [0, 1, ..., N-1] items are in the sequence
            ## input and return shapes are [1, N-1, D]
            enc_ctx = student_encoder(cls_all[:, :-1, :])

            ## input: [1, N-1, D]
            ## returns [1, D] (session cls embedding)
            predicted = predictor(enc_ctx)

            ## every item in the session including the last and the session cls embedding
            with torch.no_grad():
                enc_target = teacher_encoder(cls_all)
                ## acquire ground truth session cls embedding on index 0
                target_cls = enc_target[:, 0, :]
                target_cls = F.layer_norm(target_cls, (target_cls.size(-1),))

            ## the goal is to predict such next items, that fit into the current session's context
            ## if the predicted last item doesn't fit in -> larger MSE
            ## if the predicted last item fits in -> similar session cls embeddings -> smaller MSE
            loss = loss_fn(predicted, target_cls)

            total_loss += loss.item()

            optim_student.zero_grad()
            optim_pred.zero_grad()
            loss.backward()
            optim_student.step()
            optim_pred.step()

            ctr+=1

            if (ctr + 1) % 5000 == 0 and args.debug == "y":
                print(f"loss at iter {ctr + 1} is {loss.item():.4f}")

            if i+2 <= len(first_obs) and first_obs["Invoice"].iloc[i+2] != curr_invoice:
                session_start = i+2
                i += 4
                bar.update(4)
                _ema_update(student_encoder, teacher_encoder)
            else:
                i+=1
                bar.update(1)
        
        print(f"Epoch {j+1} ended, average loss is: {total_loss / ctr:.3f}")

    if args.debug == "y":
        torch.save(student_encoder.state_dict(), f"{args.result_folder}/student_enc_{run_id}.pth")
        torch.save(teacher_encoder.state_dict(), f"{args.result_folder}/teacher_enc_{run_id}.pth")
        torch.save(predictor.state_dict(), f"{args.result_folder}/predictor_{run_id}.pth")
        torch.save(text_enc.state_dict(), f"{args.result_folder}/txt_encoder_{run_id}.pth")
        torch.save(embedder.state_dict(), f"{args.result_folder}/embedder_{run_id}.pth")
        print("Models saved successfully!")

    bar.close()
    print("Training ended successfully...")