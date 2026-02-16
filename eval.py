from train_hjepa import (
    trained_predictor,
    trained_student,
    trained_embedder,
    trained_text_enc,
    tokenizer,
    first_obs,
    device
)
from src.eval.metrics import create_label_embedding_data
import torch.nn.functional as F

import torch

trained_predictor.load_state_dict(torch.load("results/trained_models/predictor.pth", weights_only=True, map_location=device))
trained_student.load_state_dict(torch.load("results/trained_models/student_enc.pth", weights_only=True, map_location=device))
trained_embedder.load_state_dict(torch.load("results/trained_models/embedder.pth", weights_only=True, map_location=device))
trained_text_enc.load_state_dict(torch.load("results/trained_models/text_enc.pth", weights_only=True, map_location=device))

trained_embedder.eval()
trained_text_enc.eval()
trained_predictor.eval()
trained_student.eval()

## taken from invoice id 538156
sequence = [
    "<|CLS|> GREEN CHRISTMAS TREE CARD HOLDER",
    "<|CLS|> HOT WATER BOTTLE TEA AND SYMPATHY",
    "<|CLS|> NOEL WOODEN BLOCK LETTERS ",
    "<|CLS|> CHRISTMAS CRAFT TREE TOP ANGEL",
    "<|CLS|> CHRISTMAS CRAFT LITTLE FRIENDS",
    "<|CLS|> 3 HOOK PHOTO SHELF ANTIQUE WHITE",
    "<|CLS|> METAL 4 HOOK HANGER FRENCH CHATEAU",
    "<|CLS|> BLUE OWL SOFT TOY",
    "<|CLS|> RED STAR CARD HOLDER"
]

# "<|CLS|> FOUR HOOK  WHITE LOVEBIRDS", 
# "<|CLS|> FELTCRAFT DOLL MARIA", 
# "<|CLS|> FELTCRAFT DOLL ROSIE", 
# "<|CLS|> FELTCRAFT DOLL MOLLY", 
# "<|CLS|> POMPOM CURTAIN"

## returned top k example (in order):
## 60 GOLD AND SILVER FAIRY CAKE CASES
## FELTCRAFT CUSHION BUTTERFLY
## SILVER PURSE GOLD PINK BUTTERFLY
## 60 TEATIME FAIRY CAKE CASES
## FANCY FONT BIRTHDAY CARD,

tokenized = tokenizer(sequence, return_tensors="pt", max_length=50, padding="max_length")

input_ids = tokenized["input_ids"]
att_masks = tokenized["attention_mask"]
with torch.no_grad():
    label_embs = trained_embedder(input_ids)

    txt_encoded = trained_text_enc(label_embs, att_masks[:, None, None, :])

    ## shape must be [5, 256]
    cls_tokens = txt_encoded[:, 0, :].unsqueeze(0)

    cls_att = trained_student(cls_tokens)

    pred_cls = trained_predictor(cls_att)

## embs shape is [4821, 256] (all the embeddings for all the unique labels)
embs, labels = create_label_embedding_data(first_obs, trained_embedder, tokenizer, trained_text_enc, device)

sims = F.cosine_similarity(pred_cls, embs, dim=-1)

k = 5
vals, idxs = torch.topk(sims, k)

print(f"These are the top {k} labels:")

for idx in idxs:
    print(labels[idx])