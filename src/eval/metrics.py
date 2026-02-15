import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

def create_label_embedding_data(
    df: pd.DataFrame,
    embedder,
    tokenizer,
    text_encoder,
    device,
    batch_size: int = 32,
    max_length: int = 50,
):

    text_encoder.eval()
    embedder.eval()

    labels = df["Description"].unique().tolist()
    texts = ["<|CLS|> " + str(label) for label in labels]

    cls_embeds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]

            tokens = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length
            ).to(device)

            input_ids = tokens["input_ids"]
            att_masks = tokens["attention_mask"]

            token_emb = embedder(input_ids)
            reps = text_encoder(token_emb, att_masks[:, None, None, :])
            cls = reps[:, 0, :]

            cls_embeds.append(cls)

    cls_embeddings = torch.cat(cls_embeds, dim=0)

    return cls_embeddings, labels
