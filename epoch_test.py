from transformers import GPT2Tokenizer
import torch.nn as nn
from src.enc.attention import TransformerEncoder
from src.enc.predictor import NextItemEmbeddingPredictor
import torch
import copy
from torch.nn import MSELoss


tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
spec_token_map = {"cls_token": "<|CLS|>", "pad_token": "<|PAD|>"}
tokenizer.add_special_tokens(spec_token_map)

TEST_LABEL_1 = "<|CLS|> THIS IS THE FIRST LABEL"
TEST_LABEL_2 = "<|CLS|> THIS IS THE SECOND LABEL"
TEST_LABEL_3 = "<|CLS|> THIS IS THE THIRD LABEL"
TEST_LABEL_4 = "<|CLS|> THIS IS THE FOURTH LABEL"

label_1_tokens = tokenizer(TEST_LABEL_1, return_tensors="pt", padding="max_length", max_length=50)
label_2_tokens = tokenizer(TEST_LABEL_2, return_tensors="pt", padding="max_length", max_length=50)
label_3_tokens = tokenizer(TEST_LABEL_3, return_tensors="pt", padding="max_length", max_length=50)
label_4_tokens = tokenizer(TEST_LABEL_4, return_tensors="pt", padding="max_length", max_length=50)

embedder = nn.Embedding(tokenizer.vocab_size+2, 256)
embedded_list = []
mask_list = []

for token in [label_1_tokens, label_2_tokens, label_3_tokens, label_4_tokens]:
    embedded_list.append(embedder(token["input_ids"]))
    mask_list.append(token["attention_mask"])

context_embed_minibatch = torch.cat(embedded_list[:-1], dim=0)
context_mask_minibatch = torch.cat(mask_list[:-1], dim=0)

target_embed_minibatch = torch.cat(embedded_list, dim=0)
target_mask_minibatch = torch.cat(mask_list, dim=0)

context_mask_minibatch = context_mask_minibatch[:, None, None, :]
target_mask_minibatch = target_mask_minibatch[:, None, None, :]

teacher_encoder = TransformerEncoder(
    embed_dim=256,
    num_heads=4,
    depth=4,
    seq_len=50,
    mlp_dim=512
)

student_encoder = copy.deepcopy(teacher_encoder)

student_result = student_encoder(context_embed_minibatch, context_mask_minibatch)

teacher_result = teacher_encoder(target_embed_minibatch, target_mask_minibatch)

## [3, 256] -> 3 cls tokens (N-1)
student_cls = student_result[:, 0, :].unsqueeze(0)

# [4, 256] -> 4 cls tokens containing target (N)
teacher_cls = teacher_result[:, 0, :]

# [256] -> only target embedding (N.)
target_cls = teacher_cls[-1, :].unsqueeze(0)

predictor = NextItemEmbeddingPredictor(
    embed_dim=256,
    pred_dim=256,
    max_length=50,
    num_heads=4,
    depth=4,
    mlp_dim=512
)

predicted_cls = predictor(student_cls)

loss_fn = MSELoss()

train_loss = loss_fn(predicted_cls, target_cls).item()

print(f"1 epoch train loss is: {train_loss}")
