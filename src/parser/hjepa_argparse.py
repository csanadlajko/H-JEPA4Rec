from argparse import ArgumentParser

def parse_hjepa_args():
    parser = ArgumentParser()
    parser.add_argument("--lr", help="learning rate value for H-JEPA (default: 0.0001)", type=float, default=0.0001)
    parser.add_argument("--embed_dim", help="embedding dimension for label encoding (default: 256)", type=int, default=256)
    parser.add_argument("--depth", help="depth of the transformer encoder module (default: 6)", type=int, default=6)
    parser.add_argument("--num_heads", help="number of heads in the MHSA module, must be a divisor of embed_dim (default: 4)", type=int, default=4)
    parser.add_argument("--input_path", help="path of the input dataset (default: None)", type=str, default=None)
    parser.add_argument("--result_path", help="path of the result dictionary (default: /result)", type=str, default="/result")
    parser.add_argument("--debug", help="debug mode (logging, etc..) (default: n)", type=str, default="n")
    parser.add_argument("--tokenizer", help="name of the used tokenizer (default: openai-community/gpt2)", type=str, default="openai-community/gpt2")
    parser.add_argument("--max_length", help="max length of labels with extended padding (default: 50)", type=int, default=50)
    parser.add_argument("--ff_dim", help="embed dim of the feed forward network in the MHSA module (default: 512)", type=int, default=512)
    parser.add_argument("--epochs", help="number of epochs when training (default: 20)", type=int, default=20)
    parser.add_argument("--dropout", help="dropout percentage in the transformer encoder module (default: 0.1)", type=float, default=0.1)

    args = parser.parse_args()
    return args