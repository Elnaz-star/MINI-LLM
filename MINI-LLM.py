
import torch
import requests
import torch.nn as nn
# =========================
# DATA
# =========================

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

text = requests.get(url).text

with open("input.txt", "w", encoding="utf-8") as f:
    f.write(text)

# reload cleanly (important)
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()


# =========================
# TOKENIZER
# =========================

class Tokenizer:

    def __init__(self, text):

        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])

    def encode_tensor(self, text):
        return torch.tensor(self.encode(text), dtype=torch.long)


tokenizer= Tokenizer(text)



# =========================
# DATA TENSOR
# =========================

data = tokenizer.encode_tensor(text)


# =========================
# BATCH SAMPLING
# =========================

def get_batch(data, block_size, batch_size):

    ix = torch.randint(
        0,
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i+block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i+1:i+block_size+1]
        for i in ix
    ])

    return x, y


block_size = 8
batch_size = 4

x, y = get_batch(data, block_size, batch_size)

# print(x.shape)
# print(y.shape)
vocab_size = tokenizer.vocab_size
embedding_dim = 32

embedding_table = nn.Embedding(
    vocab_size,
    embedding_dim
)

x_emb = embedding_table(x)

# print(x_emb.shape)

# =========================
# Positional embeding
# =========================

class PositionalEncoding(nn.Module):

    def __init__(self, block_size, embed_dim):

        super().__init__()

        # position embedding table
        self.pos_emb = nn.Embedding(block_size, embed_dim)

    def forward(self, x):

        B, T, C = x.shape

        # positions: 0,1,2,...,T-1
        positions = torch.arange(T, device=x.device).unsqueeze(0)

        # shape (T, C)
        pos_embeddings = self.pos_emb(positions)

        # expand to batch (B, T, C)
        return x + pos_embeddings
    
pos_encoder = PositionalEncoding(block_size=8, embed_dim=32)

x_with_pos = pos_encoder(x_emb)

# print(x_with_pos.shape)

# =========================
# CAUSAL SELF ATTENTION
# =========================

class CausalSelfAttention(nn.Module):

    def __init__(self, embed_dim, block_size):

        super().__init__()

        self.embed_dim = embed_dim

        # QKV projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

        B, T, C = x.shape

        # =====================
        # Q K V
        # =====================

        Q = self.query(x)   # (B,T,C)
        K = self.key(x)     # (B,T,C)
        V = self.value(x)   # (B,T,C)

        # =====================
        # Attention scores
        # =====================

        scores = Q @ K.transpose(-2, -1)

        # shape: (B,T,T)

        scores = scores / (C ** 0.5)

        # =====================
        # Apply causal mask
        # =====================

        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float('-inf')
        )

        # =====================
        # Softmax
        # =====================

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # =====================
        # Attention output
        # =====================

        output = attention_weights @ V

        # shape: (B,T,C)

        return output
    

attention = CausalSelfAttention(
    embed_dim=32,
    block_size=8
)

attention_output = attention(x_with_pos)

# print(attention_output.shape)