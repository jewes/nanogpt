import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """
    the single head self attention module
    """

    def __init__(self, n_embed, head_size, block_size, dropout=0.0):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is the input
        q: torch.Tensor = self.query(x)
        k: torch.Tensor = self.key(x)
        v: torch.Tensor = self.value(x)

        # calculate the weighted matrix
        # where C is the embedding dimension
        B, T, C = x.shape
        wei = q @ k.transpose(-2, -1)  # BTT
        wei = wei * self.head_size ** -0.5

        # masked attention so that future token does not talk to previous ones
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v  # BTT * BTH => BTH
        return out


class MutilHead(nn.Module):

    def __init__(self, n_embed, n_head, block_size, dropout=0.0):
        super().__init__()
        head_size = n_embed // n_head
        assert n_embed == n_head * head_size

        # each head handles 1/head_size of the embeddings
        self.heads = nn.ModuleList([
            Head(n_embed=n_embed, head_size=head_size, block_size=block_size, dropout=dropout) for _ in range(n_head)
        ])

        # project the attention result back to the mainline
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):

    def __init__(self, n_dim, dropout=0.0):
        super().__init__()
        # from the original attention is all you need paper, expand inner layer by 4
        self.net = nn.Sequential(
            nn.Linear(n_dim, 4 * n_dim),
            nn.ReLU(),
            # proj
            nn.Linear(4 * n_dim, n_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Block(nn.Module):

    def __init__(self, n_embed, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.head = MutilHead(n_embed=n_embed, n_head=n_head, block_size=block_size, dropout=dropout)

        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedForward(n_dim=n_embed, dropout=dropout)

    def forward(self, x):
        x = x + self.head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

