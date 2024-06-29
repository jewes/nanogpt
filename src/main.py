import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# step1: create tokenizer
with open("input.txt") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for i, c in enumerate(chars)}
encode = lambda s: [ctoi[c] for _, c in enumerate(s)]
decode = lambda l: "".join([itoc[i] for _, i in enumerate(l)])

# step2: create train, val split
data = torch.tensor(encode(text), dtype=torch.long)
split = int(len(data) * 0.9)
train_data, val_data = data[:split], data[split:]

# step3 create batch
batch_size = 4
block_size = 8


def get_batch(split='train'):
    data = train_data if split == 'train' else val_data
    # randomly choose a number
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


# step 4 create the simple model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # embedding table
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)

    def forward(self, idx, targets: torch.Tensor = None):
        # idx and targets shape are both B, T = batch_size * block_size(context window)
        logits: torch.Tensor = self.token_embedding_table(
            idx)  # get embedding: B, T, C(embedding_dim, vacab_size for now)
        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_token):
        # idx shape B, T, the input
        for _ in range(max_new_token):
            # run inference
            logits, loss = self(idx)  # loss is ignored here
            # logits shape: B, T, C, get the last one of the 2nd dimension,
            logits = logits[:, -1, :]  # B, C
            probs = F.softmax(logits, dim=-1)  # run softmax on the last dimension
            # probs shape: B, 1, C(normalized)
            idx_next = torch.multinomial(probs, num_samples=1)  # B, 1
            idx = torch.cat((idx, idx_next), dim=1)  # idx.shape: B, T+1

        return idx


# n_embedding = 32
# head_size = 8
# step 5 create single head attention model
# key points
# QKV linear layer
# attention layer, position embedding, wording embedding, mask attention
class Head(nn.Module):
    def __init__(self, n_embedding=32, head_size=8, dropout=0.0):
        super().__init__()
        self.head_size = head_size

        # declare QKV
        self.query = nn.Linear(n_embedding, head_size, bias=False)
        self.key = nn.Linear(n_embedding, head_size, bias=False)
        self.value = nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: input from the embedding layer, shape: B, T, C
        :return:
        """
        B, T, C = x.shape

        # B, T, head_size
        q: torch.Tensor = self.query(x)
        k: torch.Tensor = self.key(x)

        # k becomes (B, head_size, T) after transpose
        wei: torch.Tensor = q @ k.transpose(-2, -1)
        # wei.shape=(B, T, T)

        wei = wei * self.head_size ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # v.shape = B,T,head_size
        v = self.value(x)

        # out.shape = B,T, head_size
        out = wei @ v
        return out


class MultiHead(nn.Module):
    """
    create multiple heads, each head process part of the embedding, then concat them
    """

    def __init__(self, num_heads, head_size, n_embedding):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embedding=n_embedding, head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embedding, n_embedding)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class SingleHeadLanguageModel(nn.Module):

    def __init__(self, n_embed, block_size):
        super().__init__()
        self.block_size = block_size
        # input shape: B, T, where T is the block size,
        # for each T, we use its value to get its embedding
        # for each T, we use its index to get its position embedding
        # so, both table have same output shape of embedding dimension
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embed)
        self.pos_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = Head(n_embedding=n_embed, head_size=n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets: torch.Tensor = None):
        # idx and targets shape are both B, T = batch_size * block_size(context window)
        # get embedding: B, T, C(embedding_dim, vacab_size for now)
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(0, T))
        x = tok_emb + pos_emb
        x = self.sa_head(x)
        logits: torch.Tensor = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_token):
        # idx shape B, T, the input
        for _ in range(max_new_token):
            # run inference
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)  # loss is ignored here
            # logits shape: B, T, C, get the last one of the 2nd dimension,
            logits = logits[:, -1, :]  # B, C
            probs = F.softmax(logits, dim=-1)  # run softmax on the last dimension
            # probs shape: B, 1, C(normalized)
            idx_next = torch.multinomial(probs, num_samples=1)  # B, 1
            idx = torch.cat((idx, idx_next), dim=1)  # idx.shape: B, T+1

        return idx


class FeedForward(nn.Module):

    def __init__(self, n_embed):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(n_embed, 4 * n_embed),
                                   nn.ReLU(),
                                   nn.Linear(4 * n_embed, n_embed),
                                   )

    def forward(self, x):
        return self.layer(x)


class Block(nn.Module):

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa_head = MultiHead(num_heads=n_head, n_embedding=n_embed, head_size=head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MultipleHeadLanguageModel(nn.Module):

    def __init__(self, n_embed, block_size):
        super().__init__()
        self.block_size = block_size
        # input shape: B, T, where T is the block size,
        # for each T, we use its value to get its embedding
        # for each T, we use its index to get its position embedding
        # so, both table have same output shape of embedding dimension
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embed)
        self.pos_embedding_table = nn.Embedding(block_size, n_embed)

        # 4 head, 8dimensions
        self.sa_head = MultiHead(num_heads=4, n_embedding=n_embed, head_size=n_embed // 4)
        self.ffwd = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets: torch.Tensor = None):
        # idx and targets shape are both B, T = batch_size * block_size(context window)
        # get embedding: B, T, C(embedding_dim, vacab_size for now)
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(0, T))
        x = tok_emb + pos_emb
        x = self.sa_head(x)
        x = self.ffwd(x)
        logits: torch.Tensor = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_token):
        # idx shape B, T, the input
        for _ in range(max_new_token):
            # run inference
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)  # loss is ignored here
            # logits shape: B, T, C, get the last one of the 2nd dimension,
            logits = logits[:, -1, :]  # B, C
            probs = F.softmax(logits, dim=-1)  # run softmax on the last dimension
            # probs shape: B, 1, C(normalized)
            idx_next = torch.multinomial(probs, num_samples=1)  # B, 1
            idx = torch.cat((idx, idx_next), dim=1)  # idx.shape: B, T+1

        return idx


class Transformer(nn.Module):

    def __init__(self, n_embed, block_size):
        super().__init__()
        self.block_size = block_size
        # input shape: B, T, where T is the block size,
        # for each T, we use its value to get its embedding
        # for each T, we use its index to get its position embedding
        # so, both table have same output shape of embedding dimension
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embed)
        self.pos_embedding_table = nn.Embedding(block_size, n_embed)

        # 4 head, 8dimensions
        self.blocks = nn.Sequential(
            Block(n_embed, 4),
            Block(n_embed, 4),
            Block(n_embed, 4),
            Block(n_embed, 4),
            nn.LayerNorm(n_embed),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets: torch.Tensor = None):
        # idx and targets shape are both B, T = batch_size * block_size(context window)
        # get embedding: B, T, C(embedding_dim, vacab_size for now)
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(0, T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits: torch.Tensor = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_token):
        # idx shape B, T, the input
        for _ in range(max_new_token):
            # run inference
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)  # loss is ignored here
            # logits shape: B, T, C, get the last one of the 2nd dimension,
            logits = logits[:, -1, :]  # B, C
            probs = F.softmax(logits, dim=-1)  # run softmax on the last dimension
            # probs shape: B, 1, C(normalized)
            idx_next = torch.multinomial(probs, num_samples=1)  # B, 1
            idx = torch.cat((idx, idx_next), dim=1)  # idx.shape: B, T+1

        return idx


def test_head():
    B, T, C = 4, 8, 32
    h = Head(n_embedding=C, head_size=16)
    x = torch.randn(B, T, C)
    out = h(x)
    print("out.shape", out.shape)
    print("out.value", out)


def train(iteration=100):
    from tqdm import tqdm
    print(f"training the model, iteration={iteration}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in tqdm(range(iteration)):
        xs, ys = get_batch()
        logits, loss = model(xs, ys)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print("loss:", loss)


def inference(max_new_token=100):
    prompt = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(prompt, max_new_token=max_new_token)[0].tolist()))


# main
if __name__ == "__main__":
    # test_head()

    # create the mode
    # model = SingleHeadLanguageModel(n_embed=32, block_size=8)
    # model = MultipleHeadLanguageModel(n_embed=32, block_size=8)
    model = Transformer(n_embed=32, block_size=8)
    train(iteration=10000)
    inference(max_new_token=1000)
