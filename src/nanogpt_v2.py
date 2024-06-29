import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NanoGptV2(nn.Module):
    """
    multi-head,
    layer normal
    residual module
    dropout
    """

    def __init__(self, vocab_size, n_layer=4, n_embed=32, n_head=1, block_size=8, dropout=0.0):
        super().__init__()
        self.block_size = block_size

        # embedding layer
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_token = nn.Embedding(block_size, n_embed)

        # self attention layers
        from modules import Block
        self.attentions = nn.Sequential(*[Block(n_embed, n_head, block_size, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, target=None):
        # idx: a batch of tokens (integers)
        B, T = idx.shape
        tok_embedding = self.token_embedding_table(idx)
        pos_embedding = self.position_embedding_token(torch.arange(0, T))
        x = tok_embedding + pos_embedding
        x = self.attentions(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # the following are same as the bigram model
        if target is None:
            return logits, None

        # calculates the loss using cross_entropy
        # logits.shape = B, T, C, where c = vocab_size in this case
        # target.shape = B, T
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        target = target.view(B * T)
        loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, idx, max_new_token):
        # idx.shape = B, T
        # assume B = 1,
        for _ in range(max_new_token):
            # for gpt model, the max token length is block_size due to the position embedding
            # so trim it
            # logits.shape = B, T, C
            logits, _ = self(idx[:, -self.block_size:])

            # the input is like: i am<next>
            # the logits shape: B, T, C
            # we only predict the next token by the current token, so we extract the last one of T
            logits = logits[:, -1, :]

            # the probability of each token
            probs = F.softmax(logits, dim=-1)

            # sample one token
            next_idx = torch.multinomial(probs, num_samples=1)

            # concate the new token and predict the next one
            idx = torch.cat((idx, next_idx), dim=-1)

        return idx


if __name__ == '__main__':
    from text_gen import load_dataset_and_tokenizer, TextGenerator

    dataset, tokenizer = load_dataset_and_tokenizer()
    n_embed = 32
    n_layer = 4
    n_head = 4
    batch_size = 4
    block_size = 8
    dropout = 0.2
    nano_gpt_v2 = NanoGptV2(vocab_size=tokenizer.vocab_size,
                            n_layer=n_layer,
                            n_embed=n_embed,
                            n_head=n_head,
                            block_size=block_size,
                            dropout=dropout
                            ).to(device)

    text_generator = TextGenerator(nano_gpt_v2, tokenizer, dataset, batch_size=batch_size, block_size=block_size)
    text_generator.train_and_generate()
