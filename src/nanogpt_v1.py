import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NanoGptV1(nn.Module):

    def __init__(self, vocab_size, n_embed=32, n_head=1, block_size=8, head_size=32):
        super().__init__()
        self.block_size = block_size
        # embedding token to n_embed dimension
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # the input length is block_size, for each position, it gets an embedding
        self.position_embedding_token = nn.Embedding(block_size, n_embed)

        # single head
        if n_head == 1:
            from modules import Head
            self.head = Head(n_embed=n_embed, head_size=head_size, block_size=block_size)
        else:
            from modules import MutilHead
            self.head = MutilHead(n_embed=n_embed, n_head=n_head, block_size=block_size)

        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, idx, target=None):
        # idx: a batch of tokens (integers)
        B, T = idx.shape
        tok_embedding = self.token_embedding_table(idx)
        pos_embedding = self.position_embedding_token(torch.arange(0, T))
        x = tok_embedding + pos_embedding
        x = self.head(x)
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
    n_head = 4
    batch_size = 4
    block_size = 8
    head_size = 32
    nano_gpt_v1 = NanoGptV1(vocab_size=tokenizer.vocab_size,
                            n_embed=n_embed,
                            n_head=n_head,
                            head_size=head_size,
                            block_size=block_size).to(device)

    text_generator = TextGenerator(nano_gpt_v1, tokenizer, dataset, batch_size=batch_size, block_size=block_size)
    text_generator.train_and_generate()
