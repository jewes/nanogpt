import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BigramModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, target=None):
        # idx.shape = B, T
        logits = self.embedding_table(idx)

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
            # logits.shape = B, T, C
            logits, _ = self(idx)

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


if __name__ == "__main__":
    from text_gen import load_dataset_and_tokenizer, TextGenerator

    dataset, tokenizer = load_dataset_and_tokenizer()
    bigram_model = BigramModel(tokenizer.vocab_size).to(device)
    batch_size = 4
    block_size = 8
    text_generator = TextGenerator(bigram_model, tokenizer, dataset, batch_size=batch_size, block_size=block_size)
    text_generator.train_and_generate()
