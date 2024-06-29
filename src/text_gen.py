import torch
import torch.optim

from dataset import Dataset
from tokenizer import CharTokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_dataset_and_tokenizer(filename="input.txt", val_size=0.1):
    print(f"loading {filename} ... ")
    with open(filename) as f:
        text = f.read()

    tokenizer = CharTokenizer(text=text)
    dataset = Dataset(data=tokenizer.encode(text), val_size=val_size)
    return dataset, tokenizer


class TextGenerator:

    def __init__(self, model, tokenizer, dataset, batch_size, block_size):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.block_size = block_size
        self.eval_interval = 500

    def print_model_parameter(self):
        # 计算可训练参数量
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # 计算不可训练参数量
        non_trainable_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

        # 计算缓冲区量（如果有）
        buffer_params = sum(b.numel() for b in self.model.buffers())
        print(f"model parameters: trainable={trainable_params}, non_trainable={non_trainable_params}, buffer={buffer_params}")

    def train(self, iteration=100, lr=1e-3):
        self.print_model_parameter()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        for it in range(iteration):
            if it % self.eval_interval == 0 or it == iteration - 1:
                loss = self.estimate_loss()
                print(f"it = {it:05d}, train_loss={loss['train']:.4f}, eval_loss={loss['val']:.4f}")
            xs, ys = self.dataset.get_batch(self.batch_size, self.block_size)
            logits, loss = self.model(xs, ys)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def estimate_loss(self, eval_iters=200):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                xs, ys = self.dataset.get_batch(batch_size=self.batch_size, block_size=self.block_size, split=split)
                logits, loss = self.model(xs, ys)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def generate(self, input_text=None, max_new_token=10):
        input_data = torch.zeros((1, 1), dtype=torch.long, device=device)
        if input_text is not None:
            input_data = torch.tensor([self.tokenizer.encode(input_text)], device=device)
        return self.tokenizer.decode(self.model.generate(input_data, max_new_token=max_new_token)[0].tolist())

    def train_and_generate(self, it=5000, max_new_token=1000):
        self.train(iteration=it)
        generated_text = self.generate(max_new_token=max_new_token)
        print(f"text generated after {it} training iteration: {generated_text}")
