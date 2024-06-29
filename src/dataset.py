import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Dataset:

    def __init__(self, data, val_size=0.1):
        assert 0 <= val_size < 0.25
        train_size = int(len(data) * (1 - val_size))
        self.train_set = torch.tensor(data[:train_size], device=device)
        self.val_set = torch.tensor(data[train_size + 1:], device=device)

    def get_batch(self, batch_size, block_size, split='train'):
        """
        randomly get a batch for training or evaluation
        :param split: train or val
        :param batch_size:
        :param block_size:
        :return:
        """
        assert split == "train" or split == "val"
        assert batch_size > 0 and block_size > 0

        data = self.train_set if split == "train" else self.val_set
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        xs = torch.stack([data[i: i + block_size] for i in ix])
        ys = torch.stack([data[i + 1: i + 1 + block_size] for i in ix])

        return xs, ys
