
class CharTokenizer:

    def __init__(self, text):
        self.vocabs = sorted(list(set(text)))
        self.vocab_size = len(self.vocabs)

        self.ctoi = dict()
        self.itoc = dict()
        for i, c in enumerate(self.vocabs):
            self.ctoi[c] = i
            self.itoc[i] = c

    def encode(self, chars: str):
        return [self.ctoi[c] for c in chars]

    def decode(self, idx: list):
        return "".join([self.itoc[i] for i in idx])


if __name__ == "__main__":
    with open("input.txt") as f:
        text = f.read()
        t = CharTokenizer(text)
        print("vocabs:", "".join(t.vocabs))
        assert "hello world" == (t.decode(t.encode("hello world")))
