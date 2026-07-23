import json

class CharTokenizer:

    def __init__(self, chars: list[str]):
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def from_file(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        print(f'CharTokenizer: found {len(chars)} unique characters')
        return cls(chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, 0) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return ''.join((self.itos.get(i, '') for i in ids))

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'chars': self.chars}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data['chars'])
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python char_tokenizer.py <path_to_train.txt>')
        sys.exit(1)
    tok = CharTokenizer.from_file(sys.argv[1])
    sample = 'नमस्ते दुनिया'
    ids = tok.encode(sample)
    decoded = tok.decode(ids)
    print(f'Sample: {sample}')
    print(f'Encoded ({len(ids)} tokens): {ids}')
    print(f'Decoded: {decoded}')
    print(f'Vocab size: {tok.vocab_size}')
    tok.save('char_tokenizer.json')
    print('Saved to char_tokenizer.json')