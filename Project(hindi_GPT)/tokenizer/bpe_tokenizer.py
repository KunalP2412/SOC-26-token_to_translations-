import json
import argparse

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and (ids[i + 1] == pair[1]):
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

class BPETokenizer:

    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size, verbose=True):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        ids = list(text.encode('utf-8'))
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (i + 1) % 100 == 0:
                print(f'merge {i + 1}/{num_merges}: {pair} -> {new_id} ({self.vocab[new_id]}) had {stats[pair]} occurrences')

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text):
        ids = list(text.encode('utf-8'))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        b = b''.join((self.vocab[i] for i in ids))
        return b.decode('utf-8', errors='replace')

    def save(self, path):
        data = {'merges': [[list(k), v] for k, v in self.merges.items()]}
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        tok = cls()
        for pair, new_id in data['merges']:
            pair = tuple(pair)
            tok.merges[pair] = new_id
            tok.vocab[new_id] = tok.vocab[pair[0]] + tok.vocab[pair[1]]
        return tok
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train'])
    parser.add_argument('input_file')
    parser.add_argument('--vocab_size', type=int, default=2000)
    parser.add_argument('--out', default='bpe.json')
    args = parser.parse_args()
    text = open(args.input_file, encoding='utf-8').read()
    tok = BPETokenizer()
    tok.train(text, args.vocab_size)
    tok.save(args.out)
    print(f'saved {args.out}, vocab_size={tok.vocab_size}')
    sample = 'नमस्ते दुनिया, यह एक परीक्षण है।'
    ids = tok.encode(sample)
    print(f'sample encode: {len(sample)} chars -> {len(ids)} tokens')
    print(f'decode check: {tok.decode(ids) == sample}')