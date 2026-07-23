import sys
import argparse
import torch
from model import GPT
sys.path.insert(0, 'tokenizer')

def load_tokenizer(ckpt):
    if ckpt['tokenizer_type'] == 'char':
        from char_tokenizer import CharTokenizer
        return CharTokenizer.load('tokenizer/char_tokenizer.json')
    else:
        from bpe_tokenizer import BPETokenizer
        return BPETokenizer.load(ckpt['bpe_file'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path')
    parser.add_argument('--prompt', default='भारत')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt_path, map_location=device)
    tok = load_tokenizer(ckpt)
    model = GPT(ckpt['vocab_size'], ckpt['block_size'], ckpt['n_embd'], ckpt['n_head'], ckpt['n_layer']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)
    print(tok.decode(out[0].tolist()))
if __name__ == '__main__':
    main()