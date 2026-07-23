import argparse
import time
import torch
from model import GPT
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 0.0003
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

def get_tokenizer(args):
    if args.tokenizer == 'char':
        import sys
        sys.path.insert(0, 'tokenizer')
        from char_tokenizer import CharTokenizer
        return CharTokenizer.from_file(f'{args.data_dir}/train.txt')
    else:
        import sys
        sys.path.insert(0, 'tokenizer')
        from bpe_tokenizer import BPETokenizer
        return BPETokenizer.load(args.bpe_file)

def get_batch(data, split_data, batch_size, block_size, device):
    ix = torch.randint(len(split_data) - block_size, (batch_size,))
    x = torch.stack([split_data[i:i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1:i + 1 + block_size] for i in ix])
    return (x.to(device), y.to(device))

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def bits_per_char(loss_nats_per_token, n_chars, n_tokens):
    bits_per_token = loss_nats_per_token / 0.6931471805599453
    tokens_per_char = n_tokens / n_chars
    return bits_per_token * tokens_per_char

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', choices=['char', 'bpe'], required=True)
    parser.add_argument('--bpe_file', default=None)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--out', default='ckpt.pt')
    args = parser.parse_args()
    print(f'device: {device}')
    tok = get_tokenizer(args)
    vocab_size = tok.vocab_size
    print(f'vocab_size: {vocab_size}')
    if args.tokenizer == 'char':
        tok.save('tokenizer/char_tokenizer.json')
    train_text = open(f'{args.data_dir}/train.txt', encoding='utf-8').read()
    val_text = open(f'{args.data_dir}/val.txt', encoding='utf-8').read()
    train_data = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tok.encode(val_text), dtype=torch.long)
    print(f'train tokens: {len(train_data):,} | val tokens: {len(val_data):,}')
    print(f'train chars: {len(train_text):,} | tokens/char: {len(train_data) / len(train_text):.3f}')
    val_tokens_per_char = len(val_data) / len(val_text)
    model = GPT(vocab_size, block_size, n_embd, n_head, n_layer, dropout).to(device)
    print(f'params: {model.num_params() / 1000000.0:.2f}M')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    t0 = time.time()
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - t0
            val_bpc = bits_per_char(losses['val'], len(val_text), len(val_data))
            print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, val bpc {val_bpc:.4f} ({elapsed:.0f}s)')
        xb, yb = get_batch(train_data, train_data, batch_size, block_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.save({'model_state': model.state_dict(), 'vocab_size': vocab_size, 'block_size': block_size, 'n_embd': n_embd, 'n_head': n_head, 'n_layer': n_layer, 'tokenizer_type': args.tokenizer, 'bpe_file': args.bpe_file}, args.out)
    print(f'saved checkpoint to {args.out}')
if __name__ == '__main__':
    main()