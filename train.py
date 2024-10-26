import torch
import matplotlib.pyplot as plt

from model import Model
from tokenizer import BPETokenizer

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
vocab_size = 1000   # size of vocabulary

bpe = BPETokenizer.load('bpe.json')
dataset = 'sample.txt'

# Splitting dataset into train and test splits
def load_data(file_name):
    with open(file_name, 'r') as file:
        text = file.read()
        data = torch.tensor(bpe.encode(text), dtype=torch.long)
        n = int(0.9*len(data))   # 90% for training
        train_data = data[:n]
        val_data = data[n:]
    return train_data, val_data

# loading the dataset
train_data, val_data = load_data(dataset)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def train():
    model = Model(vocab_size, block_size, n_layer, n_embd, n_head, dropout)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    @torch.no_grad()
    def estimate_loss(iters=eval_iters):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(iters)
            for k in range(iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    steps = []
    
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(iters=10)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # Store losses for plotting
            steps.append(iter)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


    # Plotting the loss graph
    plt.figure(figsize=(8, 6))
    plt.plot(steps, train_losses, label='Train Loss')
    plt.plot(steps, val_losses, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Time')
    plt.legend()
    plt.show()