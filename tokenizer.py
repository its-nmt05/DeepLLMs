import json

class BPETokenizer:
    
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.base_vocab = 256
        self.num_merges = self.vocab_size - self.base_vocab
        self.merges = {}
        self.vocab = {}
        
        
    def get_pairs(self, ids):
        counts = {}
        for pair in zip(ids[0:], ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
        
    # fit the model on the input vocab
    def fit(self, text):
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        
        ids = list(text.encode('utf-8'))
        for i in range(self.num_merges):
            pairs = self.get_pairs(ids)
            top_pair = max(pairs, key=pairs.get)
            idx = 256 + i
            ids = self.merge(ids, top_pair, idx)
            
            # save the merge
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
        
        self.merges = merges
        self.vocab = vocab
            
            
    # encode a string of text into tokens
    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            pairs = self.get_pairs(tokens)
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
        
    # decode a list of tokens back into string
    def decode(self, ids):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text
    
    # save tokenizer data to config file after training    
    def save(self, filepath):
        data = {
            'vocab_size': self.vocab_size,
            'merges': {f"{k[0]}-{k[1]}": v for k, v in self.merges.items()},
            'vocab': {k: list(v) for k, v in self.vocab.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    # Load the tokenizer state from a config file
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            tokenizer = cls(data['vocab_size'])
            tokenizer.merges = {
                tuple(map(int, k.split('-'))): v 
                for k, v in data['merges'].items()
            }
            tokenizer.vocab = {int(k): bytes(v) for k, v in data['vocab'].items()} 
            return tokenizer