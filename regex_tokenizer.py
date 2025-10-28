# This exercise is from the Karpathy's minbpe repository: https://github.com/karpathy/minbpe
import regex as re

def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i+=2
    else:
      newids.append(ids[i])
      i += 1
  return newids

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer:
  def __init__(self):
    self.vocab = {idx: bytes([idx]) for idx in range(256)} 
    self.merges = {}

  def train(self, text, vocab_size, verbose=False):
    if vocab_size < 256: 
      raise ValueError("vocab_size must be greater than 256")
    num_merges = vocab_size - 256
    text_splits = re.findall(GPT4_SPLIT_PATTERN, text)
    ids = [list(text.encode("utf-8")) for text in text_splits]
    for i in range(num_merges):
      stats = {}
      for chunk in ids:
        get_stats(chunk, stats)
      pair = max(stats, key=stats.get)
      idx = 256 + i
      if verbose:
        print(f"Merging {pair} into {idx}")
      ids = [merge(chunk, pair, idx) for chunk in ids]
      self.merges[pair] = idx
      self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
      if verbose:
        print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({self.vocab[idx]}) had {stats[pair]} occurrences")

  def encode(self, text):
    tokens = list(text.encode("utf-8"))
    while True:
      stats = get_stats(tokens)
      pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
      if pair not in self.merges:
        break
      idx = self.merges[pair]
      tokens = merge(tokens, pair, idx)
    return tokens

  def decode(self, ids):
    tokens = b"".join(self.vocab[i] for i in ids)
    return tokens.decode("utf-8" , errors="replace")

if __name__ == "__main__":
  tokenizer = RegexTokenizer()
  text = open("tailorswift.txt", "r").read()
  tokenizer.train(text, 10000, verbose=False)
  print(10*"-")
  print("Vocabulary:")
  print(tokenizer.vocab)
  print(10*"-")
  print("Merges:")
  print(tokenizer.merges)
  print(10*"-")
  print(tokenizer.encode("hello world!!!? (안녕하세요!) lol123 😉"))
  print(10*"-")
  print(tokenizer.decode(tokenizer.encode("hello world!!!? (안녕하세요!) lol123 😉")))