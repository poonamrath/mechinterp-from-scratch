# src/ioi.py
import random

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]

def make_ioi_example(rng: random.Random):
    # Template: "Alice and Bob went to the store. Bob gave a book to Alice."
    a, b = rng.sample(NAMES, 2)
    clean = f"{a} and {b} went to the store. {b} gave a book to {a}."
    corrupt = f"{a} and {b} went to the store. {a} gave a book to {a}."
    # target: the correct indirect object token should be {a} in clean case
    return clean, corrupt, a

def make_ioi_dataset(n: int, seed: int = 0):
    rng = random.Random(seed)
    data = [make_ioi_example(rng) for _ in range(n)]
    return data
