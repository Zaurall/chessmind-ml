import pandas as pd
import torch
from random import shuffle

def convert(x):
    return ''.join(' ' * int(i) if i.isdigit() else i for i in x)


data = pd.read_csv('moves.csv')
data = data[['Side', 'ending_position', 'starting_position']]
data['ending_position'] = data['ending_position'].transform(convert)
data['starting_position'] = data['starting_position'].transform(convert)
mp = {'p': 0, 'r': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5}

input = []
output = []
for j, i in enumerate(data.itertuples(False)):
    side, start, end = i.Side, i.starting_position, i.ending_position
    new = [int(i != j) for i, j in zip(start, end)]
    if sum(new) != 2: continue
    output.append(new)
    side = 1 if side == 'black' else -1
    input.append([])
    for c in start:
        pt = [0] * 6
        if c != ' ':
            pt[mp[c.lower()]] = side if c.lower() == c else -side
        input[-1] += pt

del data
i = list(range(len(input)))
shuffle(i)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

input = torch.tensor(input, dtype=torch.float32).to(device)
output = torch.tensor(output, dtype=torch.float32).to(device)


def get_batch(size=32, test=False):
    n = len(input)
    l, r = 0, int(n * 0.8)
    if test:
        l, r, = r, n - 1
    i = torch.randint(l, r, (size,))
    return input[i], output[i]


if __name__ == '__main__':
    inp, out = get_batch()
    print(inp.shape)
    print(inp)
    print(out.shape)
    print(out)
