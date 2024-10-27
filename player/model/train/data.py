import pandas as pd
import torch
from random import shuffle


def convert(x):
    return ''.join(' ' * int(i) if i.isdigit() else i for i in x)


data = pd.read_csv('moves.csv')
data = data[['Side', 'ending_position', 'starting_position']]
data['ending_position'] = data['ending_position'].transform(convert)
data['starting_position'] = data['starting_position'].transform(convert)
mp = {'p': 1, 'r': 2, 'n': 3, 'b': 4, 'q': 5, 'k': 6}
replace = {' ': ' ', 'p': 'P', 'r': 'R', 'n': 'N', 'b': 'B', 'q': 'Q', 'k': 'K'}
for k, v in list(replace.items()):
    replace[v] = k

moves = {}

for i in data.itertuples(False):
    side, start, end = i.Side, i.starting_position, i.ending_position
    new = [int(i != j) for i, j in zip(start, end)]
    if sum(new) != 2: continue
    if side == 'black':
        start = ''.join(start[i * 8:i * 8 + 8] for i in range(7, -1, -1))
        start = ''.join(replace[i] for i in start)
        end = ''.join(end[i * 8:i * 8 + 8] for i in range(7, -1, -1))
        end = ''.join(replace[i] for i in end)
    if start not in moves:
        moves[start] = set()
    moves[start].add(end)

input = []
output = []
for key, value in moves.items():
    new = [0] * 64
    for v in value:
        for i, j, k in zip(range(64), key, v):
            new[i] += float(k != j) / len(value)
    output.append(new)
    input.append([])
    for c in key:
        pt = [0] * 13
        pos = 0
        if c != ' ':
            pos = mp[c.lower()] + (0 if c.isupper() else 6)
        pt[pos] = 1
        input[-1] += pt

del data
i = list(range(len(input)))
shuffle(i)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

input = torch.tensor(input, dtype=torch.float32).to(device)[i]
output = torch.tensor(output, dtype=torch.float32).to(device)[i]


def get_batches(batch_size=64, test=False):
    n = len(input)
    l, r = 0, int(n * 0.8)
    if test:
        l, r, = r, n
    i = list(range(l, r))
    i = i[:len(i) // batch_size * batch_size]
    shuffle(i)
    inp = input[i].reshape((-1, batch_size, input.shape[-1]))
    out = output[i].reshape((-1, batch_size, output.shape[-1]))
    for i, o in zip(inp, out):
        yield i, o


if __name__ == '__main__':
    for inp, out in get_batches():
        print(inp.shape)
        print(inp)
        print(out.shape)
        print(out)
        break
