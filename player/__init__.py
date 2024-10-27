import torch
from movement import possible_moves

mp = {'p': 1, 'r': 2, 'n': 3, 'b': 4, 'q': 5, 'k': 6}
path = __file__[:__file__.rfind('\\')] + '/model/model.pt'
model = torch.load(path, map_location='cpu')

replace = {' ': ' ', 'p': 'P', 'r': 'R', 'n': 'N', 'b': 'B', 'q': 'Q', 'k': 'K'}
for k, v in list(replace.items()):
    replace[v] = k


def get_move(board, side=0):
    moves = []
    for x in range(8):
        for y in range(8):
            if board[y][x].islower() == (side == 0):
                for X, Y in possible_moves(board, x, y):
                    moves.append((y * 8 + x, Y * 8 + X))
    board = ''.join(j for i in board for j in i)
    if side == 0:
        board = ''.join(board[i * 8:i * 8 + 8] for i in range(7, -1, -1))
        board = ''.join(replace[i] for i in board)
    input = []
    for c in board:
        pt = [0] * 13
        pos = 0
        if c != ' ':
            pos = mp[c.lower()] + (0 if c.isupper() else 6)
        pt[pos] = 1
        input += pt
    output = [i.item() for i in model(torch.tensor(input, dtype=torch.float32))]

    if side == 0:
        output = [j for i in range(7, -1, -1) for j in output[i * 8:i * 8 + 8]]

    scores = [output[frm] * output[to] for frm, to in moves]
    i = scores.index(max(scores))
    frm, to = moves[i]
    x, y, X, Y = frm % 8, frm // 8, to % 8, to // 8
    return (x, y), (X, Y)
