import csv
from check import CheckWin
from copy import deepcopy
from random import shuffle, randint

p = ['1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p']
s = ['1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s']
m = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m']
z = ['1z', '2z', '3z', '4z', '5z', '6z', '7z']

titles_template = []
for i in range(4):
    titles_template.extend(p)
    titles_template.extend(s)
    titles_template.extend(m)
    titles_template.extend(z)


def write(record: list):
    with open('your_name.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Diana'])
        writer.writerow(['Diana', 'Ava', 'Eileen', 'Carol'])

        max_len = max([len(r) for r in record])
        for i in range(max_len):
            line = []
            for j in range(4):
                if i < len(record[j]):
                    line.append(record[j][i])
                else:
                    line.append('')
            writer.writerow(line)


def gen_one_round(dealer: int, winner: int, win_pattern: list, turn_end: int):
    '''
    winner    : an int in interval:[-1, 3], -1 means draw
    winpattern: a list with length 14
    turn_end  : an int in interval:[14, 34], means the time when 'winner' wins by 'win_pattern'
    '''
    titles = deepcopy(titles_template)
    shuffle(titles)
    if winner == -1:
        win_pattern = titles[:14]
        winner = 1

    for t in win_pattern:
        id = titles.index(t)
        titles.pop(id)
    shuffle(win_pattern)

    for i in range(20):
        if i+14 < turn_end:
            win_pattern.append(titles.pop())
        else:
            win_pattern.insert(0, titles.pop())
    # print(len(win_pattern))
    # print(len(titles))
    # now 'winner' draws from 'win_pattern'
    # others draws from titles

    in_hand = [[] for i in range(4)]
    record = [[] for i in range(4)]

    for i in range(13):
        for j in range(4):
            player = (j+dealer) % 4
            draw = win_pattern.pop() if player == winner else titles.pop()
            record[player].append(draw)
            in_hand[player].append(draw)

    while True:
        for i in range(4):
            # print(record)
            player = (i+dealer) % 4
            # print(player)
            if (player != winner and titles == []) or (player == winner and win_pattern == []):
                print(player, 'Draw')
                return record

            draw = win_pattern.pop() if player == winner else titles.pop()
            record[player].append(draw)

            in_hand[player].append(draw)
            # print(i, win_pattern)
            # print(in_hand[i])

            if CheckWin(deepcopy(in_hand[player])):
                print(player, 'wins by', in_hand[player])
                return record

            # 'winner' drops in order
            drop = in_hand[player].pop(0) if player == winner else in_hand[player].pop(
                randint(0, len(in_hand[player])-1))
            record[player].append(drop)

# START AT HERE
num_round = 1
record = [[] for i in range(4)]

for i in range(num_round):
    dealer = i % 4
    winner = randint(-1, 3)
    # winner = randint(0, 3)
    turn_end = randint(14, 34)
    winNum = randint(0, 2)

    cur = gen_one_round(dealer,
                        winner,
                        deepcopy(['1m','2m','3m','4p','4p','4p','5s','6s','7s','1z','1z','1z','2z','2z']),
                        turn_end)
    for i in range(4):
        record[i].extend(cur[i])
    for i in range(4):
        print(len(cur[i]), cur[i])
    print()

write(record)
