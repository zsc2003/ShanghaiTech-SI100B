#!/usr/bin/env python3

def solve(data):
    ans = 16

    # implement your algorithm here
    a = [[[0] * 4 for i in range(4)] for j in range(24)]
    for i in range(12):
        for j in range(4):
            a[i][j][j] = 1
    for i in range(12, 24):
        for j in range(4):
            a[i][j][3 - j] = 1
    n = 0

    for i in range(0, 4, 1):  # hang
        for j in range(0, 4, 1):  # lie
            for k in range(0, 2, 1):  # duijiaoxiao
                num = 0
                for row in range(0, 4, 1):
                    for column in range(0, 4, 1):
                        flag = 0
                        if row == column and k == 0:
                            flag = 1
                            if data[row][column] == 0:
                                num += 1
                        if row + column == 3 and k == 1:
                            flag = 1
                            if data[row][column] == 0:
                                num += 1
                        if flag == 1:
                            continue
                        if row != i and column != j:
                            continue
                        if data[row][column] == 0:
                            num += 1
                ans = min(ans, num)

    return ans


def main():
    data = [[1, 1, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
    print(solve(data))


if __name__ == "__main__":
    main()