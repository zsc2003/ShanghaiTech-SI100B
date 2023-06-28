#!/usr/bin/env python3

def solve(data):
    all_solution_set = [[[0] * 4 for i in range(4)] for j in range(24)]

    for typo in range(0, 12):
        for i in range(0, 4):
            all_solution_set[typo][i][i] = 1

    for typo in range(12, 24):
        for i in range(0, 4):
            all_solution_set[typo][i][3 - i] = 1

    typo = 0
    for row in range(0, 4):
        for column in range(0, 4):
            if row == column:
                continue
            all_solution_set[typo][row] = [1, 1, 1, 1]
            all_solution_set[typo][0][column], all_solution_set[typo][1][column], all_solution_set[typo][2][column], \
            all_solution_set[typo][3][column] = 1, 1, 1, 1
            typo += 1
            
    for row in range(0, 4):
        for column in range(0, 4):
            if row + column == 3:
                continue
            all_solution_set[typo][row] = [1, 1, 1, 1]
            all_solution_set[typo][0][column], all_solution_set[typo][1][column], all_solution_set[typo][2][column], \
            all_solution_set[typo][3][column] = 1, 1, 1, 1
            typo += 1

    # 2 遍历解集，记录
    all_diff_set = [9] * 24
    for typo in range(24):
        for row in range(4):
            for column in range(4):
                if all_solution_set[typo][row][column] == 1 and data[row][column] == 1:
                    all_diff_set[typo] -= 1


    # 万不得已的穷举
    # 1 生成列表
    '''
    all_solution_set = [
        [[1, 1, 1, 1],
         [0, 1, 0, 0],
         [0, 1, 1, 0],
         [0, 1, 0, 1]],

        [[1, 1, 1, 1],
         [0, 1, 1, 0],
         [0, 0, 1, 0],
         [0, 0, 1, 1]],

        [[1, 1, 1, 1],
         [0, 1, 0, 1],
         [0, 0, 1, 1],
         [0, 0, 0, 1]],  # 3

        [[1, 0, 0, 0],
         [1, 1, 1, 1],
         [1, 0, 1, 0],
         [1, 0, 0, 1]],

        [[1, 0, 1, 0],
         [1, 1, 1, 1],
         [0, 0, 1, 0],
         [0, 0, 1, 1]],

        [[1, 0, 0, 1],
         [1, 1, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 0, 1]],  # 6

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 0, 0, 1]],

        [[1, 1, 0, 0],
         [0, 1, 0, 0],
         [1, 1, 1, 1],
         [0, 1, 0, 1]],

        [[1, 0, 0, 1],
         [0, 1, 0, 1],
         [1, 1, 1, 1],
         [0, 0, 0, 1]],  # 9

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 0, 1, 0],
         [1, 1, 1, 1]],

        [[1, 1, 0, 0],
         [0, 1, 0, 0],
         [0, 1, 1, 0],
         [1, 1, 1, 1]],

        [[1, 0, 1, 0],
         [0, 1, 1, 0],
         [0, 0, 1, 0],
         [1, 1, 1, 1]],  # 12        switch

        [[1, 1, 1, 1],
         [1, 0, 1, 0],
         [1, 1, 0, 0],
         [1, 0, 0, 0]],

        [[1, 1, 1, 1],
         [0, 1, 1, 0],
         [0, 1, 0, 0],
         [1, 1, 0, 0]],

        [[1, 1, 1, 1],
         [0, 0, 1, 0],
         [0, 1, 1, 0],
         [1, 0, 1, 0]],  # 15

        [[1, 0, 0, 1],
         [1, 1, 1, 1],
         [1, 1, 0, 0],
         [1, 0, 0, 0]],

        [[0, 1, 0, 1],
         [1, 1, 1, 1],
         [0, 1, 0, 0],
         [1, 1, 0, 0]],

        [[0, 0, 0, 1],
         [1, 1, 1, 1],
         [0, 1, 0, 1],
         [1, 0, 0, 1]],  # 18

        [[1, 0, 0, 1],
         [1, 0, 1, 0],
         [1, 1, 1, 1],
         [1, 0, 0, 0]],

        [[0, 0, 1, 1],
         [0, 0, 1, 0],
         [1, 1, 1, 1],
         [1, 0, 1, 0]],

        [[0, 0, 0, 1],
         [0, 0, 1, 1],
         [1, 1, 1, 1],
         [1, 0, 0, 1]],  # 21

        [[0, 1, 0, 1],
         [0, 1, 1, 0],
         [0, 1, 0, 0],
         [1, 1, 1, 1]],

        [[0, 0, 1, 1],
         [0, 0, 1, 0],
         [0, 1, 1, 0],
         [1, 1, 1, 1]],

        [[0, 0, 0, 1],
         [0, 0, 1, 1],
         [0, 1, 0, 1],
         [1, 1, 1, 1]],  # 24
    ]
    '''

    # 2 遍历解集，记录
    all_diff_set = [9] * 24
    for typo in range(24):
        for row in range(4):
            for column in range(4):
                if all_solution_set[typo][row][column] == 1 and data[row][column] == 1:
                    all_diff_set[typo] -= 1

    return min(all_diff_set)


def main():
    data = [[1, 1, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]]
    print(solve(data))


if __name__ == "__main__":
    main()
