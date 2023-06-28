from itertools import permutations
from copy import deepcopy

score_mapping = {6: 10000, 7: 36, 8: 720, 9: 360, 10: 80, 11: 252, 12: 108, 13: 72, 14: 54,
                 15: 180, 16: 72, 17: 180, 18: 119, 19: 36, 20: 306, 21: 1080, 22: 144, 23: 1800, 24: 3600}


def solveq1(data):
    return max([score_mapping[entry]
                for entry in [sum(data[0]), sum(data[1]), sum(data[2]), data[0][0] + data[1][0] + data[2][0],
                              data[0][1] + data[1][1] + data[2][1], data[0][2] + data[1][2] + data[2][2],
                              data[0][0] + data[1][1] + data[2][2], data[0][2] + data[1][1] + data[2][0]]])


def solveq2(data):
    cases_tuple = tuple([fill_in_zeros(data, permutation) for permutation in
                         permutations(set(range(10)) - set(data[0]) - set(data[1]) - set(data[2]))])

    sums_list =[[] for i in range(8)]

    for case in cases_tuple:
        sums_list[0].append(score_mapping[sum(case[0])])
        sums_list[1].append(score_mapping[sum(case[1])])
        sums_list[2].append(score_mapping[sum(case[2])])
        sums_list[3].append(score_mapping[case[0][0] + case[1][0] + case[2][0]])
        sums_list[4].append(score_mapping[case[0][1] + case[1][1] + case[2][1]])
        sums_list[5].append(score_mapping[case[0][2] + case[1][2] + case[2][2]])
        sums_list[6].append(score_mapping[case[0][0] + case[1][1] + case[2][2]])
        sums_list[7].append(score_mapping[case[0][2] + case[1][1] + case[2][2]])

    return int(max(list(map(lambda x: sum(x) / len(x), sums_list))))


def fill_in_zeros(data, permutation) -> tuple:
    zero_indices_tuple = tuple([(x, y) for x in range(3) for y in range(3) if data[x][y] == 0])
    output = tuple(deepcopy(data))
    for i in range(len(zero_indices_tuple)):
        output[zero_indices_tuple[i][0]][zero_indices_tuple[i][1]] = permutation[i]
    return output


def main():
    print(solveq1([[7, 6, 9], [4, 5, 3], [2, 1, 8]]))
    print(solveq2([[0, 6, 0], [4, 0, 3], [2, 0, 0]]))


if __name__ == "__main__":
    main()
