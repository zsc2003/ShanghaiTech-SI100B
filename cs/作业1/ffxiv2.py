#!/usr/bin/env python3
score_mapping = {6: 10000, 7: 36, 8: 720, 9: 360, 10: 80, 11: 252, 12: 108, 13: 72, 14: 54,
                 15: 180, 16: 72, 17: 180, 18: 119, 19: 36, 20: 306, 21: 1080, 22: 144, 23: 1800, 24: 3600}

def row(data,i):
    return data[i][0]+data[i][1]+data[i][2]

def column(data,i):
    return data[0][i]+data[1][i]+data[2][i]

def diagonal(data,opt):
    num=0
    if opt==1:
        for i in range(3):
             num+=data[i][i]
        return num
    for i in range(3):
        num+=data[i][2-i]
    return num

def solveq1(data):
    ans = 10_000

    #print(score_mapping[7])
    ans=0
    num=[0]*8
    for i in range(3):
        num[i]=row(data,i)
    for i in range(3):
        num[i+3]=column(data,i)
    num[6]=diagonal(data,1)
    num[7]=diagonal(data,2)

    #print(num)
    for i in range(8):
        ans=max(ans,score_mapping[num[i]])

    return ans

def solveq2(data):
    ans = 10_000

    ans=0
    bin=[0]*10

    for i in range(3):
        for j in range(3):
            bin[data[i][j]]=1
    a=[0]*5
    tot=0
    for i in range(10):
        if bin[i]==0:
            a[tot]=i
            tot+=1
    data2=[[0]*3 for i in range(3)]
    num=[0]*8
    from itertools import permutations
    for perm in permutations(a):
        tot=0
        for i in range(3):
            for j in range(3):
                data2[i][j]=data[i][j]
                if data[i][j]==0:
                    data2[i][j]=perm[tot]
                    tot+=1
        for i in range(3):
            num[i]+=score_mapping[row(data2, i)]
        for i in range(3):
            num[i+3]+=score_mapping[column(data2,i)]
        num[6]+=score_mapping[diagonal(data2,1)]
        num[7]+=score_mapping[diagonal(data2,2)]
        #print(perm)
        #print(data2)
    #print(num)
    ans=max(num)/120#5!=120
    return int(ans)


def main():
    print(solveq1([[7, 6, 9], [4, 5, 3], [2, 1, 8]]))
    print(solveq2([[0, 6, 0], [4, 0, 3], [2, 0, 0]]))


if __name__ == "__main__":
    main()
