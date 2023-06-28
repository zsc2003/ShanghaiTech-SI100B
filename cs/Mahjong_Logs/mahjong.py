def solve(filename):
    from check import CheckWin
    xfile=open(filename)
    winner=open("winner.csv", "w")
    battle=open("battle.csv","w")
    p=[[] for i in range(40)]
    #print(p)
    tile=[[] for i in range(40)]
    t3=[[] for i in range(40)]
    #print(p)
    win=[0]*40
    turn=[-1]*40
    tot=0
    flag1=0
    flag2=0
    flag=0
    for l in xfile:
        # print(line,end='')
        line=l.strip('\n')
        tot+=1
        if flag1==0:
            start=line
            flag1=1
            continue
        if flag2==0:
            name=line.split(",")
            tot=0
            flag2=1
            continue
        s=line.split(",")
        for j in range(4):
            if s[j]!='':
                tile[j].append(s[j])
       # print(s)
    #for i in range(4):
        #print(tile[i])
    score=[50000,50000,50000,50000]
    game=0
    s = "Game " + str(game) + "\n"
    battle.write(s)
    for i in range(4):
        s = name[i] + "," + str(score[i]) + "\n"
        battle.write(s)
        t3[i].append(score[i])
    battle.write("\n")
    task2=[]
    for i in range(4):
        if start==name[i]:
            dealer=i
            pos=i
            break
    game=1
    paishu=0
    while 1:
        if paishu==136:
            #print("DDDDDDDDDDDD")
            paishu=0
            turn[pos]+=1
            p[pos].remove(tile[pos][turn[pos]])
            winner.write("Draw\n")
            s = "Game " + str(game) + "\n"
            battle.write(s)
            for i in range(4):
                s = name[i] + "," + str(score[i]) + "\n"
                battle.write(s)
                t3[i].append(score[i])
            battle.write("\n")
            game+=1
            for i in range(4):
                p[i].clear()
            dealer+=1
            dealer%=4
            pos=dealer
            if turn[pos]==(len(tile[pos])-1):
                break
            continue
        pos%=4
        turn[pos]+=1
        if len(p[pos])==14:
            p[pos].remove(tile[pos][turn[pos]])
            pos+=1
            continue
        p[pos].append(tile[pos][turn[pos]])
        paishu+=1
        #for i in range(4):
        #    print(p[i])
        #print("-----------------------------------")
        if len(p[pos])!=14:
            pos+=1
            continue
        delta=[3600]
        if CheckWin(p[pos],delta):
            #print(p[pos])
            paishu=0
            for i in range(4):
                if i==pos:
                    score[i]+=delta[0]
                else:
                    score[i]-=delta[0]//3
            s = "Game " + str(game) + "\n"
            battle.write(s)
            for i in range(4):
                s = name[i] + "," + str(score[i]) + "\n"
                battle.write(s)
                t3[i].append(score[i])
            battle.write("\n")
            game+=1
            task2.append(tile[pos][turn[pos]])
            for i in range(4):
                p[i].clear()
            #print(name[pos])
            sout=name[pos]+"\n"
            winner.write(sout)
            win[pos]+=1
            dealer+=1
            dealer%=4
            pos=dealer
            if turn[pos]==(len(tile[pos])-1):
                break
#task1------------------------------------------
    game-=1
    winner.write("\n")
    for i in range(4):
        sout=name[i]+','
        #print(name[i],end=',')
        if game==0:
            x=0
        else:
            x=float(win[i]*100/game)
        x=("%.2lf" % x)
        #print(x)
        #print("%.2lf" % x,end="%")
        #print("")
        sout+=str(x)+"%\n"
        winner.write(sout)
#task2------------------------------------------
    #print(task2)
    task2.sort()
    #print(task2)
    a=[]
    length=len(task2)
    if length==0:
        cnt=0
    else:
        x=[task2[0],1]
        a.append(x)
        cnt=1
    for i in range(1,length):
        if task2[i]!=task2[i-1]:
            cnt+=1
            x=[task2[i],1]
            a.append(x)
        else:
            a[cnt-1][1]+=1
    #print(a)
    #print(b)
    b=sorted(a,key=lambda k:k[1],reverse=1)
    #print(a)
    #print(b)
    s=""
    for i in range(cnt):
        if game==0:
            num=0
        else:
            num=b[i][1]/game
        num=float(num*100)
        num=("%.2lf" % num)
        s+=b[i][0]+","+str(num)+"%\n"
        #print(s)

    tile=open("tile.csv","w")
    #s="3m,3m\n60,60"
    tile.write(s)
    #tile.write()
#task3----------------------------

    import matplotlib.pyplot as plt
    x=[]
    for i in range(game+1):
        x.append(i)
    plt.xlim(-0.5,game+0.5)
    plt.xticks(range(0,game+1))
    for i in range(4):
        y=t3[i]
        plt.plot(x,y)
    legend=[name[0],name[1],name[2],name[3]]
    plt.legend(legend)
    plt.title("Result")
    plt.show()

solve("test2.csv")
#["1p","1p","1p","2p","2p","2p","3p","3p","3p","4p","4p","4p","5p","5p"]