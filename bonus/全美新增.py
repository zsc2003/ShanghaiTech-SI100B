if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    x = datetime.datetime.now()
    data = MultipleData()
    y = datetime.datetime.now()
    print(y - x)

    day = []
    output = []

    for i in range(1, 8):
        day.append(i)
        x = data[2021, 1, i].attr('confirmed')
        x = list(map(int, x))
        output.append(sum(x))
    for i in range(6, 0, -1):
        output[i] -= output[i - 1]
    output[0] = output[1]
    a = {"day": day, 'confirmed': output}
    df = pd.DataFrame(a, index=day)
    #print(df)
    sns.relplot(x="day", y='confirmed', ci=None, kind="line", data=df)
    plt.show()