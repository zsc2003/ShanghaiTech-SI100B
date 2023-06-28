import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


def Americanconfirmed(data):

    start = (2021, 12, 12)
    end = (2021, 12, 20)

    assert isinstance(start, tuple) and len(start) == 3, 'invalid start'
    assert isinstance(end, tuple) and len(end) == 3, 'invalid end'
    assert datetime.date(start[0], start[1], start[2]) < datetime.date(end[0], end[1], end[2]), 'smaller'

    date_delta = datetime.timedelta(days=1)
    start_date = datetime.date(start[0], start[1], start[2])
    end_date = datetime.date(end[0], end[1], end[2])

    current_date = start_date
    output = []
    days = []
    while current_date <= end_date:
        year, month, day = int(current_date.year), int(current_date.month), int(current_date.day)
        date_str = f'{month}/{day}/{year}'
        days.append((year, month, day))

        ############### real code
        item = data[year, month, day][state]['confirmed']
        if math.isnan(item) is False:
            output.append(item)
        else:
            output.append(0)
        ###############

        current_date += date_delta

    days = list(range(1, 619))#
    graph_data = {'day': days, option: output}
    df = pd.DataFrame(graph_data)
    sns.relplot(x="day", y=option, ci=None, kind="line", data=df)
    #print(df)

    #sns.set_theme(style="darkgrid")

    # Plot the responses for different events and regions
    #sns.lineplot(x="day", y=option, hue='region', style="event", data=df)


    import matplotlib.dates as mdate
    sns.relplot(x="date", y='confirmed', ci=None, kind="line", data=df)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    plt.xticks(pd.date_range('2021-1-1', '2021-1-7', freq='1d'))
    plt.show()













if __name__ == '__main__':
    import JHU_spider as dt

    start = datetime.date(2020, 4, 12)
    today = datetime.date.today()
    start_tuple = (start.year, start.month, start.day)
    today_tuple = (today.year, today.month, today.day)

    data = dt.MultipleData(start=start_tuple, end=today_tuple)

    linechart1(data)



'''
def Americanconfirmed(data):
    import pandas as pd
    import math
    import matplotlib.pyplot as plt
    import seaborn as sns
    import datetime

    output = []
    date=[]
    tot=0
    daterange=pd.date_range("2021-1-1","2021-1-7",freq='1d')
    for month in range(1, 2):
        for day in range(1, 8):
            try:
                x = data[2021, month, day].attr('confirmed')
            except KeyError:
                continue
            l = len(x)
            sum = 0
            for i in range(l):
                if math.isnan(output[i])==False:
                    continue
                else:
                    sum+=int(x[i])

            output.append(sum)
            date.append(daterange[tot])
            tot+=1

    length=len(output)
    for i in range(length-1, 0, -1):
        output[i] -= output[i - 1]
    output[0] = output[1]

    a = {"date": date, 'confirmed': output}
    df = pd.DataFrame(a, index=date)
    #print(df)

    import matplotlib.dates as mdate
    sns.relplot(x="date", y='confirmed', ci=None, kind="line", data=df)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d'))
    plt.xticks(pd.date_range('2021-1-1', '2021-1-7', freq='1d'))
    plt.show()
'''