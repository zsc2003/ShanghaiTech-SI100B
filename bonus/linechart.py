import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


def linechart1(data, state: str, option: str):

    start = (2021, 12, 14)
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
        item = data[year, month, day][state][option]
        if math.isnan(item) is False:
            output.append(item)
        else:
            output.append(0)
        ###############

        current_date += date_delta

    days = list(range(1, 8))#
    graph_data = {'day': days, option: output}
    df = pd.DataFrame(graph_data)
    sns.relplot(x="day", y=option, ci=None, kind="line", data=df)
    #print(df)

    #sns.set_theme(style="darkgrid")

    # Plot the responses for different events and regions
    #sns.lineplot(x="day", y=option, hue='region', style="event", data=df)

    plt.show()


if __name__ == '__main__':
    import JHU_spider as dt

    start = datetime.date(2021, 12, 14)
    today = datetime.date.today()
    start_tuple = (start.year, start.month, start.day)
    today_tuple = (today.year, today.month, today.day)

    data = dt.MultipleData(start=start_tuple, end=today_tuple)

    while True:
        state = input('input state: ')
        option = input('input option: ')
        linechart1(data=data, state=state, option=option)

