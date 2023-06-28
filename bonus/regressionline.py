import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from plotly.graph_objs import Scatter,Layout
import plotly
import plotly.offline as py
import numpy as np
import plotly.graph_objs as go

#setting offilne
#plotly.offline.init_notebook_mode(connected=True)

def line_graph(state: str, attr: str, start=(2021, 1, 1), end=(2021, 1, 8)):
    global data

    x = [f'{year}-{month}-{day}' for year, month, day in iter(DateIterator(start=start, end=end))]
    y = [data[year, month, day][state][attr] for year, month, day in iter(DateIterator(start=start, end=end))]

    trace = go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='lines',
    )
    graph_data = [trace]
    py.iplot(graph_data)


    print(type(data))
    print(data)




if __name__ == '__main__':
    import JHU_spider as dt

    start = datetime.date(2020, 4, 12)
    today = datetime.date.today()
    start_tuple = (start.year, start.month, start.day)
    today_tuple = (today.year, today.month, today.day)

    data = dt.MultipleData(start=start_tuple, end=today_tuple)

    while True:
        state = input('input state: ')
        attr = input('input attr: ')
        start = tuple(map(int, input('input start date: ').split('-')))
        end = tuple(map(int, input('input end date: ').split('-')))

        line_graph(start=start, end=end, state=state, attr=attr)
