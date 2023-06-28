import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import datetime



def histogram1(states_list, data, date: str, attr: str):

    date = date.split(',')
    attr_list = data[int(date[0]), int(date[1]), int(date[2])].attr(attr)

    length = len(attr_list)
    for i in range(length):
        # if math.isnan(output[i])==0:
        if isinstance(attr_list[i], str):
            attr_list[i] = int(float(attr_list[i]))
        else:
            attr_list[i] = 0
    data = {"state": states_list, attr: attr_list}

    df = pd.DataFrame(data, index=states_list)
    df=df[:32]
    graph = sns.barplot(x=attr_list, y="state", ci=67, orient="h", data=df)  # orient="h"表示横向条形图
    #print(len(states_list))
    for i in range(0,32):
        graph.text(attr_list[i], i, (lambda x: format(x, ','))(attr_list[i]), color="black", ha="left")  # lambda加千分位符
    plt.show()
    df = pd.DataFrame(data, index=states_list)
    df=df[32:65]
    graph = sns.barplot(x=attr_list, y="state", ci=67, orient="h", data=df)  # orient="h"表示横向条形图
    # print(len(states_list))
    for i in range(32,64):
        graph.text(attr_list[i], i-32, (lambda x: format(x, ','))(attr_list[i]), color="black", ha="left")  # lambda加千分位符
    plt.show()


states_list = ('Alabama', 'Alaska', 'American_Samoa', 'Arizona', 'Arkansas', 'California', 'Colorado',
                   'Connecticut', 'Delaware', 'Department_of_Defense', 'Diamond_Princess', 'District_of_Columbia',
                   'Federal_Bureau_of_Prisons', 'Florida', 'Georgia', 'Grand_Princess', 'Guam', 'Hawaii', 'Idaho',
                   'Illinois', 'Indian_Health_Services', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
                   'Long_Term_Care_(LTC)_Program', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
                   'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New_Hampshire',
                   'New_Jersey', 'New_Mexico', 'New_York', 'North_Carolina', 'North_Dakota', 'Northern_Mariana_Islands',
                   'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Puerto_Rico', 'Rhode_Island', 'South_Carolina',
                   'South_Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Veterans_Health_Administration',
                   'Virgin_Islands', 'Virginia', 'Washington', 'West_Virginia', 'Wisconsin', 'Wyoming', 'the_US')
    # 64 items


if __name__ == '__main__':
    import JHU_spider as dt

    start = datetime.date(2020, 4, 12)
    today = datetime.date.today()
    start_tuple = (start.year, start.month, start.day)
    today_tuple = (today.year, today.month, today.day)

    data = dt.MultipleData(start=start_tuple, end=today_tuple)

    while True:
        date = input('input date: ')  # 2021,1,1 confirmed     ##########
        attr = input('input attr: ')
        histogram1(states_list=states_list, data=data, date=date, attr=attr)
