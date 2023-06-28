import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


def predict1(data):
    import numpy as np
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
    #64 items
    attr_list= ('confirmed', 'deaths', 'incident_rate', 'total_test_results', 'people_hospitalized',
                 'case_fatality_ratio', 'testing_rate', 'hospitalization_rate', 'people_fully_vaccinated',
                 'people_partially_vaccinated', 'tests_combined_total', 'tests_viral_positive',
                 'tests_viral_negative', 'people_viral_total', 'people_viral_positive')

    new_index= ('state_name','state_abbreviation','date','confirmed', 'deaths', 'incident_rate', 'total_test_results', 'people_hospitalized',
                 'case_fatality_ratio', 'testing_rate', 'hospitalization_rate', 'people_fully_vaccinated',
                 'people_partially_vaccinated', 'tests_combined_total', 'tests_viral_positive',
                 'tests_viral_negative', 'people_viral_total', 'people_viral_positive')

    new_states_abbr_dict = {'Alaska': 'AK', 'Alabama': 'AL', 'Arkansas': 'AR', 'American_Samoa': 'AS', 'Arizona': 'AZ',
                        'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'District_of_Columbia': 'DC',
                        'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Guam': 'GU', 'Hawaii': 'HI', 'Iowa': 'IA',
                        'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Kansas': 'KS', 'Kentucky': 'KY',
                        'Louisiana': 'LA', 'Massachusetts': 'MA', 'Maryland': 'MD', 'Maine': 'ME', 'Michigan': 'MI',
                        'Minnesota': 'MN', 'Missouri': 'MO', 'Northern_Mariana_Islands': 'MP', 'Mississippi': 'MS',
                        'Montana': 'MT', 'North_Carolina': 'NC', 'North_Dakota': 'ND', 'Nebraska': 'NE',
                        'New_Hampshire': 'NH', 'New_Jersey': 'NJ', 'New_Mexico': 'NM', 'Nevada': 'NV',
                        'New_York': 'NY', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
                        'Puerto_Rico': 'PR', 'Rhode_Island': 'RI', 'South_Carolina': 'SC', 'South_Dakota': 'SD',
                        'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Virginia': 'VA', 'Virgin_Islands': 'VI',
                        'Vermont': 'VT', 'Washington': 'WA', 'West_Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY',
                        'the_US': 'US'}

    delta_attr_list = tuple(map(lambda x: 'new_' + x, attr_list))
    extended_attr_list = tuple(list(attr_list) + list(delta_attr_list))
    # 30 items
    new_extended_attr_list = tuple(list(new_index) + list(delta_attr_list))

    start = (2021, 1, 1)
    end = (2021, 1, 7)

    assert isinstance(start, tuple) and len(start) == 3, 'invalid start'
    assert isinstance(end, tuple) and len(end) == 3, 'invalid end'
    assert datetime.date(start[0], start[1], start[2]) < datetime.date(end[0], end[1], end[2]), 'smaller'

    date_delta = datetime.timedelta(days=1)
    start_date = datetime.date(start[0], start[1], start[2])
    end_date = datetime.date(end[0], end[1], end[2])

    current_date = start_date
    output = [[[] for i in range(70)] for j in range(70)]
    days = []

    #rint(len(extended_attr_list))

    while current_date <= end_date:
        year, month, day = int(current_date.year), int(current_date.month), int(current_date.day)
        date_str = f'{month}/{day}/{year}'
        days.append((year, month, day))

        ############### real code
        for i in range(64):
            for j in range(30):
                state=states_list[i]
                option=extended_attr_list[j]
                item = data[year, month, day][state][option]
                if math.isnan(item) is False:
                    output[i][j].append(item)
                else:
                    #print()
                    if len(output[i][j])==0:
                        output[i][j].append(0)
                    else:
                        output[i][j].append(output[i][j][len(output[i][j])-1])
        ###############
        current_date += date_delta

    DATA=[]
    days = list(range(1, 8))#
    current_date = start_date
    for d in range(7):
        current_date += date_delta
        df = pd.DataFrame(index=new_extended_attr_list)
        for i in range(64):
            x = []
            x.append(states_list[i])
            #print(states_list[i])
            if states_list[i] in new_states_abbr_dict:
                x.append(new_states_abbr_dict[states_list[i]])
            else:
                x.append(' ')
            x.append(current_date)
            for j in range(30):
                N = len(days)
                sumx = sum(days)
                sumy = sum(output[i][j])
                sumx2 = sum([k ** 2 for k in days])
                sumxy = sum([days[k] * output[i][j][k] for k in range(N)])

                A = np.mat([[N, sumx], [sumx, sumx2]])
                B = np.array([sumy, sumxy])
                a, b = np.linalg.solve(A, B)
                pre=a+b*(8+d)
                pre=float(int(pre))
                x.append(pre)
                #


            data = {states_list[i]: x}
            df1 = pd.DataFrame(data, index=new_extended_attr_list)
            df=pd.concat([df, df1], axis=1, sort=False)
        DATA.append(df)
    #print(DATA[0])
    return DATA # DATA is a list with 7 elements, each element is a DataFrame of one of the nect 7 days predict

if __name__ == '__main__':
    import JHU_spider as dt

    #start = datetime.date(2021, 12, 15)
    #today = datetime.date.today()
    start = datetime.date(2021, 1, 1)
    today = datetime.date(2021, 1, 7)

    start_tuple = (start.year, start.month, start.day)
    today_tuple = (today.year, today.month, today.day)

    data = dt.MultipleData(start=start_tuple, end=today_tuple)

    predict1(data=data)