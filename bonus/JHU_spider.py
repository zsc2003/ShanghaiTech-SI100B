from bs4 import BeautifulSoup
import requests
import datetime
import csv
import pandas as pd
from functools import wraps
import math


total_time = datetime.timedelta()

start = datetime.date(2020, 4, 12)
today = datetime.date.today()
start_tuple = (start.year, start.month, start.day)
today_tuple = (today.year, today.month, today.day)

vac_df, test_df = None, None


def timer_decorator(func):
    """
    An NB decorator.

    :param func: function
    :return:ret
    """

    @wraps(func)
    def wrapped_function(*args, **kwargs):
        global total_time
        if func.__name__ == 'initialize_vac_and_tests_data':
            print('Initializing...')

        x = datetime.datetime.now()
        ret = func(*args, **kwargs)
        y = datetime.datetime.now()
        print(func.__name__, y - x)

        if func.__name__ == 'request_url':
            total_time += (y - x)
        if func.__name__ == 'MultipleData':
            print('            ', y - x - total_time)
            print('            ', total_time)
        return ret

    return wrapped_function


@timer_decorator
def initialize_vac_and_tests_data():
    global vac_df, test_df
    # vaccinated
    try:
        vac_df = pd.read_csv('https://raw.githubusercontent.com/govex/COVID-19/master/'
                             'data_tables/vaccine_data/us_data/time_series/people_vaccinated_us_timeline.csv')
        if vac_df is not None:
            print('vaccinated data successfully crawled')
            vac_df.to_csv('people_vaccinated_us_timeline.csv')
            print('latest vaccinated data saved')
        else:
            raise ZeroDivisionError
    except:
        print('Error detected, searching local datasets.')
        vac_df = pd.read_csv('people_vaccinated_us_timeline.csv')

    # tested
    try:
        test_df = pd.read_csv('https://raw.githubusercontent.com/govex/COVID-19/'
                              'master/data_tables/testing_data/time_series_covid19_US.csv')
        if test_df is not None:
            print('tested data successfully crawled')
            test_df.to_csv('time_series_covid19_US_legacy.csv')
            print('latest tested data saved')
        else:
            raise ZeroDivisionError
    except:
        print('Error detected, searching local datasets.')
        test_df = pd.read_csv('time_series_covid19_US_legacy.csv')


initialize_vac_and_tests_data()


class DateIterator(object):

    def __init__(self, start: tuple, end: tuple, step=1, truncate=False):
        assert isinstance(start, tuple) and len(start) == 3, 'invalid start'
        assert isinstance(end, tuple) and len(end) == 3, 'invalid end'
        assert datetime.date(start[0], start[1], start[2]) < datetime.date(end[0], end[1], end[2]),\
            'End must be later than start.'
        assert isinstance(step, int) and step > 0, 'step must be int and positive'
        assert isinstance(truncate, bool), 'invalid truncate'

        self.date_delta = datetime.timedelta(days=1)
        self.start_date = datetime.date(start[0], start[1], start[2])
        self.end_date = datetime.date(end[0], end[1], end[2])
        self.step = step

        if truncate:
            self.start_date = datetime.date(2021, 12, 16)

    def __iter__(self):
        self.current_date = self.start_date
        return self

    def __next__(self):
        while self.current_date <= self.end_date:
            year, month, day = int(self.current_date.year), int(self.current_date.month), int(self.current_date.day)
            self.current_date += (self.date_delta * self.step)
            return year, month, day
        else:
            raise StopIteration


@timer_decorator
class MultipleData(object):

    def __init__(self, start=None, end=None):
        assert isinstance(start, tuple) and len(start) == 3, 'invalid start'
        assert isinstance(end, tuple) and len(end) == 3, 'invalid end'
        assert datetime.date(start[0], start[1], start[2]) <= datetime.date(end[0], end[1], end[2]),\
            'End must be later than start.'

        self.update_time = None
        self.source = None
        self.start = start
        self.end = end
        self.__storage_series = {}    # in fact a pd.Series

        self.__set_storage_series(start=self.start, end=self.end)
        self.__concat_new_to_daily_data(start=self.start, end=self.end)

    def __getitem__(self, date):
        assert isinstance(date, tuple), 'invalid date'
        if isinstance(date[0], int):
            return self.__storage_series[date]
        elif isinstance(date[0], str):
            return self.__storage_series[(int(date[0]), int(date[1]), int(date[2]))]
        else:
            raise ZeroDivisionError  # a randomly chosen error

    def __setitem__(self, key, value):
        assert isinstance(key, tuple) and len(key) == 3, 'invalid date'
        self.__storage_series[key] = value
        return

    def __repr__(self):
        return f'MultipleData start={self.start} end={self.end}'

    def __set_storage_series(self, start=None, end=None):
        if start is None:
            start = self.start
        if end is None:
            end = self.end

        for year, month, day in iter(DateIterator(start=start, end=end)):
            current_date = datetime.date(year, month, day)
            if current_date <= datetime.date(2021, 12, 15):
                self[(year, month, day)] = DailyData(date=(year, month, day), mode='read')  # local
            else:
                self[(year, month, day)] = DailyData(date=(year, month, day), mode='crawl')  # crawl
        return

    def __concat_new_to_daily_data(self, start=None, end=None):
        if start is None:
            start = self.start
        if end is None:
            end = self.end

        date_delta = datetime.timedelta(days=1)
        real_start = datetime.date(start[0], start[1], start[2]) + date_delta
        start = (real_start.year, real_start.month, real_start.day)

        for year1, month1, day1 in iter(DateIterator(start=start, end=end, step=1, truncate=True)):
            current_date1 = datetime.date(year1, month1, day1)
            current_date2 = current_date1 - date_delta
            year2, month2, day2 = int(current_date2.year), int(current_date2.month), int(current_date2.day)

            df = self[year1, month1, day1].get_df().iloc[3: 18] - self[year2, month2, day2].get_df().iloc[3: 18]
            df.index = DailyData.delta_attr_list
            self[year1, month1, day1].update_df(pd.concat([self[year1, month1, day1].get_df(), df]))
        return

    # should only be called outside! and dont forget to change folder name
    @timer_decorator
    def save_to_csv(self, start=None, end=None, folder_name='updated_library'):
        if start is None:
            start = self.start
        if end is None:
            end = self.end

        for year, month, day in iter(DateIterator(start=self.start, end=self.end)):
            date_str = f'{year}-{month}-{day}'        # format: 2020-8-6
            self[year, month, day].save_data_to_csv(folder_name=folder_name)
            print(f'{date_str} saved!')
        return


class DailyData(object):
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

    attr_list = ('confirmed', 'deaths', 'incident_rate', 'total_test_results', 'people_hospitalized',
                 'case_fatality_ratio', 'testing_rate', 'hospitalization_rate', 'people_fully_vaccinated',
                 'people_partially_vaccinated', 'tests_combined_total', 'tests_viral_positive',
                 'tests_viral_negative', 'people_viral_total', 'people_viral_positive')

    delta_attr_list = tuple(map(lambda x: 'new_' + x, attr_list))

    extended_attr_list = tuple(list(attr_list) + list(delta_attr_list))

    states_abbr_dict = {'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AS': 'American_Samoa', 'AZ': 'Arizona',
                        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DC': 'District_of_Columbia',
                        'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'GU': 'Guam', 'HI': 'Hawaii', 'IA': 'Iowa',
                        'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas', 'KY': 'Kentucky',
                        'LA': 'Louisiana', 'MA': 'Massachusetts', 'MD': 'Maryland', 'ME': 'Maine', 'MI': 'Michigan',
                        'MN': 'Minnesota', 'MO': 'Missouri', 'MP': 'Northern_Mariana_Islands', 'MS': 'Mississippi',
                        'MT': 'Montana', 'NC': 'North_Carolina', 'ND': 'North_Dakota', 'NE': 'Nebraska',
                        'NH': 'New_Hampshire', 'NJ': 'New_Jersey', 'NM': 'New_Mexico', 'NV': 'Nevada',
                        'NY': 'New_York', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania',
                        'PR': 'Puerto_Rico', 'RI': 'Rhode_Island', 'SC': 'South_Carolina', 'SD': 'South_Dakota',
                        'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VA': 'Virginia', 'VI': 'Virgin_Islands',
                        'VT': 'Vermont', 'WA': 'Washington', 'WV': 'West_Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
                        'US': 'the_US'}

    states_abbr_reversed_dict = {pair[1]: pair[0] for pair in states_abbr_dict.items()}

    def __init__(self, date: tuple, mode: str):     # mode: 'crawl', 'read' or 'create'
        assert isinstance(date, tuple) and len(date) == 3, 'invalid date'
        self.date = date
        self.year, self.month, self.day = DailyData.set_date(date)    # they are strings!
        self.mode = mode

        print(self.date, end=' ')
        if self.mode == 'crawl':
            self.__storage_df = pd.DataFrame(columns=DailyData.states_list,
                                             index=('state_name', 'state_abbreviation', 'date') + DailyData.attr_list,
                                             dtype=object)
            self.__set_storage_df()
            self.__crawl_basic_daily_data()
            self.__parse_vac_and_tested_data()
            self.__calculate_us_sums()

        elif self.mode == 'read':
            self.__storage_df = self.__read_data_from_csv(folder_name='previous_data_library')
            self.__calculate_us_sums()    # maybe sums are missing in saved data?

        self.__storage_df.index.name = 'attributes'

    def __getitem__(self, state_name: str):
        assert isinstance(state_name, str), 'invalid state'
        if state_name.isupper():
            state_whole_name = DailyData.states_abbr_dict[state_name]
            return self.__storage_df[state_whole_name]   # returns a pd.Series
        else:
            return self.__storage_df[state_name]

    def __repr__(self):
        return repr(self.__storage_df)

    # get row, not recommended using
    def attr(self, attr):
        return self.__storage_df.loc[attr]

    # be careful when using this! when reading, use df
    def get_df(self):
        return self.__storage_df

    @property   # read only
    def df(self):    # read only
        return self.__storage_df

    def __set_storage_df(self):
        for state in DailyData.states_list:
            self.__storage_df[state]['state_name'] = state  # state name

            # state abbreviation
            if state in DailyData.states_abbr_reversed_dict:
                self.__storage_df[state]['state_abbreviation'] = DailyData.states_abbr_reversed_dict[state]
            else:
                self.__storage_df[state]['state_abbreviation'] = float('nan')

            self.__storage_df[state]['date'] = self.date  # date
            for attr in DailyData.attr_list:
                self.__storage_df[state][attr] = float('nan')  # all others
                self.__storage_df.iloc[3:] = self.__storage_df.iloc[3:].astype('float64')    # the first 3 are str
        return

    def __crawl_basic_daily_data(self):
        # request
        URL = f'https://github.com/CSSEGISandData/COVID-19/blob/master/' \
              f'csse_covid_19_data/csse_covid_19_daily_reports_us/{self.month}-{self.day}-{self.year}.csv'
        states_html = DailyData.request_url(URL)
        if states_html is None:  # date might be invalid, for example 2021.2.31, or dates upcoming
            return
        soup = BeautifulSoup(states_html, 'lxml')

        # parse
        states_info = soup.find_all(class_='js-file-line')
        del states_info[0]  # delete the first line because this is not a state

        temp_attr_list = ('confirmed', 'deaths', 'incident_rate', 'total_test_results', 'people_hospitalized',
                          'case_fatality_ratio', 'testing_rate', 'hospitalization_rate')
        parse_attr_index_list = (6, 7, 11, 12, 13, 14, 17, 18)

        for state_info in states_info:
            infos = state_info.find_all('td')
            state = str(infos[1].string).replace(' ', '_')
            for attr, index in zip(temp_attr_list, parse_attr_index_list):
                self[state][attr] = set_float(infos[index].string)    # save as float
        return

    def __parse_vac_and_tested_data(self):
        year, month, day = int(self.year), int(self.month), int(self.day)
        current_date = datetime.date(year, month, day)

        if current_date == today:
            try:
                date_df = pd.read_csv('https://raw.githubusercontent.com/govex/COVID-19/master/'
                                      'data_tables/vaccine_data/us_data/hourly/vaccine_people_vaccinated_US.csv')
                if date_df is not None:
                    print('today\'s vaccinated data successfully crawled')
                    date_df.to_csv(f'latest_vaccinated/{today}-vaccine_people_vaccinated_US.csv')
                    print('today\'s latest vaccinated data saved')
                else:
                    raise ZeroDivisionError
            except:
                date_str = f'{year}-{month}-{day}'  # format: 2021-06-30 which is different from excel but dunno why
                date_df = vac_df[vac_df.Date == date_str]
        else:
            date_str = f'{year}-{month}-{day}'  # format: 2021-06-30 which is different from excel but dunno why
            date_df = vac_df[vac_df.Date == date_str]

        date_df.index = range(date_df.shape[0])
        for index in range(len(date_df)):
            state = str(date_df['Province_State'][index]).replace(' ', '_')
            pfv = date_df['People_Fully_Vaccinated'][index]
            ppv = date_df['People_Partially_Vaccinated'][index]
            self[state]['people_fully_vaccinated'] = set_float(pfv)  # notice the lowercase form
            self[state]['people_partially_vaccinated'] = set_float(ppv)

        date_str2 = f'{month}/{day}/{year}'  # format: 7/19/2020
        date_df2 = test_df[test_df.date == date_str2]
        date_df2.index = range(date_df2.shape[0])

        for index in range(len(date_df2)):
            state = date_df2['state'][index]
            tbt = date_df2['tests_combined_total'][index]
            tvp = date_df2['tests_viral_positive'][index]
            tvn = date_df2['tests_viral_negative'][index]
            pvt = date_df2['people_viral_total'][index]
            pvp = date_df2['people_viral_positive'][index]
            self[state]['tests_combined_total'] = set_float(tbt)     # save as float
            self[state]['tests_viral_positive'] = set_float(tvp)
            self[state]['tests_viral_negative'] = set_float(tvn)
            self[state]['people_viral_total'] = set_float(pvt)
            self[state]['people_viral_positive'] = set_float(pvp)
        return

    def __calculate_us_sums(self):
        year, month, day = int(self.year), int(self.month), int(self.day)
        us = self.__storage_df.iloc[3:].sum(axis=1)   # us is a pd.Series
        us_head = pd.Series({'state_name': 'the_US', 'state_abbreviation': 'US', 'date': (year, month, day)})
        self.__storage_df['the_US'] = pd.concat([us_head, us])

    @timer_decorator
    def __read_data_from_csv(self, folder_name='previous_data_library'):
        year, month, day = self.year, self.month, self.day
        ret = pd.read_csv(f'{folder_name}/{year}-{month}-{day}.csv', index_col='attributes')
        ret.iloc[3:] = ret.iloc[3:].astype('float64')
        return ret

    # not recommended using alone
    # not private only to enable MultiData to call it
    def save_data_to_csv(self, folder_name='temp222'):
        year, month, day = self.year, self.month, self.day
        self.__storage_df.to_csv(path_or_buf=f'{folder_name}/{year}-{month}-{day}.csv')
        return

    # not recommended using
    # this exists only because get_df() does not support value assignment
    def update_df(self, input_df):
        self.__storage_df = input_df
        self.__storage_df.index.name = 'attributes'
        return

    @staticmethod
    def set_date(date):
        year, month, day = str(date[0]), str(date[1]), str(date[2])

        # normalize dates  e.g.     2021, 1, 9   --->   2021, 01, 09
        if len(month) == 1:
            month = '0' + month
        if len(day) == 1:
            day = '0' + day
        return year, month, day

    @staticmethod
    @timer_decorator
    def request_url(url):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.80 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.text
        except requests.RequestException:
            return None

    # abandoned
    @timer_decorator
    def __read_from_downloaded(self):
        temp_attr_list = ('confirmed', 'deaths', 'incident_rate', 'total_test_results', 'people_hospitalized',
                          'case_fatality_ratio', 'testing_rate', 'hospitalization_rate')
        temp_attr_list_cap = ('Confirmed', 'Deaths', 'Incident_Rate', 'Total_Test_Results', 'People_Hospitalized',
                              'Case_Fatality_Ratio', 'Testing_Rate', 'Hospitalization_Rate')
        year, month, day = self.year, self.month, self.day

        df = pd.read_csv(f'temptemptemp/{month}-{day}-{year}.csv')
        date = f'{year}-{month}-{day}'

        for index in range(df.shape[0]):
            state_infos = df.iloc[index]
            state = state_infos['Province_State'].replace(' ', '_')
            for attr, cap_attr in zip(temp_attr_list, temp_attr_list_cap):
                try:
                    self[state][attr] = state_infos[cap_attr]
                except KeyError:
                    pass

        print(f'{date} read!')
        return


def set_float(input):
    if input is None:
        return float('nan')
    else:
        return float(input)


if __name__ == '__main__':

    data = MultipleData(start=start_tuple, end=today_tuple)

    # data.save_to_csv(folder_name='updated_library')






    '''
    while True:
        date_raw = input('date: ').split(', ')
        date = (date_raw[0], date_raw[1], date_raw[2])
        print(data[date].get_df().to_string())
    '''
    while True:
        command = input('input any code: ')
        try:
            exec(command)
        finally:
            continue

    # JHU spider 1.5.0 更新日志 2021.12.18 16:43
    # 1 自动显示爬虫运行时间
    # 2 新增自定义起止日期，可以通过往MultipleData括号里传start/end两个参数实现，如上

    # JHU spider 2.0.0 更新日志 2021.12.18 22:53
    # 1 新增对pd.DataFrame的支持。data[2021, 10, 5].get_df() 此表达式返回与当日data相对应的dataframe，columns为州，index为各指标
    #   借由此特性可以在某日的数据上运用dataframe的函数
    # 2 支持通过以下print语句查看信息：
    #   print(data)
    #   print(data[2021, 10, 2])
    #   print(data[2021, 10, 2]['California'])
    #   注意：对于data而言，print出的形式并不是这个变量的返回值
    # 3 运行时间显示优化
    # 4 添加指标：完全免疫人数（people_fully_vaccinated），部分免疫人数（people_partially_vaccinated），
    #   总检测人次（tests_combined_total）, 检测阳性人次（tests_viral_positive， 检测阴性人次（tests_viral_negative），
    #   总检测人数（people_viral_total）, 阳性检测人数（people_viral_positive）
    #   数据来源不同，所以不同数据可能有矛盾。 e.g. 总检测人次不等于检测阳性人次加检测阴性人次
    # 5 支持用大写简称索引州  e.g. data[2021, 10, 2]['CA']          #  ( == data[2021, 10, 2]['California'] )

    # 优化方向：
    # 1 多线程，增加速度
    # 6 疫苗接种，检测，等各项指标的解释，可存放在其他文件里

    # JHU spider 5.0.0 2021.12.20
    # 最终版
