import os
import pickle
import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import chain
from sklearn.preprocessing import MinMaxScaler, Imputer

intersections_tollgate = ['A2', 'A3', 'B1', 'B3', 'C1', 'C3']


def parse_link(link_file):
    links = pd.read_csv(link_file)
    links_dict = {}
    for i, r in links.iterrows():
        links_dict[r.link_id] = dict(r)
    return links_dict


def parse_route(route_file):
    route_dict = {}
    route = pd.read_csv(route_file)
    for i, r in route.iterrows():
        route_dict[r.intersection_id + str(r.tollgate_id)] = [int(x) for x in r.link_seq.split(',')]
    return route_dict


class Data:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.link_file = os.path.join(data_dir, 'training', 'links (table 3).csv')
        self.route_file = os.path.join(data_dir, 'training', 'routes (table 4).csv')
        self.route = parse_route(self.route_file)
        self.link = parse_link(self.link_file)
        self._traj = None
        self._vol = None
        self._weather = None

    @property
    def traj(self):
        if self._traj is None:
            traj_file = os.path.join(self.data_dir, 'testing_phase1', 'trajectories(table 5)_test1.csv')
            test_traj = pd.read_csv(traj_file)
            traj_file = os.path.join(data_dir, 'training', 'trajectories(table 5)_training.csv')
            train_traj = pd.read_csv(traj_file)
            traj = pd.concat((train_traj, test_traj), axis=0, ignore_index=True)
            traj['end_time'] = traj['starting_time'].apply(
                lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) + traj['travel_time'].apply(
                lambda x: datetime.timedelta(seconds=x))
            traj['end_time'] = traj['end_time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            traj['route'] = traj['intersection_id'] + traj['tollgate_id'].apply(str)
            self._traj = traj
        return self._traj

    @property
    def vol(self):
        if self._vol is None:
            vol_file = os.path.join(self.data_dir, 'testing_phase1', 'volume(table 6)_test1.csv')
            test_vol = pd.read_csv(vol_file)
            vol_file = os.path.join(data_dir, 'training', 'volume(table 6)_training.csv')
            train_vol = pd.read_csv(vol_file)
            self._vol = pd.concat((train_vol, test_vol), axis=0, ignore_index=True)
        return self._vol

    @property
    def weather(self):
        if self._weather is None:
            wea_file = os.path.join(self.data_dir, 'testing_phase1', 'weather (table 7)_test1.csv')
            test_weather = pd.read_csv(wea_file, parse_dates=['date'])
            wea_file = os.path.join(data_dir, 'training', 'weather (table 7)_training.csv')
            train_weather = pd.read_csv(wea_file, parse_dates=['date'])
            weather = pd.concat((train_weather, test_weather), axis=0, ignore_index=True)
            weather = self._preprocess_weather(weather)
            self._weather = weather
        return self._weather

    def _preprocess_weather(self, weather):
        weather['time'] = weather['date'] + weather['hour'].apply(lambda x: datetime.timedelta(hours=x))
        weather.drop(['date', 'hour'], axis=1, inplace=True)
        weather.loc[weather['wind_direction'] > 360, 'wind_direction'] = 0
        weather.loc[weather['wind_direction'] > 180, 'wind_direction'] -= 360
        weather.set_index('time', inplace=True)
        t_index = pd.date_range(weather.index.min(), weather.index.max(), freq='60S')
        weather = weather.reindex(t_index).interpolate(method='time')
        names = weather.columns.tolist()
        # names.remove('time')
        weather[names] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(weather[names])
        weather = weather[['pressure', 'wind_speed', 'temperature', 'precipitation']]
        return weather

    def _feature_time_slice(self, time, interval=20):
        time_start = time - datetime.timedelta(hours=2)
        time_slice = [str(time)]
        cur_time = time - datetime.timedelta(minutes=interval)
        while cur_time >= time_start:
            time_slice.append(str(cur_time))
            cur_time -= datetime.timedelta(minutes=interval)
        # time_slice.append(str(time_start))
        time_slice.reverse()
        return time_slice

    def _target_time_slice(self, time):
        time_slice = []
        for i in range(7):
            t = time + datetime.timedelta(minutes=20 * i)
            time_slice.append(str(t))
        return time_slice

    def _times_features(self, data):
        """
        提取某段时间内某个link的特征
        :param data: 该时间内，经过某个link的所有车辆的时间
        :return:
        """
        if data:
            if len(data) == 1:
                return [np.mean(data),
                        np.median(data),
                        np.max(data),
                        np.min(data),
                        np.percentile(data, 25),
                        np.percentile(data, 75),
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan]
            else:
                return [np.mean(data),
                        np.median(data),
                        np.max(data),
                        np.min(data),
                        np.percentile(data, 25),
                        np.percentile(data, 75),
                        np.std(data),
                        np.mean(data) - np.std(data),
                        np.mean(data) + np.std(data),
                        np.max(data) - np.min(data)]
        else:
            return [np.nan] * 10

    def traj_features(self, time, interval=20):
        traj = self.traj
        time_start = time - datetime.timedelta(hours=2)
        traj = traj[(traj['starting_time'] >= str(time_start)) & (traj['starting_time'] < str(time))]

        time_slice = self._feature_time_slice(time, interval=interval)
        T = len(time_slice) - 1

        link_ftr = defaultdict(list)
        route_ftr = defaultdict(list)
        for t in range(T):
            tr = traj[(traj['starting_time'] >= str(time_slice[t])) &
                      (traj['starting_time'] < str(time_slice[t + 1]))]
            trout = traj[(traj['end_time'] >= str(time_slice[t])) &
                         (traj['end_time'] < str(time_slice[t + 1]))]
            num_in = dict(tr.groupby('route').count().iloc[:, 0])
            num_out = dict(trout.groupby('route').count().iloc[:, 0])

            traj_link = defaultdict(list)
            traj_route = defaultdict(list)

            for i, r in tr.iterrows():
                it = r.intersection_id + str(r.tollgate_id)
                traj_route[it].append(r.travel_time)

                link_time = [(int(x.split('#')[0]), float(x.split('#')[-1])) for x in r.travel_seq.split(';')]
                for k, v in link_time:
                    traj_link[k].append(v)

            for k in self.route:
                ftr = self._times_features(traj_route.get(k, []))
                num = [num_in.get(k, 0), num_out.get(k, 0), num_in.get(k, 0) - num_out.get(k, 0)]
                route_ftr[k].append(ftr + num)

            for k in self.link:
                ftr = self._times_features(traj_link.get(k, []))
                link_ftr[k].append(ftr)

        # 插值补缺
        for k, v in route_ftr.items():
            df = pd.DataFrame(data=v)
            df.interpolate(limit_direction='both', limit=T, inplace=True)
            route_ftr[k] = df.values.astype('float32')
        for k, v in link_ftr.items():
            df = pd.DataFrame(data=v)
            df.interpolate(limit_direction='both', limit=T, inplace=True)
            link_ftr[k] = df.values.astype('float32')
        return link_ftr, route_ftr

    def vol_features(self, time, interval=20):
        vol = self.vol
        time_start = time - datetime.timedelta(hours=2)
        vol = vol[(vol['time'] >= str(time_start)) & (vol['time'] < str(time)) & (vol['direction'] == 0)]
        time_slice = self._feature_time_slice(time, interval=interval)
        T = len(time_slice) - 1

        vol_ftr = defaultdict(list)
        multi_idx = pd.MultiIndex.from_product([[1, 2, 3], [0, 1]])
        for t in range(T):
            vl = vol[(vol['time'] >= str(time_slice[t])) & (vol['time'] < str(time_slice[t + 1]))]
            ftr = vl.groupby(['tollgate_id', 'has_etc']).count().iloc[:, 0].reindex(multi_idx) \
                .fillna(0).values.reshape((3, 2)).astype('float32')
            for i in range(3):
                vol_ftr[i + 1].append(ftr[i])
        for k in vol_ftr:
            vol_ftr[k] = np.array(vol_ftr[k])
        return vol_ftr

    def wea_features(self, time, interval=20):
        weather = self.weather
        weather['time'] = weather.index
        weather = weather[(weather['time'] >= time) & (weather['time'] < time + datetime.timedelta(hours=2))].copy()
        weather['slice'] = (weather['time'] - weather['time'].min()).apply(
            lambda x: x.total_seconds() // (interval * 60))
        ftr = weather.groupby('slice').mean().values
        return ftr

    def get_target(self, time, interval=20):
        traj = self.traj
        vol = self.vol

        multi_idx = pd.MultiIndex.from_product([[1, 2, 3], [0, 1]])
        time_slice = self._target_time_slice(time)
        T = len(time_slice) - 1

        target_vol = np.zeros((T, 6), dtype='float32')
        target_link = defaultdict(list)
        target_route = defaultdict(list)

        for t in range(T):
            tr = traj[(traj['starting_time'] >= str(time_slice[t])) &
                      (traj['starting_time'] < str(time_slice[t + 1]))]
            traj_link = defaultdict(list)
            traj_route = defaultdict(list)
            for i, r in tr.iterrows():
                link_time = [(int(x.split('#')[0]), float(x.split('#')[-1])) for x in r.travel_seq.split(';')]
                it = r.intersection_id + str(r.tollgate_id)
                traj_route[it].append(r.travel_time)
                for k, v in link_time:
                    traj_link[k].append(v)

            for k in self.link:
                the_link_time = np.mean(traj_link.get(k, []))
                target_link[k].append(the_link_time)
            for k in self.route:
                the_route_time = np.mean(traj_route.get(k, [0]))
                target_route[k].append(the_route_time)

            vl = vol[(vol['time'] >= str(time_slice[t])) & (vol['time'] < str(time_slice[t + 1]))]
            target_vol[t] = vl.groupby(['tollgate_id', 'direction']).count().iloc[:, 0].reindex(
                multi_idx).fillna(0).values.reshape((6,)).astype('float32')

        for k in self.link:
            if np.isnan(target_link[k]).sum() > 0:
                s = pd.Series(target_link[k])
                target_link[k] = s.interpolate().tolist()
        return target_link, target_route, target_vol


def gen_dates(start_date, end_date, holidays):
    date = start_date
    dates = []
    while date <= end_date:
        if date not in holidays:
            dates.append(date)
        date += datetime.timedelta(days=1)
    return dates


def append_(src, dst):
    for k in dst:
        src[k].append(dst[k])
    return src


def imputer3D(data):
    data = np.array(data)
    T = data.shape[1]
    for i in range(T):
        data[:, i] = Imputer(axis=0).fit_transform(data[:, i])
    return data


def extract_features(dates, times, interval=20):
    times = list(times)
    route_x = defaultdict(list)
    link_x = defaultdict(list)
    # vol_x = defaultdict(list)
    wea_x = []
    time_x = []
    tms = []
    for h, m in times:
        route_ftrs = defaultdict(list)
        link_ftrs = defaultdict(list)
        wea_ftrs = []
        time_ftrs = []
        # vol_ftrs = defaultdict(list)
        for date in dates:
            time = datetime.datetime(date.year, date.month, date.day, h, m)
            if h < 12:
                time_ftr = [1, ]
            else:
                time_ftr = [0, ]
            tmp = [0, 0, 0, 0, 0, 0, 0]
            tmp[time.weekday()] = 1
            time_ftr.extend(tmp)
            tms.append(str(time))
            print('\r{}'.format(str(time)), end='')
            link_ftr, route_ftr = data.traj_features(time, interval=interval)
            wea_ftr = data.wea_features(time, interval=20)
            wea_ftrs.append(wea_ftr)
            time_ftrs.append(time_ftr)
            # vol_ftr = data.vol_features(time, interval=interval)
            # vol_ftrs = append_(vol_ftrs, vol_ftr)
            route_ftrs = append_(route_ftrs, route_ftr)
            link_ftrs = append_(link_ftrs, link_ftr)

        link_ftrs = {k: imputer3D(v) for k, v in link_ftrs.items()}
        route_ftrs = {k: imputer3D(v) for k, v in route_ftrs.items()}

        time_x.append(np.array(time_ftrs, 'float32'))
        wea_x.append(np.array(wea_ftrs, 'float32'))

        # vol_x = append_(vol_x, vol_ftrs)
        route_x = append_(route_x, route_ftrs)
        link_x = append_(link_x, link_ftrs)

    wea_x = np.concatenate(wea_x)
    time_x = np.concatenate(time_x)

    # vol_x = {k: np.concatenate(v).astype('float32') for k, v in vol_x.items()}
    route_x = {k: np.concatenate(v).astype('float32') for k, v in route_x.items()}
    link_x = {k: np.concatenate(v).astype('float32') for k, v in link_x.items()}

    return route_x, link_x, wea_x, time_x, tms

def get_weight(time):
    # time = datetime.datetime(date.year, date.month, date.day, h, m)
    t=datetime.datetime(time.year,time.month,time.day,8)
    if time.hour>12:
        time=time-datetime.timedelta(hours=9)
    delta=abs((time-t).total_seconds())/60
    w=1-delta*(0.7/60)
    return w


def extract_target(dates, times, interval=20):
    times = list(times)
    route_y = defaultdict(list)
    link_y = defaultdict(list)
    weights=[]
    for h, m in times:
        route_targets = defaultdict(list)
        link_targets = defaultdict(list)
        for date in dates:
            time = datetime.datetime(date.year, date.month, date.day, h, m)
            weights.append(get_weight(time))
            print('\r{}'.format(str(time)), end='')
            target_link, target_route, target_vol = data.get_target(time, interval=interval)
            route_targets = append_(route_targets, target_route)
            link_targets = append_(link_targets, target_link)

        link_targets = {k: Imputer().fit_transform(v) for k, v in link_targets.items()}
        route_targets = {k: np.nan_to_num(v) for k, v in route_targets.items()}

        route_y = append_(route_y, route_targets)
        link_y = append_(link_y, link_targets)

    route_y = {k: np.concatenate(v).astype('float32') for k, v in route_y.items()}
    link_y = {k: np.concatenate(v).astype('float32') for k, v in link_y.items()}
    weights=np.array(weights,dtype='float32')
    return route_y, link_y, weights


if __name__ == '__main__':
    data_dir = 'dataSets'
    data = Data(data_dir)
    holidays = [datetime.date(2016, 9, 15), datetime.date(2016, 9, 16), datetime.date(2016, 9, 17)] + \
               [datetime.date(2016, 10, i + 1) for i in range(7)]
    intervals = [120, 60, 8, 10, 12, 15, 17, 24, 20, 30, 40]
    for interval in intervals:
        # times = chain(((i, 0) for i in range(9, 17)),
        #               ((i, 3) for i in range(9, 17)),
        #               ((i, 6) for i in range(9, 17)),
        #               ((i, 9) for i in range(9, 17)),
        #               ((i, 12) for i in range(9, 17)),
        #               ((i, 15) for i in range(9, 16)),
        #               ((i, 18) for i in range(9, 16)),
        #               ((i, 21) for i in range(9, 16)),
        #               ((i, 24) for i in range(9, 16)),
        #               ((i, 27) for i in range(9, 16)),
        #
        #               ((i, 30) for i in range(9, 16)),
        #               ((i, 33) for i in range(9, 16)),
        #               ((i, 36) for i in range(9, 16)),
        #               ((i, 39) for i in range(9, 16)),
        #               ((i, 42) for i in range(9, 16)),
        #               ((i, 45) for i in range(8, 16)),
        #               ((i, 48) for i in range(8, 16)),
        #               ((i, 51) for i in range(8, 16)),
        #               ((i, 54) for i in range(8, 16)),
        #               ((i, 57) for i in range(8, 16)), )
        # times = list(times)
        # dates = gen_dates(datetime.date(2016, 7, 19), datetime.date(2016, 10, 10),holidays)
        # print(len(dates) * len(times))
        # route_ftrs, link_ftrs, vol_ftrs, wea_ftrs, tms = extract_features(dates, times, interval=interval)
        # if interval==20:
        #     route_targets, link_targets = extract_target(dates, times, interval=20)
        # else:
        #     route_targets, link_targets= 0,0
        # with open(os.path.join(data_dir, 'data/pretrain_ftrs_{}.pkl'.format(interval)), 'wb') as f:
        #     pickle.dump([route_ftrs, link_ftrs, vol_ftrs, wea_ftrs, tms, route_targets, link_targets], f)

        # times = [(7, 50), (7, 52), (7, 54), (7, 56), (7, 58), (8, 0),
        #          (8, 2), (8, 4), (8, 6), (8, 8), (8, 10),
        #          (16, 50), (16, 52), (16, 54), (16, 56), (16, 58), (17, 0),
        #          (17, 2), (17, 4), (17, 6), (17, 8), (17, 10)]
        # dates = gen_dates(datetime.date(2016, 7, 19), datetime.date(2016, 10, 17),holidays)
        # print(len(dates) * len(times))
        # route_ftrs, link_ftrs, vol_ftrs, wea_ftrs, tms = extract_features(dates, times, interval=interval)
        # if interval == 20:
        #     route_targets, link_targets = extract_target(dates, times, interval=20)
        # else:
        #     route_targets, link_targets = 0, 0
        # with open(os.path.join(data_dir, 'trainval_ftrs_{}.pkl'.format(interval)), 'wb') as f:
        #     pickle.dump([route_ftrs, link_ftrs, vol_ftrs, wea_ftrs, tms, route_targets, link_targets], f)

        times = list([(7, i) for i in range(60)]) + \
                list([(8, i) for i in range(60)]) + \
                list([(16, i) for i in range(60)]) + \
                list([(17, i) for i in range(60)])
        # times = list([(7, i) for i in range(55,60)]) + \
        #         list([(8, i) for i in range(6)]) + \
        #         list([(16, i) for i in range(55,60)]) + \
        #         list([(17, i) for i in range(6)])
        dates = gen_dates(datetime.date(2016, 7, 19), datetime.date(2016, 10, 10), holidays)
        print(len(dates) * len(times))
        route_ftrs, link_ftrs, wea_ftrs, times_ftrs, tms = extract_features(dates, times, interval=interval)
        if interval == 20:
            route_targets, link_targets, weights = extract_target(dates, times, interval=20)
        else:
            route_targets, weights = 0, 0
        with open(os.path.join(data_dir, 'data/train_ftrs_{}.pkl'.format(interval)), 'wb') as f:
            pickle.dump([route_ftrs, link_ftrs, wea_ftrs, times_ftrs, tms, route_targets, weights], f)

        times = list([(7, i) for i in range(55,60)]) + \
                list([(8, i) for i in range(6)]) + \
                list([(16, i) for i in range(55,60)]) + \
                list([(17, i) for i in range(6)])
        dates = gen_dates(datetime.date(2016, 10, 11), datetime.date(2016, 10, 17),holidays)
        print(len(dates) * len(times))
        route_ftrs, link_ftrs, wea_ftrs, times_ftrs, tms = extract_features(dates, times, interval=interval)
        if interval == 20:
            route_targets, link_targets, weights = extract_target(dates, times, interval=20)
        else:
            route_targets, weights = 0, 0
        with open(os.path.join(data_dir, 'datanew/val_ftrs_{}.pkl'.format(interval)), 'wb') as f:
            pickle.dump([route_ftrs, link_ftrs, wea_ftrs, times_ftrs, tms, route_targets, weights], f)

        times = [(8, 0), (17, 0)]
        dates = gen_dates(datetime.date(2016, 10, 11), datetime.date(2016, 10, 17), holidays)
        print(len(dates) * len(times))
        route_ftrs, link_ftrs, wea_ftrs, times_ftrs, tms = extract_features(dates, times, interval=interval)
        if interval == 20:
            route_targets, link_targets, weights = extract_target(dates, times, interval=20)
        else:
            route_targets, weights = 0, 0
        with open(os.path.join(data_dir, 'datanew/sval_ftrs_{}.pkl'.format(interval)), 'wb') as f:
            pickle.dump([route_ftrs, link_ftrs, wea_ftrs, times_ftrs, tms, route_targets, weights], f)

        times = [(8, 0), (17, 0)]
        dates = gen_dates(datetime.date(2016, 10, 18), datetime.date(2016, 10, 24),holidays)
        print(len(dates) * len(times))
        route_ftrs, link_ftrs, wea_ftrs, times_ftrs, tms = extract_features(dates, times, interval=interval)
        with open(os.path.join(data_dir, 'data/test_ftrs_{}.pkl'.format(interval)), 'wb') as f:
            pickle.dump([route_ftrs, link_ftrs, wea_ftrs, times_ftrs, tms], f)
