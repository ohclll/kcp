import os
import datetime
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from sklearn.preprocessing import MinMaxScaler

intersections_tollgate = ['A2', 'A3', 'B1', 'B3', 'C1', 'C3']


def parse_link(link_file):
    links = pd.read_csv(link_file)
    links_dict = {}
    for i, link in links.iterrows():
        links_dict[link.link_id] = dict(link)
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
        self.traj = None
        self.vol = None
        self.weather = None

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
            return [len(data),
                    np.mean(data),
                    np.median(data),
                    np.max(data),
                    np.min(data),
                    np.std(data),
                    np.max(data) - np.min(data)]
        else:
            return [0, ] + [np.nan] * 6

    def traj_features(self, time, interval=20):
        if self.traj is None:
            traj_file = os.path.join(self.data_dir, 'testing_phase1', 'trajectories(table 5)_test1.csv')
            test_traj = pd.read_csv(traj_file)
            traj_file = os.path.join(data_dir, 'training', 'trajectories(table 5)_training.csv')
            train_traj = pd.read_csv(traj_file)
            self.traj = pd.concat((train_traj, test_traj), axis=0, ignore_index=True)
        traj = self.traj
        time_start = time - datetime.timedelta(hours=2)
        traj = traj[(traj['starting_time'] >= str(time_start)) & (traj['starting_time'] < str(time))]

        time_slice = self._feature_time_slice(time, interval=interval)
        T = len(time_slice) - 1

        traj_ftr = defaultdict(list)
        for t in range(T):
            tr = traj[(traj['starting_time'] >= str(time_slice[t])) &
                      (traj['starting_time'] < str(time_slice[t + 1]))]
            traj_link = defaultdict(list)
            for i, r in tr.iterrows():
                link_time = [(int(x.split('#')[0]), float(x.split('#')[-1])) for x in r.travel_seq.split(';')]
                for k, v in link_time:
                    traj_link[k].append(v)
            for k in self.link.keys():
                ftr = self._times_features(traj_link.get(k, []))
                traj_ftr[k].append(ftr)
        # 插值补缺
        for k, v in traj_ftr.items():
            df = pd.DataFrame(data=v)
            df.interpolate(inplace=True)
            traj_ftr[k] = df.values.astype('float32')
        return traj_ftr

    def vol_features(self, time, interval=20):
        if self.vol is None:
            vol_file = os.path.join(self.data_dir, 'testing_phase1', 'volume(table 6)_test1.csv')
            test_vol = pd.read_csv(vol_file)
            vol_file = os.path.join(data_dir, 'training', 'volume(table 6)_training.csv')
            train_vol = pd.read_csv(vol_file)
            self.vol = pd.concat((train_vol, test_vol), axis=0, ignore_index=True)
        vol = self.vol
        time_start = time - datetime.timedelta(hours=2)
        vol = vol[(vol['time'] >= str(time_start)) & (vol['time'] < str(time))]
        time_slice = self._feature_time_slice(time, interval=interval)
        T = len(time_slice) - 1

        vol_cnt = np.zeros((T, 6, 2), dtype='float32')
        vol_mean = np.zeros((T, 6, 2), dtype='float32')

        multi_idx = pd.MultiIndex.from_product([[1, 2, 3], [0, 1], [0, 1]])
        for t in range(T):
            vl = vol[(vol['time'] >= str(time_slice[t])) & (vol['time'] < str(time_slice[t + 1]))]
            count_by_ect = vl.groupby(['tollgate_id', 'direction', 'has_etc']).count().iloc[:, 0].reindex(
                multi_idx).fillna(0).values.reshape((6, 2)).astype('float32')
            mean_by_mode = vl.groupby(['tollgate_id', 'direction', 'has_etc']).mean().loc[:, 'vehicle_model'].reindex(
                multi_idx).fillna(0).values.reshape((6, 2)).astype('float32')
            vol_cnt[t] = count_by_ect
            vol_mean[t] = mean_by_mode
        return vol_cnt, vol_mean

    def _preprocess_weather(self, weather):
        weather['time'] = weather['date'] + weather['hour'].apply(lambda x: datetime.timedelta(hours=x))
        weather.drop(['date', 'hour'], axis=1, inplace=True)
        weather[weather['wind_direction'] > 360] = 0
        weather[weather['wind_direction'] > 180] -= 360
        names = weather.columns.tolist()
        names.remove('time')
        weather[names] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(weather[names])
        return weather

    def wea_featuree(self, time, interval=20):
        if self.weather is None:
            wea_file = os.path.join(self.data_dir, 'testing_phase1', 'weather (table 7)_test1.csv')
            test_weather = pd.read_csv(wea_file, parse_dates=['date'])
            wea_file = os.path.join(data_dir, 'training', 'weather (table 7)_training.csv')
            train_weather = pd.read_csv(wea_file, parse_dates=['date'])
            weather = pd.concat((train_weather, test_weather), axis=0, ignore_index=True)
            weather = self._preprocess_weather(weather)
            self.weather = weather
        weather = self.weather
        weather = weather[(weather['time'] > time - datetime.timedelta(hours=1, minutes=59)) &
                          (weather['time'] <= time + datetime.timedelta(hours=4, minutes=1))]
        if weather.shape[0] == 0:
            return np.zeros((2, 2), dtype='float32')
        if weather.shape[0]!=2:
            print('')
        assert weather.shape[0] == 2, weather
        ftr = weather[['temperature', 'precipitation']].values
        return ftr


def gen_sample(traj, vol, weather, route, day, hours):
    t_slice = time_slice(day, hours)
    T = len(t_slice) - 1
    traj_data = []
    vol_cnt = np.zeros((T, 2, 3, 2), dtype='float32')
    vol_mean = np.zeros((T, 2, 3, 2), dtype='float32')
    multi_idx = pd.MultiIndex.from_product([[0, 1], [1, 2, 3], [0, 1]])
    for i in range(T):
        tj = traj[(traj['starting_time'] >= t_slice[i]) & (traj['starting_time'] < t_slice[i + 1])]
        d = defaultdict(list)
        for _, r in tj.iterrows():
            key = r.intersection_id + str(r.tollgate_id)
            time = [float(x.split('#')[-1]) for x in r.travel_seq.split(';')]
            if len(time) != len(route[key]):
                print(key, r.starting_time)
                continue
            time.append(r.travel_time)
            d[key].append(time)
        d = {k: np.array([np.mean(v, axis=0), np.std(v, axis=0)], dtype='float32') for k, v in d.items()}
        traj_data.append(d)

        vl = vol[(vol['time'] >= t_slice[i]) & (vol['time'] < t_slice[i + 1])]
        count_by_ect = vl.groupby(['direction', 'tollgate_id', 'has_etc']).count().iloc[:, 0].reindex(
            multi_idx).fillna(0).values.reshape((2, 3, 2)).astype('float32')
        mean_by_mode = vl.groupby(['direction', 'tollgate_id', 'has_etc']).mean().loc[:, 'vehicle_model'].reindex(
            multi_idx).fillna(0).values.reshape((2, 3, 2)).astype('float32')
        vol_cnt[i] = count_by_ect
        vol_mean[i] = mean_by_mode

    traj_tmp = {}
    for k in route.keys():
        d = np.zeros((T, 2, len(route[k]) + 1), dtype='float32')
        for i, t in enumerate(traj_data):
            if k in t:
                d[i] = t[k]
        traj_tmp[k] = d
    weather = weather[(weather['date'] == day) &
                      (weather['hour'] >= int(hours[0])) &
                      (weather['hour'] <= int(hours[0]) + 6)]
    wea_data = weather[['wind_speed', 'precipitation']].values
    return traj_tmp, vol_cnt, vol_mean, wea_data


def gen_target(traj, vol, route, day, hours):
    hours = ['{:0>2}'.format(int(h) + 2) for h in hours]
    t_slice = time_slice(day, hours)
    T = len(t_slice) - 1
    time_target = defaultdict(list)
    vol_target = np.zeros((T, 2, 3), dtype='float32')

    multi_idx = pd.MultiIndex.from_product([[0, 1], [1, 2, 3]])
    for i in range(T):
        tj = traj[(traj['starting_time'] >= t_slice[i]) & (traj['starting_time'] < t_slice[i + 1])]
        d = defaultdict(list)
        for _, r in tj.iterrows():
            key = r.intersection_id + str(r.tollgate_id)
            d[key].append(r.travel_time)
        d = {k: sum(v) / len(v) for k, v in d.items()}
        for k in route.keys():
            if k in d:
                time_target[k].append(d[k])
            else:
                time_target[k].append(0)

        vl = vol[(vol['time'] >= t_slice[i]) & (vol['time'] < t_slice[i + 1])]
        vol_target[i] = vl.groupby(['direction', 'tollgate_id']).count().iloc[:, 0].reindex(
            multi_idx).fillna(0).values.reshape((2, 3)).astype('float32')
    return time_target, vol_target


if __name__ == '__main__':
    data_dir = 'dataSets'
    data = Data(data_dir)
    dates = []
    date = datetime.date(2016, 10, 15)
    while date <= datetime.date(2016, 10, 17):
        dates.append(date)
        date += datetime.timedelta(days=1)
    ts = [(8, 0), (17, 0)]
    for date in dates:
        print(date)
        for t in ts:
            time = datetime.datetime(date.year, date.month, date.day, t[0], t[1])
            traj_ftr = data.traj_features(time, 20)
            vol_ftr = data.vol_features(time, 20)
            wea_ftr = data.wea_featuree(time, 20)
