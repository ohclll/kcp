import os
import pickle
import datetime
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from sklearn.preprocessing import MinMaxScaler

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
            self._traj = pd.concat((train_traj, test_traj), axis=0, ignore_index=True)
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
        weather[weather['wind_direction'] > 360] = 0
        weather[weather['wind_direction'] > 180] -= 360
        names = weather.columns.tolist()
        names.remove('time')
        weather[names] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(weather[names])
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
                return [len(data),
                        np.mean(data),
                        np.median(data),
                        np.max(data),
                        np.min(data),
                        np.nan,
                        np.nan]
            else:
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

            traj_link = defaultdict(list)
            traj_route = defaultdict(list)

            for i, r in tr.iterrows():
                it = r.intersection_id + str(r.tollgate_id)
                traj_route[it].append(r.travel_time)

                link_time = [(int(x.split('#')[0]), float(x.split('#')[-1])) for x in r.travel_seq.split(';')]
                for k, v in link_time:
                    traj_link[k].append(v)

            for k in self.route:
                route_ftr[k].append(np.mean(traj_route.get(k, [])))

            for k in self.link:
                ftr = self._times_features(traj_link.get(k, []))
                link_ftr[k].append(ftr)

        # 插值补缺
        for k, v in route_ftr.items():
            if np.isnan(v).sum() > 0:
                s = pd.Series(v)
                route_ftr[k] = s.interpolate().tolist()
        for k, v in link_ftr.items():
            df = pd.DataFrame(data=v)
            df.interpolate(inplace=True)
            link_ftr[k] = df.values.astype('float32')
        return link_ftr, route_ftr

    def vol_features(self, time, interval=20):
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

    def wea_features(self, time, interval=20):
        weather = self.weather
        weather = weather[(weather['time'] > time - datetime.timedelta(hours=1, minutes=59)) &
                          (weather['time'] <= time + datetime.timedelta(hours=4, minutes=1))]
        if weather.shape[0] == 0:
            return np.zeros((2, 2), dtype='float32')
        if weather.shape[0] != 2:
            print('')
        assert weather.shape[0] == 2, weather
        ftr = weather[['temperature', 'precipitation']].values
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


def gen_times(start_date, end_date, time=((8, 0), (17, 0))):
    date = start_date
    times = []
    while date <= end_date:
        for h, m in time:
            times.append(datetime.datetime(date.year, date.month, date.day, h, m))
        date += datetime.timedelta(days=1)
    return times


def append_(src, dst):
    for k in dst:
        src[k].append(dst[k])
    return src


def extract_features(times, interval=20):
    route_targets = defaultdict(list)
    link_targets = defaultdict(list)

    route_ftrs = defaultdict(list)
    link_ftrs = defaultdict(list)

    for i, time in enumerate(times):
        print('\r{}/{}'.format(i, len(times)), end='')
        target_link, target_route, target_vol = data.get_target(time, interval=interval)
        link_ftr, route_ftr = data.traj_features(time, interval=interval)
        route_targets = append_(route_targets, target_route)
        link_targets = append_(link_targets, target_link)
        route_ftrs = append_(route_ftrs, route_ftr)
        link_ftrs = append_(link_ftrs, link_ftr)
    return route_ftrs, link_ftrs, route_targets, link_targets


if __name__ == '__main__':
    data_dir = 'dataSets'
    data = Data(data_dir)
    interval = 20

    time=list(((i,0) for i in range(9,17)))+list(((i,30) for i in range(8,17)))
    times = gen_times(datetime.date(2016, 7, 19), datetime.date(2016, 10, 10),time=time)
    route_ftrs, link_ftrs, route_targets, link_targets = extract_features(times, interval=interval)
    with open(os.path.join(data_dir, 'pretrain_ftrs.pkl'), 'wb') as f:
        pickle.dump([route_ftrs, link_ftrs, route_targets, link_targets], f)

    # times = gen_times(datetime.date(2016, 7, 19), datetime.date(2016, 10, 10))
    # route_ftrs, link_ftrs, route_targets, link_targets = extract_features(times, interval=interval)
    # with open(os.path.join(data_dir,'train_ftrs.pkl'),'wb') as f:
    #     pickle.dump([route_ftrs, link_ftrs, route_targets, link_targets],f)
    #
    # times = gen_times(datetime.date(2016, 10, 11), datetime.date(2016, 10, 17))
    # route_ftrs, link_ftrs, route_targets, link_targets = extract_features(times, interval=interval)
    # with open(os.path.join(data_dir, 'val_ftrs.pkl'), 'wb') as f:
    #     pickle.dump([route_ftrs, link_ftrs, route_targets, link_targets], f)
