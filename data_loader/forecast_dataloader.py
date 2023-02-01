import torch.utils.data as torch_data
import numpy as np
import torch
import pandas as pd


def normalized(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        # print(norm_statistic['max'])
        scale = np.array(norm_statistic['max']) - np.array(norm_statistic['min'])
        data = (data - norm_statistic['min']) / scale
        # data = np.clip(data, 0.0, 1.0)
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = (data - mean) / std
        norm_statistic['std'] = std
    return data, norm_statistic


def de_normalized(data, normalize_method, norm_statistic):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = np.array(norm_statistic['max']) - np.array(norm_statistic['min'])
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data


class ForecastDataset(torch_data.Dataset):
    def __init__(self, df, window_size, horizon, name, step_type, normalize_method=None, norm_statistic=None, interval=1, is_train=True):
        masked_list = ['pems07', 'pems04', 'pems08', 'pems03']
        unmasked_list = ['']
        assert name in masked_list or name in unmasked_list
        self.is_mask = name in masked_list
        self.is_train = is_train
        self.step_type = step_type
        self.window_size = window_size
        self.interval = interval
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.norm_statistic = norm_statistic
        df = pd.DataFrame(df)
        if self.is_mask:
            # self.interpolated_data_unnorm = df.values
            # self.interpolated_data, _ = normalized(self.interpolated_data_unnorm, normalize_method, norm_statistic)
            interpolated_data = df.replace(0, np.nan)
            interpolated_data = interpolated_data.interpolate(method='values', axis=0)
            interpolated_data = interpolated_data.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df))#.interpolate(method='linear', axis=0)
            self.interpolated_data = interpolated_data.values
            self.interpolated_data_unnorm = self.interpolated_data
            
            if normalize_method:
                self.interpolated_data, _ = normalized(self.interpolated_data, normalize_method, norm_statistic)

        # df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data = df.values
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()
        self.data_unnorm = self.data.copy()
        
        if normalize_method:
            self.data, _ = normalized(self.data, normalize_method, norm_statistic)

    def __getitem__(self, index):
        hi = self.x_end_idx[index]
        lo = hi - self.window_size

        if self.is_mask:
            
            train_data = self.interpolated_data[lo: hi]
            if self.is_train:
                target_data = self.interpolated_data[hi:hi + self.horizon]
            else:
                target_data = self.data[hi:hi + self.horizon]
            x_unnorm = torch.from_numpy(self.interpolated_data_unnorm[lo: hi]).float()
            if self.is_train:
                y_unnorm = torch.from_numpy(self.interpolated_data_unnorm[hi:hi + self.horizon]).float()
            else:
                y_unnorm = torch.from_numpy(self.data_unnorm[hi:hi + self.horizon]).float()
            # train_data = self.data[lo: hi]
            # target_data = self.data[hi:hi + self.horizon]
            # x_unnorm = torch.from_numpy(self.data_unnorm[lo: hi]).float()
            # y_unnorm = torch.from_numpy(self.data_unnorm[hi:hi + self.horizon]).float()
            x = torch.from_numpy(train_data).type(torch.float)
            y = torch.from_numpy(target_data).type(torch.float)
        else:
            train_data = self.data[lo: hi]
            target_data = self.data[hi:hi + self.horizon]
            x_unnorm = torch.from_numpy(self.data_unnorm[lo: hi]).float()
            y_unnorm = torch.from_numpy(self.data_unnorm[hi:hi + self.horizon]).float()
            x = torch.from_numpy(train_data).type(torch.float)
            y = torch.from_numpy(target_data).type(torch.float)

        if self.step_type == 'single':
            return x, y[-1:], x_unnorm, y_unnorm[-1:]
        return x, y, x_unnorm, y_unnorm

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx
