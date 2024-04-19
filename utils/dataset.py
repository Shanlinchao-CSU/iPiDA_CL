from logging import getLogger
from collections import Counter
import os
import pandas as pd
import numpy as np


# Name unified into "dataset", DataFrame data names "self.data"
class LoadDataset(object):
    def __init__(self, config, fold, data=None):
        self.config = config
        self.dataset_path = os.path.abspath(config['dataset_path'])
        self.dataset_loaded = False
        self.logger = getLogger()
        # main.py 23
        self.dataset_name = config['dataset']
        self.fold = fold
        # piRNA_id:token
        self.piRNA_label = self.config['piRNA_LABEL']
        self.piRNA_num = self.config['piRNA_num']
        # disease_id:token
        self.disease_label = self.config['DISEASE_LABEL']
        self.disease_num = self.config['disease_num']
        # timestamp:float
        self.timestamp_label = self.config['TIMESTAMP_LABEL']
        self.inter_num = -1

        if data is not None:
            self.data = data
            self.inter_num = len(self.data)
            return

        self.filter_split_infoStr = self.getSFInfoStr()
        self.dataset_filename = 'data_{}.inter'.format(self.fold)

        if self.config['load_dataset'] and self._load_dataset():
            self.dataset_loaded = True
            self.logger.info('\nPreprocessed Data has been loaded from: ' + self.dataset_path + '\n')
            return
        self._from_scratch()
        self._data_processing()

    def getSFInfoStr(self):
        min_num_p = 1 if self.config['min_piRNA_inter_num'] is None else max(self.config['min_piRNA_inter_num'], 1)
        min_num_d = 1 if self.config['min_disease_inter_num'] is None else max(self.config['min_disease_inter_num'], 1)

        ratios = self.config['split_ratio']
        total_ratio = sum(ratios)
        ratios = [ratio for ratio in ratios if ratio > .0]
        ratios = [str(int(_ * 10 / total_ratio)) for _ in ratios]
        s = ''.join(ratios)
        return 'u{}i{}_s'.format(min_num_p, min_num_d) + s

    # Read dataset/data_i.inter
    def _load_dataset(self):
        file_path = os.path.join(self.dataset_path, self.dataset_filename)
        if not os.path.isfile(file_path):
            raise ValueError('File {} not exist'.format(file_path))
        self.data = self._load_data(file_path, self.config['load_cols']+[self.config['data_dividing_label']])
        self.inter_num = len(self.data)
        return True

    def _load_data(self, file_path, load_columns):
        with open(file_path, 'r') as f:
            labels = f.readline()[:-1]
            label_delimiter = self.config['label_delimiter']
            if set(labels.split(label_delimiter)) != set(load_columns):
                raise ValueError('File {} lost some required columns.'.format(file_path))
        return pd.read_csv(file_path, sep=self.config['label_delimiter'], usecols=load_columns)

    def _from_scratch(self):
        self.logger.info('Loading {} from scratch'.format(self.__class__))
        file_path = os.path.join(self.dataset_path, '{}.inter'.format(self.dataset_name))
        if not os.path.isfile(file_path):
            raise ValueError('File {} not exist'.format(file_path))
        self.data = self._load_data(file_path, self.config['load_cols'])
        self.inter_num = len(self.data)

    def _data_processing(self):
        self.data.dropna(inplace=True)
        self.data.drop_duplicates(inplace=True)
        self._filter()
        self._reset_index()

    def _filter(self):
        data = self.data
        while True:
            illegal_piRNAs_id = self._get_illegal_ids_by_inter_num(data, field=self.piRNA_label,
                                                            max_num=self.config['max_piRNA_inter_num'],
                                                            min_num=self.config['min_piRNA_inter_num'])
            illegal_diseases_id = self._get_illegal_ids_by_inter_num(data, field=self.disease_label,
                                                              max_num=self.config['max_disease_inter_num'],
                                                              min_num=self.config['min_disease_inter_num'])

            if len(illegal_piRNAs_id) != 0 or len(illegal_diseases_id) != 0:
                dropped_inter = pd.Series(False, index=data.index)
                if self.piRNA_label:
                    dropped_inter |= data[self.piRNA_label].isin(illegal_piRNAs_id)
                if self.disease_label:
                    dropped_inter |= data[self.disease_label].isin(illegal_diseases_id)
                data.drop(data.index[dropped_inter], inplace=True)

    def _get_illegal_ids_by_inter_num(self, data, field, max_num=None, min_num=None):
        if field is None:
            return set()
        if max_num is None and min_num is None:
            return set()

        max_num = max_num or np.inf
        min_num = min_num or 0

        all_id = data[field].values
        id_counter = Counter(all_id)
        all_id = {id for id in id_counter if id_counter[id] < min_num or id_counter[id] > max_num}

        self.logger.debug('[{}] illegal_ids_by_inter_num, field=[{}]'.format(len(all_id), field))
        return all_id

    def _reset_index(self):
        data = self.data
        if data.empty:
            raise ValueError('Some feat is empty, please check the filtering settings.')
        data.reset_index(drop=True, inplace=True)

    def split(self, ratios):
        if self.dataset_loaded:
            result = []
            dividing_label = self.config['data_dividing_label']
            for i in range(2):
                # train:validate:test
                data_split = (self.data[self.data[dividing_label] == i].copy()
                              .drop(dividing_label, inplace=True, axis=1))
                result.append(self.copy(data_split))
            return result

        total_ratio = sum(ratios)
        ratios = [ratio for ratio in ratios if ratio > .0]
        ratios = [_ / total_ratio for _ in ratios]

        split_ratios = np.cumsum(ratios)[:-1]
        split_timestamps = list(np.quantile(self.data[self.timestamp_label], split_ratios))

        data_train = self.data.loc[self.data[self.timestamp_label] < split_timestamps[0]]

        unique_piRNAs = pd.unique(data_train[self.piRNA_label])
        unique_diseases = pd.unique(data_train[self.disease_label])

        piRNA_id_map = {k: i for i, k in enumerate(unique_piRNAs)}
        self.data[self.piRNA_label] = self.data[self.piRNA_label].map(piRNA_id_map)
        disease_id_map = {k: i for i, k in enumerate(unique_diseases)}
        self.data[self.disease_label] = self.data[self.disease_label].map(disease_id_map)

        self.data.dropna(inplace=True)
        self.data = self.data.astype(int)

        data_list = []
        start = 0
        for i in split_timestamps:
            data_list.append(self.copy(self.data.loc[(start <= self.data[self.timestamp_label]) & (self.data[self.timestamp_label] < i)].copy()))
            start = i
        data_list.append(self.copy(self.data.loc[start <= self.data[self.timestamp_label]].copy()))
        self._save_to_disk(piRNA_id_map, disease_id_map, data_list)
        return data_list

    def _save_to_disk(self, piRNA_map, disease_map, data_list):
        if self.config['load_preprocessed_data'] and not self.dataset_loaded:
            dir_name = self.dataset_path
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            piRNA_data = pd.DataFrame(list(piRNA_map.diseases()), columns=[self.piRNA_label, 'new_id'])
            disease_data = pd.DataFrame(list(disease_map.diseases()), columns=[self.disease_label, 'new_id'])
            piRNA_data.to_csv(os.path.join(self.dataset_path,
                                           '{}_u_{}_mapping.csv'.format(self.dataset_name, self.filter_split_infoStr)),
                              sep=self.config['tag_delimiter'], index=False)
            disease_data.to_csv(os.path.join(self.dataset_path,
                                             '{}_i_{}_mapping.csv'.format(self.dataset_name, self.filter_split_infoStr)),
                                sep=self.config['tag_delimiter'], index=False)

            for i, temp_df in enumerate(data_list):
                temp_df[self.config['preprocessed_data_splitting']] = i
            temp_df = pd.concat(data_list)
            temp_df.to_csv(os.path.join(self.dataset_path, self.dataset_filename),
                           sep=self.config['tag_delimiter'], index=False)

    def copy(self, new_data):
        ld = LoadDataset(self.config, self.fold, new_data)
        return ld

    def num(self, field):
        if field not in self.config['load_cols']:
            raise ValueError('field [{}] not defined in dataset'.format(field))
        uni_len = len(pd.unique(self.data[field]))
        return uni_len

    def shuffle(self):
        self.data = self.data.sample(frac=1, replace=False).reset_index(drop=True)

    def sort_by_chronological(self):
        self.data.sort_values(by=[self.timestamp_label], inplace=True, ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        # self.inter_num = len(self.data)
        unique_piRNA = pd.unique(self.data[self.piRNA_label])
        unique_disease = pd.unique(self.data[self.disease_label])
        if self.piRNA_label:
            # self.piRNA_num = len(unique_piRNA)
            self.avg_actions_of_users = self.inter_num / self.piRNA_num
            info.extend(['The number of piRNAs: {}'.format(self.piRNA_num),
                         'Average actions of piRNAs: {}'.format(self.avg_actions_of_users)])
        if self.disease_label:
            # self.disease_num = len(unique_disease)
            self.avg_actions_of_items = self.inter_num / self.disease_num
            info.extend(['The number of piRNAs: {}'.format(self.piRNA_num),
                         'Average actions of diseases: {}'.format(self.avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.piRNA_label and self.disease_label:
            sparsity = 1 - self.inter_num / self.piRNA_num / self.disease_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)
