import math
import torch
import random
import numpy as np
from scipy.sparse import coo_matrix


class AbstractDataLoader(object):
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, use_negative_sample=False, shuffle=False):
        # use __str__ to init inter_num, piRNA_num and dataset_num
        print(dataset)
        self.config = config
        self.dataset = dataset
        self.dataset_copy = self.dataset.copy(self.dataset.data)
        self.additional_dataset = additional_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_negative_sample = use_negative_sample
        self.device = config['device']
        self.sparsity = 1 - self.dataset.inter_num / self.dataset.piRNA_num / self.dataset.disease_num
        self.batch_first_index = 0
        self.inter_pr = 0

    def __len__(self):
        return math.ceil(self.pr_end / self.batch_size)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.batch_first_index >= self.pr_end:
            self.batch_first_index = 0
            self.inter_pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        raise NotImplementedError('Method [next_batch_data] should be implemented.')


class TrainDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, batch_size=1, shuffle=False):
        super().__init__(config, dataset, additional_dataset=None, batch_size=batch_size, use_negative_sample=True,
                         shuffle=shuffle)
        self.diseases_eachRNA = dict()
        self.piRNAs = self.dataset.data[self.dataset.piRNA_label].unique()
        self.diseases = self.dataset.data[self.dataset.disease_label].unique()
        self.use_full_sampling = config['use_full_sampling']

        if config['use_neg_sampling']:
            if self.use_full_sampling:
                self.sample_func = self._get_full_piRNA_sample
            else:
                self.sample_func = self._get_neg_sample
        else:
            self.sample_func = self._get_non_neg_sample
        self._generate_diseases_eachRNA()

    def pretrain_setup(self):
        if self.shuffle:
            self.dataset = self.dataset_copy.copy(self.dataset_copy.data)
        if self.use_full_sampling:
            self.piRNAs.sort()
        random.shuffle(self.diseases)

    def inter_matrix(self, form='coo', value_field=None):
        if not self.dataset.piRNA_label or not self.dataset.disease_label:
            raise ValueError('dataset doesn\'t exist uid/iid, thus can not converted to sparse matrix')
        return self._create_sparse_matrix(self.dataset.data, self.dataset.piRNA_label,
                                          self.dataset.disease_label, form, value_field)

    def _create_sparse_matrix(self, source_data, source_field, target_field, form='coo', value_label=None):
        # row index in sparse matrix
        row_index = source_data[source_field].values
        # column index in sparse matrix
        column_index = source_data[target_field].values
        # SparseMatrix[row_index[i], column_index[i]] = value[i]
        if value_label is None:
            value = np.ones(len(source_data))
        else:
            if value_label not in source_data.columns:
                raise ValueError('{} is not in `source_data`\'s features.'.format(value_label))
            value = source_data[value_label].values
        mat = coo_matrix((value, (row_index, column_index)), shape=(self.dataset.piRNA_num, self.dataset.disease_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError('sparse matrix format [{}] has not been implemented.'.format(form))

    @property
    def pr_end(self):
        if self.use_full_sampling:
            return len(self.piRNAs)
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()
        if self.use_full_sampling:
            np.random.shuffle(self.piRNAs)

    def _next_batch_data(self):
        return self.sample_func()

    def _get_neg_sample(self):
        batch_data = self.dataset.data[self.batch_first_index: self.batch_first_index + self.batch_size]
        self.batch_first_index += self.batch_size

        piRNAs = batch_data[self.config['piRNA_LABEL']]
        piRNA_tensor = torch.tensor(piRNAs.values).type(torch.LongTensor).to(
            self.device)
        disease_tensor = torch.tensor(batch_data[self.config['DISEASE_LABEL']].values).type(torch.LongTensor).to(
            self.device)
        batch_tensor = torch.cat((torch.unsqueeze(piRNA_tensor, 0),
                                  torch.unsqueeze(disease_tensor, 0)))

        neg_ids = self._sample_neg_ids(piRNAs).to(self.device)
        batch_tensor = torch.cat((batch_tensor, neg_ids.unsqueeze(0)))
        return batch_tensor

    def _get_non_neg_sample(self):
        batch_data = self.dataset[self.batch_first_index: self.batch_first_index + self.batch_size]
        self.batch_first_index += self.batch_size

        piRNA_tensor = torch.tensor(batch_data[self.config['piRNA_LABEL']].values).type(torch.LongTensor).to(
            self.device)
        disease_tensor = torch.tensor(batch_data[self.config['DISEASE_LABEL']].values).type(torch.LongTensor).to(
            self.device)
        batch_tensor = torch.cat((torch.unsqueeze(piRNA_tensor, 0), torch.unsqueeze(disease_tensor, 0)))
        return batch_tensor

    def _get_full_piRNA_sample(self):
        piRNA_tensor = (torch.tensor(self.piRNAs[self.batch_first_index: self.batch_first_index + self.batch_size]).
                        type(torch.LongTensor).to(self.device))
        self.batch_first_index += self.batch_size
        return piRNA_tensor

    def _sample_neg_ids(self, u_ids):
        neg_ids = []
        for u in u_ids:
            iid = self._random()
            while iid in self.diseases_eachRNA[u]:
                iid = self._random()
            neg_ids.append(iid)
        return torch.tensor(neg_ids).type(torch.LongTensor)

    def _random(self):
        rd_id = random.sample(self.diseases, 1)[0]
        return rd_id

    def _generate_diseases_eachRNA(self):
        piRNA_label = self.dataset.piRNA_label
        disease_label = self.dataset.disease_label
        group_by_piRNA = self.dataset.data.groupby(piRNA_label)[disease_label]
        for piRNA, diseases in group_by_piRNA:
            self.diseases_eachRNA[piRNA] = diseases.values
        return self.diseases_eachRNA


class EvalDataLoader(AbstractDataLoader):
    def __init__(self, config, dataset, additional_dataset=None,
                 batch_size=1, shuffle=False):
        super().__init__(config, dataset, additional_dataset=additional_dataset,
                         batch_size=batch_size, use_negative_sample=False, shuffle=shuffle)

        if additional_dataset is None:
            raise ValueError('Training datasets is nan')
        self.eval_diseases_per_u = []
        self.eval_len_list = []
        self.train_pos_len_list = []

        self.eval_u = self.dataset.data[self.dataset.piRNA_label].unique()
        self.pos_diseases_per_u = self._get_pos_diseases_per_u(self.eval_u).to(self.device)
        self._get_eval_diseases_per_u(self.eval_u)
        self.eval_u = torch.tensor(self.eval_u).type(torch.LongTensor).to(self.device)

    @property
    def pr_end(self):
        return self.eval_u.shape[0]

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        inter_cnt = sum(self.train_pos_len_list[self.batch_first_index: self.batch_first_index + self.batch_size])
        batch_piRNAs = self.eval_u[self.batch_first_index: self.batch_first_index + self.batch_size]
        batch_mask_matrix = self.pos_diseases_per_u[:, self.inter_pr: self.inter_pr + inter_cnt].clone()
        batch_mask_matrix[0] -= self.batch_first_index
        self.inter_pr += inter_cnt
        self.batch_first_index += self.batch_size

        return [batch_piRNAs, batch_mask_matrix]

    def _get_pos_diseases_per_u(self, eval_piRNAs):
        pirna_id_field = self.additional_dataset.pirna_id_field
        disease_id_field = self.additional_dataset.disease_id_field
        uid_freq = self.additional_dataset.data.groupby(pirna_id_field)[disease_id_field]
        u_ids = []
        i_ids = []
        for i, u in enumerate(eval_piRNAs):
            u_ls = uid_freq.get_group(u).values
            i_len = len(u_ls)
            self.train_pos_len_list.append(i_len)
            u_ids.extend([i] * i_len)
            i_ids.extend(u_ls)
        return torch.tensor([u_ids, i_ids]).type(torch.LongTensor)

    def _get_eval_diseases_per_u(self, eval_piRNAs):
        piRNA_label = self.dataset.piRNA_label
        disease_label = self.dataset.disease_label
        uid_freq = self.dataset.data.groupby(piRNA_label)[disease_label]
        for u in eval_piRNAs:
            u_ls = uid_freq.get_group(u).values
            self.eval_len_list.append(len(u_ls))
            self.eval_diseases_per_u.append(u_ls)
        self.eval_len_list = np.asarray(self.eval_len_list)

    def get_eval_diseases(self):
        return self.eval_diseases_per_u

    def get_eval_len_list(self):
        return self.eval_len_list

    def get_eval_piRNAs(self):
        return self.eval_u.cpu()
