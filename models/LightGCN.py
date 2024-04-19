import numpy as np
import torch
import torch.nn as nn
from models.common.abstract import General
import scipy.sparse as sp
import os


class LightGCN_Encoder(General):
    def __init__(self, config, dataloader):
        super(LightGCN_Encoder, self).__init__(config, dataloader)

        self.interaction_matrix = dataloader.inter_matrix(
            form='coo').astype(np.float32)
        self.embedding_size = config['embedding_size']

        self.n_layers = 3 if config['n_layers'] is None else config['n_layers']
        self.layers = [self.embedding_size] * self.n_layers

        self.drop_ratio = 1.0
        self.drop_flag = True
        if not os.path.exists('../dataset/piSim.npy'):
            raise ValueError("Please use GIS.py to generate piSim.npy&DiseaSim.npy first")
        self.piRNA_e = np.load('../dataset/piSim.npy')
        self.disease_e = np.load('../dataset/DiseaSim.npy')
        self.linear1 = nn.Linear(self.piRNA_num, self.embedding_size).to(self.device)
        self.linear2 = nn.Linear(self.disease_num, self.embedding_size).to(self.device)
        self._init_model()
        self.interaction_matrix = self.get_normalized_interaction_matrix().to(self.device)

    def _init_model(self):
        self.embedding_dict = nn.ParameterDict({
            'piRNA_emb': nn.Parameter(self.linear1(torch.FloatTensor(self.piRAN_e).to(self.device))),
            'disease_emb': nn.Parameter(self.linear2(torch.FloatTensor(self.disease_e).to(self.device)))
        })
        return self.embedding_dict

    def get_normalized_interaction_matrix(self):
        r"""Get the normalized interaction matrix of piRNAs and diseases.

        Construct the square matrix from the training dataset and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.piRNA_num + self.disease_num,
                           self.piRNA_num + self.disease_num), dtype=np.float32)
        im_transpose = self.interaction_matrix.transpose()
        data_dict = {coordinate: 1 for coordinate in
                     zip(self.interaction_matrix.row, self.piRNA_num + self.interaction_matrix.col)}
        data_dict.update({coordinate: 1 for coordinate in zip(self.piRNA_num + im_transpose.row, im_transpose.col)})
        # Build association matrix
        A._update(data_dict)
        # normalize
        row_sum = (A > 0).sum(axis=1)
        # add epsilon to avoid dividing by zero
        diag = np.array(row_sum.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        diag_matrix = sp.diags(diag)
        laplace = diag_matrix * A * diag_matrix
        # covert norm_adj matrix to tensor
        laplace = sp.coo_matrix(laplace)
        row = laplace.row
        col = laplace.col
        coordinate = torch.LongTensor([row, col])
        data = torch.FloatTensor(laplace.data)
        sparse_tensor = torch.sparse.FloatTensor(coordinate, data, torch.Size(laplace.shape))
        return sparse_tensor

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(self.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(self.device)
        return out * (1. / (1 - rate))

    def forward(self, inputs):
        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict({
            'piRNA_emb': nn.Parameter(initializer(self.linear1(torch.FloatTensor(self.piRAN_e).to(self.device)))),
            'disease_emb': nn.Parameter(initializer(self.linear2(torch.FloatTensor(self.disease_e).to(self.device))))
        })
        A_hat = self.sparse_dropout(self.interaction_matrix,
                                    np.random.random() * self.drop_ratio,
                                    self.interaction_matrix._nnz()) if self.drop_flag else self.interaction_matrix

        ego_embeddings = torch.cat([self.embedding_dict['piRNA_emb'], self.embedding_dict['disease_emb']], 0)
        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            all_embeddings += [ego_embeddings]

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        piRNA_all_embeddings = all_embeddings[:self.piRNA_num, :]
        disease_all_embeddings = all_embeddings[self.piRNA_num:, :]

        piRNAs, diseases = inputs[0], inputs[1]
        piRNA_embeddings = piRNA_all_embeddings[piRNAs, :]
        disease_embeddings = disease_all_embeddings[diseases, :]

        return piRNA_embeddings, disease_embeddings

    @torch.no_grad()
    def get_embedding(self):
        embeddings_one_layer = torch.cat([self.embedding_dict['piRNA_emb'], self.embedding_dict['disease_emb']], 0)
        embeddings_all_layer = [embeddings_one_layer]
        for i in range(len(self.layers)):
            embeddings_all_layer.append(torch.sparse.mm(self.interaction_matrix, embeddings_one_layer))
        embeddings_all_layer = torch.stack(embeddings_all_layer, dim=1)
        embeddings_all_layer = torch.mean(embeddings_all_layer, dim=1)

        piRNA_embeddings_output = embeddings_all_layer[:self.piRNA_num, :]
        disease_embeddings_output = embeddings_all_layer[self.piRNA_num:, :]

        return piRNA_embeddings_output, disease_embeddings_output
