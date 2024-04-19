import torch
import torch.nn as nn
import torch.nn.functional as F
from models.LightGCN import LightGCN_Encoder
from models.common.loss import L2Loss
from models.common.abstract import General


class iPiDA_CL(General):
    def __init__(self, config, dataset):
        super(iPiDA_CL, self).__init__(config, dataset)
        self.embedding_size = config['embedding_size']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.linear_weight = config['linear_weight']

        self.online_encoder = LightGCN_Encoder(config, dataset)
        self.predictor = nn.Linear(self.embedding_size, self.embedding_size)
        self.reg_loss = L2Loss()

    def forward(self):
        piRNA_online, disease_online = self.online_encoder.get_embedding()
        piRNA_target = F.dropout(piRNA_online.detach(), self.dropout)
        disease_target = F.dropout(disease_online.detach(), self.dropout)
        return piRNA_online, piRNA_target, disease_online, disease_target

    @torch.no_grad()
    def get_embedding(self):
        p_online, d_online = self.online_encoder.get_embedding()
        return self.predictor(p_online), p_online, self.predictor(d_online), d_online

    def loss_fn(self, p, z):
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def calculate_loss(self, interaction):
        piRNA_online, piRNA_target, disease_online, disease_target = self.forward()
        piRNAs, diseases = interaction[0], interaction[1]

        piRNA_online_batch = piRNA_online[piRNAs, :]
        disease_online_batch = disease_online[diseases, :]
        piRNA_target_batch = piRNA_target[piRNAs, :]
        disease_target_batch = disease_target[diseases, :]

        reg_loss = self.reg_loss(piRNA_online_batch, disease_online_batch)

        piRNA_online_batch, disease_online_batch = self.predictor(piRNA_online_batch), self.predictor(disease_online_batch)
        loss_cl = (self.loss_fn(piRNA_online_batch, disease_target_batch)+self.loss_fn(disease_online_batch, piRNA_target_batch))/2

        linear_loss = .0
        for param in self.predictor.parameters():
            linear_loss += torch.norm(param, 1) ** 2

        return loss_cl + self.reg_weight * reg_loss + self.linear_weight * linear_loss

    def full_sort_predict(self, interaction):
        piRNAs = interaction[0]

        piRNA_online, piRNA_target, disease_online, disease_target = self.get_embedding()
        score_po_dt = torch.matmul(piRNA_online[piRNAs], disease_target.transpose(0, 1))
        score_pt_do = torch.matmul(piRNA_target[piRNAs], disease_online.transpose(0, 1))
        scores = score_pt_do + score_po_dt

        return scores

    # def getebding(self):
    #     p_online, p_target, d_online, d_target = self.get_embedding()
    #     return p_online, p_target, d_online, d_target

    def dependent_test(self, test_pd):
        num_piRNAs = len(test_pd)
        num_diseases = len(test_pd[0])
        self.piRNA_embedding = nn.Embedding(num_piRNAs, self.embedding_size)
        self.disease_embedding = nn.Embedding(num_diseases, self.embedding_size)
        # p_online = F.dropout(self.piRNA_embedding.weight, 0.0)
        # d_online = F.dropout(self.disease_embedding.weight, 0.0)
        # p_target = self.predictor(p_online)
        # d_target = self.predictor(d_online)
        p_online, p_target, d_online, d_target = self.get_embedding()
        score_mat_pd = torch.matmul(p_online, d_target.transpose(0, 1))
        score_mat_dp = torch.matmul(p_target, d_online.transpose(0, 1))
        scores = score_mat_pd + score_mat_dp

        return scores
