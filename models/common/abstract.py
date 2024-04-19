import numpy as np
import torch
import torch.nn as nn
class Abstract(nn.Module):

    def pre_epoch_processing(self):
        pass

    def post_epoch_processing(self):
        pass

    def calculate_loss(self, interaction):
        raise NotImplementedError

    def predict(self, interaction):
        raise NotImplementedError

    def full_sort_predict(self, interaction):
        raise NotImplementedError

    def __str__(self):
        model_parameters = self.parameters()
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class General(Abstract):

    def __init__(self, config, dataloader):
        super(General, self).__init__()

        # load dataset info
        self.piRNA_LABEL = config['piRNA_LABEL']
        self.DISEASE_LABEL = config['DISEASE_LABEL']
        self.NEG_disease_ID = config['NEG_PREFIX'] + self.DISEASE_LABEL
        self.piRNA_num = config['piRNA_num']
        self.disease_num = config['disease_num']
        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']
