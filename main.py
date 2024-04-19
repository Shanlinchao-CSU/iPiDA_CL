from itertools import product
from utils.dataset import LoadDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer, dict2str
import os
import numpy as np

os.environ['NUMEXPR_MAX_THREADS'] = '48'

initial_config = {
    'gpu_id': 0,
    'epochs': 100,
    'n_layers': 4,
    'reg_weight': [0.01],
    'momentum': [0.05],
    'dropout': [0.1],
}

if __name__ == '__main__':

    for i in range(5):
        model = 'iPiDA_CL'
        dataset = 'dateset'
        save_model = True
        config = Config(model, dataset, initial_config)

        dataset = LoadDataset(config, i)
        train_dataset, test_dataset = dataset.split(config['split_ratio'])
        train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=False)

        data_array = np.loadtxt('./dataset/data_{}.inter'.format(i), dtype=int, skiprows=1)
        test_data = data_array[(data_array[:, 3] == 1), :2].tolist()

        hyper_ret = []
        val_metric = config['valid_metric'].lower()
        best_test_value = 0.0
        idx = 0
        best_test_idx = 0

        hyper_list = []
        if "seed" not in config['hyper_parameters']:
            config['hyper_parameters'].extend(['seed'])
        for parameter in config['hyper_parameters']:
            hyper_list.append(config[parameter] or [None])
        combinators = list(product(*hyper_list))
        total_loops = len(combinators)
        for hyper_tuple in combinators:
            for j, k in zip(config['hyper_parameters'], hyper_tuple):
                config[j] = k
            init_seed(config['seed'])
            train_data.pretrain_setup()
            model = get_model(config['model'])(config, train_data).to(config['device'])
            trainer = get_trainer()(config, model, i)
            best_valid_score, best_valid_result = trainer.fit(train_data, valid_data=None, test_data=test_data, saved=save_model)

        model = get_model(config['model'])(config, train_data).to(config['device'])
        trainer = get_trainer()(config, model, i)
        y_true, y_predict, score, coordinate_score = trainer.ROC(test_data)
