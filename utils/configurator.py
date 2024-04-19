import re
import os
import yaml
import torch


class Config(object):

    def __init__(self, model=None, dataset=None, initial_config=None):
        initial_config['model'] = model
        initial_config['dataset'] = dataset
        # 从文件加载配置
        self.config = self._load_config(initial_config)
        # 加载默认配置
        self.config.update(initial_config)
        self._init_parameters()
        self._init_device()

    def _load_config(self, config_dict):
        config_loaded = dict()
        cur_dir = os.path.join(os.getcwd(), 'configs')

        for file in ["overall.yaml", "dataset/{}.yaml".format(config_dict['dataset']),
                     "model/{}.yaml".format(config_dict['model'])]:
            file = os.path.join(cur_dir, file)
            if os.path.isfile(file):
                with open(file, 'r', encoding='utf-8') as f:
                    config_loaded.update(yaml.load(f.read(), Loader=self._get_yaml_loader()))
        return config_loaded

    def _get_yaml_loader(self):
        loader = yaml.Loader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        return loader

    def _init_parameters(self):
        smaller_metric = ['rmse', 'mae', 'logloss']
        valid_metric = self.config['valid_metric'].split('@')[0]
        self.config['valid_metric_bigger'] = False if valid_metric in smaller_metric else True

    def _init_device(self):
        use_gpu = self.config['use_gpu']
        if use_gpu and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config['gpu_id'])
            self.config['device'] = torch.device("cuda")
        else:
            self.config['device'] = torch.device("cpu")

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.config[key] = value

    def __getitem__(self, item):
        if item in self.config:
            return self.config[item]
        else:
            return None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.config

    def __str__(self):
        args_info = '\n'
        args_info += '\n'.join(["{}={}".format(arg, value) for arg, value in self.config.items()])
        args_info += '\n\n'
        return args_info

    def __repr__(self):
        return self.__str__()
