
from gfedplat.main import initialize, read_params, outFunc
import os


from gfedplat.Algorithm import Algorithm
from gfedplat.Client import Client
from gfedplat.DataLoader import DataLoader
from gfedplat.Module import Module
from gfedplat.Metric import Metric
from gfedplat.seed import setup_seed

from gfedplat.metric.Correct import Correct
from gfedplat.metric.Precision import Precision
from gfedplat.metric.Recall import Recall

from gfedplat.model.CNN import CNN
from gfedplat.model.MLP import MLP


import gfedplat.algorithm
from gfedplat.algorithm.CoReFed.CoReFed import CoReFed


from gfedplat.dataloaders.separate_data import separate_data, create_data_pool
from gfedplat.dataloaders.DataLoader_cifar10_dir import DataLoader_cifar10_dir
from gfedplat.dataloaders.DataLoader_fashion_dir import DataLoader_fashion_dir


data_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)


pool_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/pool/'
if not os.path.exists(pool_folder_path):
    os.makedirs(pool_folder_path)
