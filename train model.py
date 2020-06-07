import tensorflow as tf
import os

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data
config = Config()


with tf.Session() as sess:
    data = prepare_train_data(config)
    print(data)

