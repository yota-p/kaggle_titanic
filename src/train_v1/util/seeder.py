import os
import random
import numpy as np
import torch
import tensorflow as tf


def seed_everything(seed=42, seed_gpu=True):
    '''
    Fix ramdom seed to make all processes reproducable.
    Call this before running scripts.
    Note deterministic operation may have a negative single-run performance impact.
    To avoid seeding gpu, pass seed_gpu=None.
    Frameworks below are not supported; You have to fix when calling the method.
        - Scikit-learn
        - Optuna
    Reference:
    https://qiita.com/si1242/items/d2f9195c08826d87d6ad
    TODO: Use fwd9m after released
    https://github.com/NVIDIA/framework-determinism#announcement
    '''
    # random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Pytorch
    torch.manual_seed(seed)
    if seed_gpu:
        torch.cuda.manual_seed_all(seed_gpu)
        torch.backends.cudnn.deterministic = True

    # Tensorflow
    tf.random.set_seed(seed)
    if seed_gpu:
        # Note: you need to pip install tensorflow-determinism
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

    # Keras
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    seed_everything()
