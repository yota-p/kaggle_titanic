import os
import sys
import torch
import hydra


def get_exec_env() -> str:
    kaggle_kernel_run_type = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
    if kaggle_kernel_run_type == 'Interactive':
        return 'kaggle-Interactive'
    elif kaggle_kernel_run_type == 'Batch':
        return 'kaggle-Batch'
    elif 'google.colab' in sys.modules:
        return 'colab'
    else:
        return 'local'


def is_gpu() -> bool:
    return torch.cuda.is_available()


def get_device() -> torch.device:
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def is_ipykernel() -> bool:
    if 'ipykernel' in sys.modules:
        # Kaggle Notebook interactive, Kaggle Notebook Batch, Kaggle script Interactive, Jupyter notebook
        return True
    else:
        # ipython, python script, Kaggle script Batch
        return False


def get_original_cwd() -> str:
    '''
    Returns original working directory for execution.
    In CLI, hydra changes cwd to outputs/xxx.
    In Jupyter Notebook, hydra doesn't change cwd.
    This is due to that you need to initialize & compose hydra config using compose API in Jupyter.
    Under compose API, hydra.core.hydra_config.HydraConfig is not initialized.
    Thus, in Jupyter Notebook, you need to avoid calling hydra.utils.get_original_cwd().
    Refer:
    https://github.com/facebookresearch/hydra/issues/828
    https://github.com/facebookresearch/hydra/blob/master/hydra/core/hydra_config.py
    https://github.com/facebookresearch/hydra/blob/master/hydra/utils.py
    '''
    if hydra.core.hydra_config.HydraConfig.initialized():
        return hydra.utils.get_original_cwd()
    else:
        return os.getcwd()


def get_datadir() -> str:
    '''
    Returns absolute path to data store dir
    TODO: Add config for colab
    '''
    env = get_exec_env()
    if env in ['kaggle-Interactive', 'kaggle-Batch']:
        return '/kaggle/working/data'
    elif env == 'colab':
        return ''
    elif env == 'local':
        return get_original_cwd() + '/data'
    else:
        raise ValueError


if __name__ == '__main__':
    print(f'Execution environment: {get_exec_env()}')
    print(f'Is GPU: {is_gpu()}')
    print(f'Is ipykernel: {is_ipykernel()}')
    print(f'Original cwd: {get_original_cwd()}')
    print(f'Datadir: {get_datadir()}')
