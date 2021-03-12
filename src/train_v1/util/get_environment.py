import os
import sys
import torch
import hydra
import subprocess
from typing import Union


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


# This doesn't work correct with hydra
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
    experiment_dirname = 'train_v1'
    env = get_exec_env()
    if env in ['kaggle-Interactive', 'kaggle-Batch']:
        return '/kaggle/working/data/' + experiment_dirname
    elif env == 'colab':
        return ''
    elif env == 'local':
        return get_original_cwd() + '/data/' + experiment_dirname
    else:
        raise ValueError


def has_changes_to_commit() -> bool:
    '''
    Function to check for changes not commited in current git repository.
    Returns False if current repository is not a git repo.
    '''
    command = 'git diff --exit-code'
    proc = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode in [0, 128]:  # 0: no changes, 128: not a repository
        return False
    elif proc.returncode == 1:  # 1: has changes
        return True
    else:
        raise ValueError(f'Unexpected return code: {proc.returncode}')


def get_head_commit() -> Union[str, None]:
    '''
    Function to get head commit in current git repository.
    Returns None if current repository is not a git repo.
    '''
    command = "git rev-parse HEAD"
    proc = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode == 0:
        commit = subprocess.check_output(command.split()).strip().decode('utf-8')
        return commit
    elif proc.returncode == 128:  # 128: not a repository
        return None
    else:
        raise ValueError(f'Unexpected return code: {proc.returncode}')


if __name__ == '__main__':
    print(f'get_exec_env(): {get_exec_env()}')
    print(f'is_gpu(): {is_gpu()}')
    print(f'is_ipykernel(): {is_ipykernel()}')
    print(f'get_original_cwd(): {get_original_cwd()}')
    print(f'get_datadir(): {get_datadir()}')
    print(f'has_changes_to_commit(): {has_changes_to_commit()}')
    print(f'get_head_commit(): {get_head_commit()}')
