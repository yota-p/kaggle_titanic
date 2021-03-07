import subprocess
from typing import Union


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
    print(f'has_changes_to_commit(): {has_changes_to_commit()}')
    print(f'get_head_commit(): {get_head_commit()}')
