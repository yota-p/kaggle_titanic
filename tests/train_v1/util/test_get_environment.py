import pytest
from src.train_v1.util.get_environment import get_datadir
import pathlib


class TestGetDataDir:
    def test_kaggleInteractive(self, mocker):
        mocker.patch('src.train_v1.util.get_environment.get_exec_env', return_value='kaggle-Interactive')
        assert(get_datadir() == '/kaggle/working/data/train_v1')

    def test_kaggleBatch(self, mocker):
        mocker.patch('src.train_v1.util.get_environment.get_exec_env', return_value='kaggle-Batch')
        assert(get_datadir() == '/kaggle/working/data/train_v1')

    def test_colab(self, mocker):
        mocker.patch('src.train_v1.util.get_environment.get_exec_env', return_value='colab')
        assert(get_datadir() == '')

    def test_local(self, mocker):
        mocker.patch('src.train_v1.util.get_environment.get_exec_env', return_value='local')
        assert(get_datadir() == str(pathlib.Path('./data/train_v1').resolve()))

    def test_other(self, mocker):
        mocker.patch('src.train_v1.util.get_environment.get_exec_env', return_value='')
        with pytest.raises(ValueError):
            get_datadir()
