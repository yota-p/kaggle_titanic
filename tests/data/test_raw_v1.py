import pytest
from pathlib import Path
import hashlib
from src.data import raw_v1


class TestRawV1:
    '''
    Note:
    No tests implemented for download & unzip
    '''
    @pytest.mark.parametrize('ENV', ['kaggle-Interactive', 'kaggle-Batch'])
    def test_kaggle(self, mocker, tmpdir, ENV):
        DATA_DIR = str(tmpdir) + '/working/data'  # This is equal to /kaggle/working
        IN_DIR = f'{DATA_DIR}/../../input/jane-street-market-prediction'
        OUT_DIR = f'{DATA_DIR}/raw'
        Path(IN_DIR).mkdir(exist_ok=True, parents=True)
        Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

        # create input
        # dummy.pkl won't be copied
        copyfiles = ['train.csv',
                     'example_sample_submission.csv',
                     'example_test.csv',
                     'features.csv']
        dummyfile = 'dummy.pkl'

        for file in copyfiles + [dummyfile]:
            f = open(f'{IN_DIR}/{file}', 'w')
            f.write(f'This is {file}')

        mocker.patch('src.data.raw_v1.get_datadir', return_value=DATA_DIR)
        mocker.patch('src.data.raw_v1.get_exec_env', return_value=ENV)
        raw_v1.main()

        # assert copied files
        for file in copyfiles:
            with open(f'{IN_DIR}/{file}', 'rb') as f:
                in_hash = hashlib.sha256(f.read()).hexdigest()
            with open(f'{OUT_DIR}/{file}', 'rb') as f:
                out_hash = hashlib.sha256(f.read()).hexdigest()
            assert(in_hash == out_hash)

        # non-csv won't be copied
        with pytest.raises(FileNotFoundError):
            open(f'{OUT_DIR}/{dummyfile}', 'rb')

    def test_abnormal(self, mocker, tmpdir):
        DATA_DIR = str(tmpdir)
        OUT_DIR = f'{DATA_DIR}/raw'
        Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

        ENV = 'fizz'

        mocker.patch('src.data.raw_v1.get_datadir', return_value=DATA_DIR)
        mocker.patch('src.data.raw_v1.get_exec_env', return_value=ENV)

        with pytest.raises(ValueError):
            raw_v1.main()
