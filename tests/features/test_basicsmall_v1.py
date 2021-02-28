import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from src.features import basicsmall_v1


class TestBasicSmallV1:
    def test_create_data(self, mocker, tmpdir):
        DATASET = 'basicsmall_v1'
        DATA_DIR = str(tmpdir)
        IN_DIR = f'{DATA_DIR}/processed/basic_v1'
        OUT_DIR = f'{DATA_DIR}/processed/{DATASET}'
        Path(IN_DIR).mkdir(exist_ok=True, parents=True)
        Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

        # create input
        df_in = pd.DataFrame({'key': [1, 2, 3, 4], 'date': [np.int16(0), np.int16(10), np.int16(100), np.int16(101)]})
        df_in.to_pickle(f'{IN_DIR}/train.pkl')

        copyfiles = ['example_sample_submission.pkl', 'example_test.pkl', 'features.pkl']
        for file in copyfiles:
            data = pd.DataFrame({'filename': [file]})
            data.to_pickle(f'{IN_DIR}/{file}')

        # call target
        mocker.patch('src.features.basicsmall_v1.get_datadir', return_value=DATA_DIR)
        basicsmall_v1.main()

        # assert
        df_actual = pd.read_pickle(f'{OUT_DIR}/train.pkl')
        df_expected = df_in[[True, False, True, False]]

        assert df_actual.equals(df_expected)

        # assert copied files
        for file in copyfiles:
            with open(f'{IN_DIR}/{file}', 'rb') as f:
                in_hash = hashlib.sha256(f.read()).hexdigest()
            with open(f'{OUT_DIR}/{file}', 'rb') as f:
                out_hash = hashlib.sha256(f.read()).hexdigest()
            assert(in_hash == out_hash)
