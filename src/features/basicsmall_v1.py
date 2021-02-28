from pathlib import Path
import pandas as pd
import numpy as np
import shutil
from src.util.get_environment import get_datadir


def main():
    '''
    Function:
    - Extract from train where mod(date, 100)==0. (Approx. 5% of full dataset)
      Note: Column 'date' is integer distributed uniformly in range 0 <= date < 500.
      Note: example_sample_submission, example_test, features aren't changed from original
    Input:
    - train.pkl
    - example_sample_submission.pkl
    - example_test.pkl
    - features.pkl
    Output:
    - train.pkl
    - example_sample_submission.pkl
    - example_test.pkl
    - features.pkl
    '''
    DATASET = 'basicsmall_v1'
    DATA_DIR = get_datadir()
    IN_DIR = f'{DATA_DIR}/processed/basic_v1'
    OUT_DIR = f'{DATA_DIR}/processed/{DATASET}'
    Path(OUT_DIR).mkdir(exist_ok=True, parents=True)
    print(DATA_DIR)

    df = pd.read_pickle(f'{IN_DIR}/train.pkl')
    print('# Of unique dates before extraction:')
    print(df['date'].nunique())

    df = df[np.mod(df['date'], 100) == 0]
    df.to_pickle(f'{OUT_DIR}/train.pkl')

    print('# Of unique dates before extraction:')
    print(df['date'].nunique())

    for file in ['example_sample_submission', 'example_test', 'features']:
        shutil.copyfile(f'{IN_DIR}/{file}.pkl', f'{OUT_DIR}/{file}.pkl')


if __name__ == '__main__':
    main()
