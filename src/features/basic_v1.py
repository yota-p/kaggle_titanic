# Machine requirement: Memory size > 16GB to execute this script
import os
from pathlib import Path
import pandas as pd
from src.util.reduce_mem_usage import reduce_mem_usage
from src.util.get_environment import get_datadir
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)


def main():
    '''
    Function:
    - Reduce size of train.csv from 2.5GB to 600MB (On memory)
      Note: Files except train.csv aren't reduced (small enough)
    - Cut weight <=0
    - Calculate action
    Output:
    - train.pkl
    - example_sample_submission.pkl
    - example_test.pkl
    - features.pkl
    - train_dtypes.csv
    '''
    DATASET = 'basic_v1'
    DATA_DIR = get_datadir()
    IN_DIR = f'{DATA_DIR}/raw'
    OUT_DIR = f'{DATA_DIR}/processed/{DATASET}'
    Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

    if os.path.exists(f'{OUT_DIR}/train.pkl'):
        print(f'Output exists. Skip processing: {OUT_DIR}/train.pkl')
        return

    df = pd.read_csv(f'{IN_DIR}/train.csv')
    print(df.info())  # Size of the dataframe is about 2.5 GB
    dfnew = reduce_mem_usage(df)
    dfnew.memory_usage(deep=True)
    print(dfnew.info())  # The dataframe size has decreased to 1.2GB (50% less).

    # add target column
    dfnew['action'] = (dfnew['resp'] > 0).astype('int32')

    # Save reduced data
    dfnew.dtypes.to_csv(f'{OUT_DIR}/train_dtypes.csv', header=False)
    dfnew.to_pickle(f'{OUT_DIR}/train.pkl')
    del df, dfnew

    for file in ['example_sample_submission', 'example_test', 'features']:
        df = pd.read_csv(f'{IN_DIR}/{file}.csv')
        df.to_pickle(f'{OUT_DIR}/{file}.pkl')


if __name__ == '__main__':
    main()
