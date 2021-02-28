from pathlib import Path
import pandas as pd
from src.util.get_environment import get_datadir


def main():
    '''
    Function:
    - Reduce size of train.csv from 2.5GB to 600MB (On memory)
      Note: Files except train.csv aren't reduced (small enough)
    - Cut weight <=0
    - Calculate action
    - Create crossed features, such as:
        - cross_41_42_43 = feature_41 + feature_42 + feature_43
        - cross_1_2 = feature_1 / feature_2
    Input:
    - basic_v1/train.pkl
    Output:
    - cross_v1/train.pkl
    '''
    DATASET = 'cross_v1'
    DATA_DIR = get_datadir()
    IN_DIR = f'{DATA_DIR}/processed/basic_v1'
    OUT_DIR = f'{DATA_DIR}/processed/{DATASET}'
    Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

    df = pd.read_pickle(f'{IN_DIR}/train.pkl')

    # feature engineering
    df['cross_41_42_43'] = df['feature_41'] + df['feature_42'] + df['feature_43']
    df['cross_1_2'] = df['feature_1'] / (df['feature_2'] + 1e-5)
    df = df[['cross_41_42_43', 'cross_1_2']]

    print(f'Created dataset {DATASET}')
    print(f'Columns: {df.columns}')

    df.to_pickle(f'{OUT_DIR}/train.pkl')


if __name__ == '__main__':
    main()
