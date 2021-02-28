from pathlib import Path
import pandas as pd
from src.train_v1.util.get_environment import get_datadir


def main():
    DATA_DIR = get_datadir()
    IN_DIR = f'{DATA_DIR}/raw'
    OUT_DIR = f'{DATA_DIR}/processed/raw_pickle_v1'
    Path(OUT_DIR).mkdir(exist_ok=True, parents=True)

    df_train = pd.read_csv(f'{IN_DIR}/train.csv')
    df_test = pd.read_csv(f'{IN_DIR}/test.csv')

    df_train.to_pickle(f'{OUT_DIR}/train.pkl')
    df_test.to_pickle(f'{OUT_DIR}/test.pkl')


if __name__ == '__main__':
    main()
