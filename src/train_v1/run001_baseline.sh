#!/bin/bash

python src/train_v1/train_v1.py run=run001 model=CatBoostClassifier

python src/train_v1/train_v1.py run=run001 model=LGBMClassifier

python src/train_v1/train_v1.py run=run001 model=RandomForestClassifier2

python src/train_v1/train_v1.py run=run001 model=XGBClassifier
