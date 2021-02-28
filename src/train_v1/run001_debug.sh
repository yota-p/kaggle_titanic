#!/bin/sh

python src/train_v1/train_v1.py \
run=run001 \
experiment.tags.exec=dev \
model.model_param.n_estimators=2 \
model.model_param.max_depth=2 \
model.model_param.tree_method=auto \
$1
