#!/bin/bash

echo 'Starting mlflow server'
DIR=`pwd`
TRACKING_URI=$DIR/data/mlruns
ARTIFACT_URI=$DIR/data/mlruns
echo 'TRACKING URI:' $TRACKING_URI
echo 'ARTIFACT URI:' $ARTIFACT_URI
nohup mlflow server --backend-store-uri $TRACKING_URI --default-artifact-root $ARTIFACT_URI --host 0.0.0.0  >> mlflow.log 2>&1 &
