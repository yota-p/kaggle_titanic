#!/bin/bash

PROJECT_NAME="titanic"

if [ "$1" = "" ]
then
    echo "Error: SUBMISSION_FILE_PATH required"
    exit 1
else
    SUBMISSION_FILE_PATH=$1
fi

if ["$2" = ""]
then
    echo "Error: MESSAGE required"
    exit 1
else
    MESSAGE=$2
fi

kaggle competitions submit -c $PROJECT_NAME -f $SUBMISSION_FILE_PATH -m $MESSAGE
