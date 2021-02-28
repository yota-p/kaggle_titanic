#!/bin/bash

echo 'Starting Jupyter server'
nohup jupyter notebook --port 8080 --ip=0.0.0.0 --allow-root >> jupyter.log 2>&1 &

# Wait $WAIT seconds until the server starts
# Command below will show a progress bar
WAIT=15
COUNT=0
while [ $COUNT -le $WAIT ]
do
    BAR=""
    for i in `seq 1 $WAIT`
    do
	if [ $COUNT -eq $i ]
	then
	    BAR="$BAR>"
	else
	    if [ $i -le $COUNT ]
	    then
	        BAR="$BAR="
	    else
	        BAR="$BAR "
	    fi
	fi
    done
    PERCENTAGE=`expr $COUNT \* 100 / $WAIT`
    PERCENTAGE=`printf %3d $PERCENTAGE`
    WAIT=`expr $WAIT - $COUNT`
    WAIT=`printf %2d $WAIT`
    echo -en "Starting Jupyter server. Wait $WAIT s...  [$BAR] $PERCENTAGE %\r"
    sleep 1
    COUNT=`expr $COUNT + 1`
done
echo -e "Starting Jupyter server. Wait $WAIT s...  [$BAR] $PERCENTAGE %\r"

echo 'Started. URL for Jupyter is:'
tail -n 1 jupyter.log | awk '{print substr($0, index($0, "http"))}'
