#!/bin/bash

if [ $1 -eq 1 ];
then
    python3 dqn-half-pong-org.py -width 640 -height 480 -train
elif [ $1 -eq 2 ];
then
    python3 dqn-half-pong-org.py
elif [ $1 -eq 3 ];
then
    python3 dqn-half-pong-syr.py
else
    echo "Unknown command"
fi
