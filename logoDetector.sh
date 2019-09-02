#!/bin/bash

cd logo_XXX_detect

sh streamStart.sh $1 $2

python3 visualize.py

ps axf | grep vlc | grep -v grep | awk '{print "kill -9 " $1}' | sh
