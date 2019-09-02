#!/bin/bash

ps axf | grep vlc | grep -v grep | awk '{print "kill -9 " $1}' | sh
