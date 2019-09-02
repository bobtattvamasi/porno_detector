#!/bin/bash

gnome-terminal -x cvlc -d --ttl 12 $1 --sout-deinterlace-mode=discard  --sout '#duplicate{dst=std{access=udp, mux=ts, dst=localhost:1235}, select="program='$2'"}'
