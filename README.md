# porno_detector

## 1) Requariments and installation

--- python 3.6
--- tensorflow or tensorflow-gpu
--- vlc (for create single streams from mpts)
in bash write:

:: pip install -r actual_requirements.txt

## 2) downloads

folder models (weights of  nn)-->
https://drive.google.com/open?id=1OR6WOE9hLVYdVRqCRYX_4p5zpl6v928_

porno content for tests-->
https://drive.google.com/open?id=1qfZpyTmlPTnpp1TxrbbxSe-KMYT9qsCB

logo-channel content for tests-->
https://drive.google.com/drive/folders/1LR7X6O_pE4U66hvsyQZVTld0fSaV-fI2

## 3) Usage:

Change file logoDetector.sh

and run it:

:: sh logoDetector.sh <link_to_mpts> <program_id>

or if run just a file or spts in folder run:

:: python3 visualize.py -i <link_or_videofile>





