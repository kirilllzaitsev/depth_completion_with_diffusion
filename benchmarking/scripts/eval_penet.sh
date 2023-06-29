#!/bin/bash

cd /media/master/wext/msc_studies/second_semester/research_project/related_work/PENet_ICRA2021 || exit 1

python main.py -b 1 -n pe --evaluate models/pe.pth.tar