#!/bin/bash

# combined projections and angles increase
#python proj_ang_comb.py

# gray value experiments
python gray_values_exp.py -type 0 -exp_name "gray_rand"
python gray_values_exp.py -type 1 -exp_name "gray_over"
python gray_values_exp.py -type 2 -exp_name "gray_under"
