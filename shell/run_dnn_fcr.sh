#!/bin/bash

# alpha_values = 0.0 0.1 0.5 1.0 3.0 5.0 10.0 30.0 50.0 100.0 200.0
# datasets = 'crimes' 'adult' 'ACS_I' 'ACS_E'
for dataset in 'adult' ##'ACS_I' 'ACS_E'
do
    for alpha_value in 0.0 0.1 0.5 1.0 3.0 5.0 10.0 30.0 50.0 
    do
        python -u main_dnn_fcr.py --data $dataset --times 5 --n_epoch 100 --batch_size 200 --gpu 0  \
                    --alpha 1.0 --beta 2.0 --hyper_pent $alpha_value &
        python -u main_dnn_fcr.py --data $dataset --times 5 --n_epoch 100 --batch_size 200 --gpu 1  \
                    --alpha 1.5 --beta 3.0 --hyper_pent $alpha_value &
        python -u main_dnn_fcr.py --data $dataset --times 5 --n_epoch 100 --batch_size 200 --gpu 2  \
                    --alpha 3.0 --beta 6.0 --hyper_pent $alpha_value &
    done
    wait
done