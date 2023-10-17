#!/bin/bash

# alpha_values = 0.0 0.1 0.5 1.0 3.0 5.0 10.0 30.0 50.0 100.0 200.0
# datasets = 'ACS_I' 'ACS_E'

for dataset in 'ACS_I' 'ACS_E'
do
    for alpha_value in 0.0 0.1 0.5 1.0 5.0 10.0 30.0 50.0 100.0
    do
        # python -u main_dnn_fcr.py --data $dataset --times 5 --n_epoch 100 --batch_size 200 --gpu 1  \
        #             --shift 'real' --real_shift 'states' --ori_state 'MI' --shift_state 'CA' --hyper_pent $alpha_value &
        # python -u main_dnn_fcr.py --data $dataset --times 5 --n_epoch 100 --batch_size 200 --gpu 3  \
        #              --shift 'real' --real_shift 'time' --ori_time '2016' --shift_time '2018' --hyper_pent $alpha_value &

        python -u main_dnn_fcr.py --data $dataset --times 5 --n_epoch 100 --batch_size 200 --gpu 2  \
                    --shift 'real_iid' --hyper_pent $alpha_value &
    done
    wait
done
