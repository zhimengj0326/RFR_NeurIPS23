#!/bin/bash
# alpha_values = 0. 0.3 1.0 5.0 10.0 50.0 100.0 500.0 1000.0 2000.0
# datasets = 'crimes' 'adult' 'ACS_I' 'ACS_E'

for dataset in 'ACS_E' ## 'ACS_I' 'ACS_E'
do
    for alpha_value in 0.0 10.0 50.0 100.0 500.0 ## 0.0 10.0 100.0 500.0 1000.0 1500.0 ##2000.0 3000.0
    do
        ## across state
        python -u main_dnn_adv.py --data $dataset --times 5 --n_epoch 100 --batch_size 200 --gpu 2  --model_save True\
                    --shift 'real' --real_shift 'states' --ori_state 'MI' --shift_state 'CA' \
                    --hyper_pent $alpha_value --model 'GBDT' &
        
        # ## across time
        # python -u main_dnn_adv.py --data $dataset --times 5 --n_epoch 100 --batch_size 200 --gpu 1 --model_save True \
        #             --shift 'real' --real_shift 'time' --ori_time '2016' --shift_time '2018' \
        #             --hyper_pent $alpha_value --model 'GBDT' &
        
    done
    wait
done

