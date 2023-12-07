#! /bin/bash

train_args=(
    --task_name DuEE-fin
    --data_dir './Data'
    --exp_dir './Exps'
    --train_file_name 'dueefin_train_wo_tgg.json'
    --dev_file_name 'dueefin_dev_wo_tgg.json'
    --test_file_name 'dueefin_dev_wo_tgg.json'
    --event_type_num 13
    --train_batch_size 64
    --gradient_accumulation_steps 8
    --eval_batch_size 8
    --learning_rate 5e-5
    --num_train_epochs 200
    --dim_reduce False
    --bert_size 1024
    --hidden_size 1024
    --ff_size 1280
    --template_text_type 'DuEE-fin-1'
    --use_bert True
    --use_ernie False
    --bert_model 'hfl/chinese-lert-large'
    --merge_entity False
    --use_child_tune True
    --child_tune_mode 'ChildTuning-F'
    --reserve_p 0.25
    --debug_mode False
    --skip_train True
    --inference_only True
    --resume_cpt_epoch 192 # your best cpt
) 

CUDA_VISIBLE_DEVICES='0' python3 run_dee_task.py "${train_args[@]}"