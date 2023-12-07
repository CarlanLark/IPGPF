#! /bin/bash

NUM_GPUS=$1

train_args=(
    --task_name IGDEE-DuEE-CV
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
    --bert_size 768
    --hidden_size 768
    --ff_size 1024
    --template_text_type 'DuEE-fin-1'
    --use_bert False
    --use_ernie False
    --bert_model 'hlf/chinese-lert-large'
    --merge_entity False
    --use_child_tune False
    --child_tune_mode 'ChildTuning-F'
    --reserve_p 0.25
) 


python3 -m torch.distributed.launch \
--nproc_per_node ${NUM_GPUS} \
--master_port 9001 \
run_dee_task.py "${train_args[@]}"