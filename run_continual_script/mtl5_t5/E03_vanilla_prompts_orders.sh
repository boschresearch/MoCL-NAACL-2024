export CUDA_VISIBLE_DEVICES=0

bs=8
dropout=0.1
sl=512
psl=50
gpsl=10
epoch=40

A=agnews
C=amazon
D=yahoo
E=dbpedia
lr_A=2e-2
lr_C=5e-2
lr_D=2e-2
lr_E=5e-2

for seed in 0 1 2
    do
        task_list=${E}_${C}_${D}_${A}
        lr_list=${lr_E}_${lr_C}_${lr_D}_${lr_A}
        python3 src/run_continual_mtl5_t5.py \
            --model_name_or_path google-t5/t5-large \
            --mtl_task_list $task_list \
            --continual_learning \
            --vanilla_continual_learning \
            --multi_peft_modules False \
            --do_train \
            --do_eval \
            --do_predict \
            --max_train_samples 16 \
            --max_eval_samples 200 \
            --early_stop \
            --early_stopping_patience 5 \
            --max_seq_length $sl \
            --per_device_train_batch_size $bs \
            --learning_rate_list $lr_list \
            --num_train_epochs $epoch \
            --pre_seq_len $psl \
            --output_dir checkpoints_continual_mtl5_t5/order1_E03_vanilla_prompts \
            --overwrite_output_dir \
            --hidden_dropout_prob $dropout \
            --seed $seed \
            --save_strategy epoch \
            --evaluation_strategy epoch \
            --prefix


        task_list=${E}_${C}_${A}_${D}
        lr_list=${lr_E}_${lr_C}_${lr_A}_${lr_D}
        python3 src/run_continual_mtl5_t5.py \
            --model_name_or_path /fs/scratch/rb_bd_dlp_rng-dl01_cr_AIM_employees/model_cache/t5-large/ \
            --mtl_task_list $task_list \
            --continual_learning \
            --vanilla_continual_learning \
            --multi_peft_modules False \
            --do_train \
            --do_eval \
            --do_predict \
            --max_train_samples 16 \
            --max_eval_samples 200 \
            --early_stop \
            --early_stopping_patience 5 \
            --max_seq_length $sl \
            --per_device_train_batch_size $bs \
            --learning_rate_list $lr_list \
            --num_train_epochs $epoch \
            --pre_seq_len $psl \
            --output_dir checkpoints_continual_mtl5_t5/order2_E03_vanilla_prompts \
            --overwrite_output_dir \
            --hidden_dropout_prob $dropout \
            --seed $seed \
            --save_strategy epoch \
            --evaluation_strategy epoch \
            --prefix


        task_list=${D}_${C}_${A}_${E}
        lr_list=${lr_D}_${lr_C}_${lr_A}_${lr_E}
        python3 src/run_continual_mtl5_t5.py \
            --model_name_or_path /fs/scratch/rb_bd_dlp_rng-dl01_cr_AIM_employees/model_cache/t5-large/ \
            --mtl_task_list $task_list \
            --continual_learning \
            --vanilla_continual_learning \
            --multi_peft_modules False \
            --do_train \
            --do_eval \
            --do_predict \
            --max_train_samples 16 \
            --max_eval_samples 200 \
            --early_stop \
            --early_stopping_patience 5 \
            --max_seq_length $sl \
            --per_device_train_batch_size $bs \
            --learning_rate_list $lr_list \
            --num_train_epochs $epoch \
            --pre_seq_len $psl \
            --output_dir checkpoints_continual_mtl5_t5/order3_E03_vanilla_prompts \
            --overwrite_output_dir \
            --hidden_dropout_prob $dropout \
            --seed $seed \
            --save_strategy epoch \
            --evaluation_strategy epoch \
            --prefix
    done
