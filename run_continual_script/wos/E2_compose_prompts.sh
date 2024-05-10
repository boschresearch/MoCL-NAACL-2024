export CUDA_VISIBLE_DEVICES=0

bs=32
lr=3e-2
dropout=0.1
sl=256
psl=16
epoch=40
early_stopping_patience=5

for seed in 0 1 2
    do
        python3 src/run_continual_mtl_wos.py \
            --model_name_or_path bert-base-uncased \
            --mtl_task_list 0_1_2_3_4_5_6 \
            --continual_learning \
            --compose_prompts \
            --task_specific_classifier \
            --matching_loss_v2 \
            --do_train \
            --do_eval \
            --do_predict \
            --early_stop \
            --early_stopping_patience ${early_stopping_patience} \
            --max_seq_length $sl \
            --per_device_train_batch_size $bs \
            --learning_rate $lr \
            --num_train_epochs $epoch \
            --pre_seq_len $psl \
            --output_dir checkpoints_continual_mtl_wos/E2_compose_prompts \
            --overwrite_output_dir \
            --hidden_dropout_prob $dropout \
            --seed $seed \
            --save_strategy epoch \
            --evaluation_strategy epoch \
            --prefix
    done
