export CUDA_VISIBLE_DEVICES=0

bs=8
lr=2e-4
dropout=0.1
psl=8
epoch=5


A=am
B=dz
C=ha
D=ig
E=kr
F=ma
G=pcm
H=pt
I=sw
J=ts
K=twi
L=yo
for seed in 0 1 2
    do
        task_order=${A}_${B}_${C}_${D}_${E}_${F}_${G}_${H}_${I}_${J}_${K}_${L}
        python3 src/run_continual_mtl_afrisenti.py \
            --model_name_or_path Davlan/afro-xlmr-large \
            --cl_language_list $task_order \
            --continual_learning \
            --vanilla_continual_learning \
            --multi_peft_modules False \
            --do_train \
            --do_eval \
            --do_predict \
            --early_stop \
            --early_stopping_patience 5 \
            --max_seq_length 128 \
            --per_device_train_batch_size $bs \
            --learning_rate $lr \
            --num_train_epochs $epoch \
            --pre_seq_len $psl \
            --output_dir checkpoints_continual_afrisenti/order1_E03_vanilla_sequential_ft \
            --overwrite_output_dir \
            --hidden_dropout_prob $dropout \
            --seed $seed \
            --save_strategy epoch \
            --evaluation_strategy epoch \
            --prefix

        
        task_order=${F}_${G}_${E}_${H}_${D}_${I}_${C}_${J}_${B}_${K}_${A}_${L}
        python3 src/run_continual_mtl_afrisenti.py \
            --model_name_or_path Davlan/afro-xlmr-large \
            --cl_language_list $task_order \
            --continual_learning \
            --vanilla_continual_learning \
            --multi_peft_modules False \
            --do_train \
            --do_eval \
            --do_predict \
            --early_stop \
            --early_stopping_patience 5 \
            --max_seq_length 128 \
            --per_device_train_batch_size $bs \
            --learning_rate $lr \
            --num_train_epochs $epoch \
            --pre_seq_len $psl \
            --output_dir checkpoints_continual_afrisenti/order2_E03_vanilla_sequential_ft \
            --overwrite_output_dir \
            --hidden_dropout_prob $dropout \
            --seed $seed \
            --save_strategy epoch \
            --evaluation_strategy epoch \
            --prefix

        
        task_order=${A}_${B}_${C}_${F}_${D}_${E}_${I}_${J}_${K}_${L}_${G}_${H}
        python3 src/run_continual_mtl_afrisenti.py \
            --model_name_or_path Davlan/afro-xlmr-large \
            --cl_language_list $task_order \
            --continual_learning \
            --vanilla_continual_learning \
            --multi_peft_modules False \
            --do_train \
            --do_eval \
            --do_predict \
            --early_stop \
            --early_stopping_patience 5 \
            --max_seq_length 128 \
            --per_device_train_batch_size $bs \
            --learning_rate $lr \
            --num_train_epochs $epoch \
            --pre_seq_len $psl \
            --output_dir checkpoints_continual_afrisenti/order3_E03_vanilla_sequential_ft \
            --overwrite_output_dir \
            --hidden_dropout_prob $dropout \
            --seed $seed \
            --save_strategy epoch \
            --evaluation_strategy epoch \
            --prefix
    done
