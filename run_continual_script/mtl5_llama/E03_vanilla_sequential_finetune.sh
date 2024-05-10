export CUDA_VISIBLE_DEVICES=0

bs=2
dropout=0.1
sl=512
lr=1e-3
epoch=40
gradient_accumulation_steps=4


A=agnews
C=amazon
D=yahoo
E=dbpedia
sql=512
tsql=50

for seed in 0 1 2
do
    task_list=${E}_${C}_${D}_${A}
    python3 src/run_continual_causal_llama2.py \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --task_list $task_list \
        --continual_learning \
        --multi_peft_modules False \
        --mpeft_enabled \
        --do_train \
        --do_eval \
        --do_predict \
        --n_train_per_class 16 \
        --n_val_per_class 32 \
        --early_stop \
        --early_stopping_patience 5 \
        --padding_strategy longest \
        --max_seq_length $sql \
        --max_target_length $tsql \
        --per_device_train_batch_size $bs \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --output_dir checkpoints_continual_mtl5_llama/order1_E03_vanilla_sequential_finetune \
        --overwrite_output_dir \
        --hidden_dropout_prob $dropout \
        --seed $seed \
        --save_strategy epoch \
        --evaluation_strategy epoch

    
    task_list=${E}_${C}_${A}_${D}
    python3 src/run_continual_causal_llama2.py \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --task_list $task_list \
        --continual_learning \
        --multi_peft_modules False \
        --mpeft_enabled \
        --do_train \
        --do_eval \
        --do_predict \
        --n_train_per_class 16 \
        --n_val_per_class 32 \
        --early_stop \
        --early_stopping_patience 5 \
        --padding_strategy longest \
        --max_seq_length $sql \
        --max_target_length $tsql \
        --per_device_train_batch_size $bs \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --output_dir checkpoints_continual_mtl5_llama/order2_E03_vanilla_sequential_finetune \
        --overwrite_output_dir \
        --hidden_dropout_prob $dropout \
        --seed $seed \
        --save_strategy epoch \
        --evaluation_strategy epoch


    task_list=${D}_${C}_${A}_${E}
    python3 src/run_continual_causal_llama2.py \
        --model_name_or_path meta-llama/Llama-2-7b-hf \
        --task_list $task_list \
        --continual_learning \
        --multi_peft_modules False \
        --mpeft_enabled \
        --do_train \
        --do_eval \
        --do_predict \
        --n_train_per_class 16 \
        --n_val_per_class 32 \
        --early_stop \
        --early_stopping_patience 5 \
        --padding_strategy longest \
        --max_seq_length $sql \
        --max_target_length $tsql \
        --per_device_train_batch_size $bs \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --output_dir checkpoints_continual_mtl5_llama/order3_E03_vanilla_sequential_finetune \
        --overwrite_output_dir \
        --hidden_dropout_prob $dropout \
        --seed $seed \
        --save_strategy epoch \
        --evaluation_strategy epoch
done
