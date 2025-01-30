export device=0

CUDA_VISIBLE_DEVICES=$device python src/baseline/train_moepp.py \
    --data mimic \
    --train_epochs 50 \
    --modality LNC \
    --lr 0.0001 \
    --batch_size 32 \
    --hidden_dim 64 \
    --num_layers_fus 2 \
    --num_layers_enc 1 \
    --num_layers_pred 2 \
    --num_patches 4 \
    --num_experts 4 \
    --num_heads 1 \
    --dropout 0.5 \
    --n_runs 3 \
    --save False \
    --use_common_ids True
