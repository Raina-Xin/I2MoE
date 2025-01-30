export device=0

CUDA_VISIBLE_DEVICES=$device python src/baseline/train_moepp.py \
    --data adni \
    --train_epochs 1 \
    --modality IGCB \
    --lr 0.0001 \
    --batch_size 32 \
    --hidden_dim 64 \
    --num_layers_fus 2 \
    --num_layers_enc 1 \
    --num_layers_pred 2 \
    --num_patches 16 \
    --num_experts 8 \
    --num_heads 4 \
    --dropout 0.5 \
    --n_runs 1 \
    --save False \
    --use_common_ids True
