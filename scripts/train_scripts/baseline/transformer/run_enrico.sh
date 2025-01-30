export device=0

CUDA_VISIBLE_DEVICES=$device python src/baseline/train_transformer.py \
    --data enrico \
    --gate None \
    --train_epochs 50 \
    --modality SW \
    --fusion_sparse False \
    --lr 1e-4 \
    --batch_size 64 \
    --hidden_dim 128 \
    --num_layers_fus 1 \
    --num_layers_enc 1 \
    --num_layers_pred 1 \
    --num_patches 4 \
    --num_experts 4 \
    --num_routers 1 \
    --top_k 2 \
    --num_heads 2 \
    --dropout 0.5 \
    --n_runs 3 \
    --gate_loss_weight 0.01 \
    --save False \
    --use_common_ids True
