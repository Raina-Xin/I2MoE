export device=0

CUDA_VISIBLE_DEVICES=$device python src/baseline/train_transformer.py \
    --data mmimdb \
    --gate None \
    --train_epochs 50 \
    --modality LI \
    --fusion_sparse False \
    --lr 0.0001 \
    --batch_size 64 \
    --hidden_dim 256 \
    --num_layers_fus 2 \
    --num_layers_enc 1 \
    --num_layers_pred 2 \
    --num_patches 8 \
    --num_experts 4 \
    --num_routers 1 \
    --top_k 2 \
    --num_heads 1 \
    --dropout 0.5 \
    --n_runs 3 \
    --gate_loss_weight 0.01 \
    --save False \
    --use_common_ids True
