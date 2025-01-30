export device=0
for hidden_dim in 64 128 256
do
for batch_size in 64 
do
for num_patches in 4 8 12 16
do
CUDA_VISIBLE_DEVICES=$device python src/baseline/train_transformer.py \
    --data adni \
    --gate None \
    --train_epochs 50 \
    --modality IGCB \
    --fusion_sparse False \
    --lr 0.0001 \
    --batch_size $batch_size \
    --hidden_dim $hidden_dim \
    --num_layers_fus 2 \
    --num_layers_enc 1 \
    --num_layers_pred 1 \
    --num_patches $num_patches \
    --num_experts 8 \
    --num_routers 1 \
    --top_k 2 \
    --num_heads 4 \
    --dropout 0.5 \
    --n_runs 3 \
    --gate_loss_weight 0.01 \
    --save False \
    --use_common_ids True
done
done
done