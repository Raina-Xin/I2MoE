export device=0
for lr in 1e-4
do
for temperature_rw in 2
do
for hidden_dim_rw in 256
do
for num_layer_rw in 1
do
for hidden_dim in 128
do
CUDA_VISIBLE_DEVICES=$device python src/ablation/transformer/transformer_zero_vector.py \
    --temperature_rw $temperature_rw \
    --hidden_dim_rw  $hidden_dim_rw \
    --num_layer_rw  $num_layer_rw \
    --lr $lr \
    --data mosi \
    --gate None \
    --train_epochs 50 \
    --modality TVA \
    --fusion_sparse False \
    --batch_size 32 \
    --hidden_dim $hidden_dim \
    --num_layers_fus 1 \
    --num_layers_enc 1 \
    --num_layers_pred 1 \
    --num_patches 4 \
    --num_experts 4 \
    --num_routers 1 \
    --top_k 2 \
    --num_heads 1 \
    --dropout 0.5 \
    --n_runs 3 \
    --gate_loss_weight 0.01 \
    --save False \
    --use_common_ids True
done
done
done
done
done