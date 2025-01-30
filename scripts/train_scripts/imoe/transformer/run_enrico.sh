export device=0
for lr in 0.0001
do
for temperature_rw in 2
do
for hidden_dim_rw in 256
do
for num_layer_rw in 3
do
for interaction_loss_weight in 0.1 0.3 0.5
do
CUDA_VISIBLE_DEVICES=$device python src/imoe/train_transformer.py \
    --temperature_rw $temperature_rw \
    --hidden_dim_rw $hidden_dim_rw \
    --num_layer_rw $num_layer_rw \
    --interaction_loss_weight $interaction_loss_weight \
    --lr $lr \
    --data enrico \
    --gate None \
    --train_epochs 50 \
    --modality SW \
    --fusion_sparse False \
    --batch_size 32 \
    --hidden_dim 256 \
    --num_layers_fus 2 \
    --num_layers_enc 2 \
    --num_layers_pred 2 \
    --num_patches 8 \
    --num_experts 4 \
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
done
done
