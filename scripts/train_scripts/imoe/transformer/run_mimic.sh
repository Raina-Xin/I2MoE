export device=0
for lr in 1e-4
do
for temperature_rw in 1
do
for hidden_dim_rw in 256
do
for num_layer_rw in 3
do
for interaction_loss_weight in 0.01
do
for hidden_dim in 128
do
for num_patches in 4
do
for num_heads in 4
do
CUDA_VISIBLE_DEVICES=$device python src/imoe/train_transformer.py \
    --temperature_rw $temperature_rw \
    --hidden_dim_rw  $hidden_dim_rw \
    --num_layer_rw  $num_layer_rw \
    --interaction_loss_weight $interaction_loss_weight \
    --data mimic \
    --gate None \
    --train_epochs 30 \
    --modality LNC \
    --fusion_sparse False \
    --batch_size 32 \
    --hidden_dim $hidden_dim \
    --num_layers_fus 2 \
    --num_layers_enc 1 \
    --num_layers_pred 2 \
    --num_patches $num_patches \
    --num_experts 4 \
    --num_routers 1 \
    --top_k 2 \
    --num_heads $num_heads \
    --dropout 0.5 \
    --lr $lr \
    --n_runs 1 \
    --seed 1 \
    --gate_loss_weight 0.01 \
    --save False \
    --use_common_ids True
done
done
done
done
done
done
done
done