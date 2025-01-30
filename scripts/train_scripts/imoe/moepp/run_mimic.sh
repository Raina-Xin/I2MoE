export device=0

for lr in 1e-4 #1e-5
do
for modality in LNC
do
for batch_size in 32
do
for hidden_dim in 64 128
do
for num_patches in 4
do
for num_experts in 4
do
for num_layers_pred in 2 #1 2 3
do
for num_layers_fus in 2 #1 2 3
do
for num_layers_enc in 2 #1 2 3
do
for num_heads in 4 #1 2 3 4
do
for interaction_loss_weight in 0.1 0.2 0.3
do
for temperature_rw in 1.5 2 #4 8 10
do
for hidden_dim_rw in 256
do
for num_layer_rw in 2
do
CUDA_VISIBLE_DEVICES=$device python src/imoe/train_moepp.py \
    --temperature_rw $temperature_rw \
    --hidden_dim_rw $hidden_dim_rw \
    --num_layer_rw $num_layer_rw \
    --data mimic \
    --train_epochs 40 \
    --modality $modality \
    --fusion_sparse False \
    --lr $lr \
    --batch_size $batch_size \
    --hidden_dim $hidden_dim \
    --warm_up_epochs $warm_up_epochs \
    --num_layers_enc $num_layers_enc \
    --num_layers_fus $num_layers_fus \
    --num_layers_pred $num_layers_pred \
    --num_patches $num_patches \
    --num_experts $num_experts \
    --num_heads $num_heads \
    --dropout 0.5 \
    --n_runs 3 \
    --interaction_loss_weight $interaction_loss_weight \
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
done
done
done
done
done
done