export device=0

for lr in 1e-4 # 1e-5
do
for top_k in 1
do
for modality in TVA
do
for gate in SwitchGate
do
for batch_size in 32
do
for hidden_dim in 128
do
for num_patches in 16
do
for num_experts in 4
do
for num_layers_pred in 1 #1 2 3
do
for num_layers_fus in 1 #1 2 3
do
for num_layers_enc in 1 #1 2 3
do
for num_routers in 1
do
for num_heads in 4
do
for interaction_loss_weight in 0.001
do
for gate_loss_weight in 0.01 
do
for temperature_rw in 2 #4 8 10
do
for temperature_rw in 2
do
for hidden_dim_rw in 128
do
for num_layer_rw in 1
do
for train_epochs in 30 40 50
do
CUDA_VISIBLE_DEVICES=$device python src/imoe/train_switchgate.py \
    --temperature_rw $temperature_rw \
    --hidden_dim_rw $hidden_dim_rw \
    --num_layer_rw $num_layer_rw \
    --data mosi_regression \
    --gate $gate \
    --train_epochs $train_epochs \
    --modality $modality \
    --fusion_sparse True \
    --lr $lr \
    --batch_size $batch_size \
    --hidden_dim $hidden_dim \
    --warm_up_epochs $warm_up_epochs \
    --num_layers_enc $num_layers_enc \
    --num_layers_fus $num_layers_fus \
    --num_layers_pred $num_layers_pred \
    --num_patches $num_patches \
    --num_experts $num_experts \
    --num_routers $num_routers \
    --top_k $top_k \
    --num_heads $num_heads \
    --dropout 0.5 \
    --n_runs 3 \
    --gate_loss_weight $gate_loss_weight \
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
done
done
done
done
done
done