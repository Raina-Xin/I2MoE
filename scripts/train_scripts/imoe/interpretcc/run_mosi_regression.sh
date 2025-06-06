export device=0

for lr in 1e-4
do
for modality in TVA
do
for batch_size in 32
do
for hidden_dim in 128
do
for tau in 1.0
do
for threshold in 0.6
do
for num_layers_pred in 2 #1 2 3
do
for num_layers_fus in 2 #1 2 3
do
for num_layers_enc in 2 #1 2 3
do
for interaction_loss_weight in 0.001
do
for temperature_rw in 1.5 #4 8 10
do
for hidden_dim_rw in 256
do
for num_layer_rw in 2
do
CUDA_VISIBLE_DEVICES=$device python src/imoe/train_interpretcc.py \
    --data mosi_regression \
    --temperature_rw $temperature_rw \
    --hidden_dim_rw $hidden_dim_rw \
    --num_layer_rw $num_layer_rw \
    --train_epochs 50 \
    --modality $modality \
    --fusion_sparse False \
    --lr $lr \
    --batch_size $batch_size \
    --hidden_dim $hidden_dim \
    --warm_up_epochs $warm_up_epochs \
    --num_layers_enc $num_layers_enc \
    --num_layers_fus $num_layers_fus \
    --num_layers_pred $num_layers_pred \
    --tau $tau \
    --hard True \
    --threshold $threshold \
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
done