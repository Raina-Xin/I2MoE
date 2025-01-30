export device=0
for lr in 0.0001
do
for temperature_rw in 2
do
for hidden_dim_rw in 256
do
for num_layer_rw in 3
do
for hidden_dim in 64
do
for batch_size in 64 
do
for num_patches in 4 8 12 16
do
CUDA_VISIBLE_DEVICES=$device python src/ablation/transformer/transformer_no_interaction_loss.py \
    --temperature_rw $temperature_rw \
    --hidden_dim_rw $hidden_dim_rw \
    --num_layer_rw $num_layer_rw \
    --lr $lr \
    --data enrico \
    --gate None \
    --train_epochs 30 \
    --modality SW \
    --fusion_sparse False \
    --batch_size $batch_size \
    --hidden_dim $hidden_dim \
    --num_layers_fus 1 \
    --num_layers_enc 1 \
    --num_layers_pred 1 \
    --num_patches $num_patches \
    --num_experts 4 \
    --num_routers 1 \
    --top_k 2 \
    --num_heads 2 \
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
done
done

