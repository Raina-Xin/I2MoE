export device=0
for hidden_dim in 64
do
for tau in 1.0
do
for threshold in 0.5
do
for lr in 1e-4
do
CUDA_VISIBLE_DEVICES=$device python src/baseline/train_interpretcc.py \
    --data adni \
    --tau $tau \
    --hard True \
    --threshold $threshold \
    --lr $lr \
    --batch_size 32 \
    --hidden_dim $hidden_dim \
    --patch False \
    --dropout 0.5 \
    --train_epochs 1 \
    --modality IGCB \
    --n_runs 1 \
    --save False \
    --use_common_ids True
done
done
done
done