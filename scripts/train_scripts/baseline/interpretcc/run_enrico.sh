export device=0
for hidden_dim in 256
do
for tau in 0.5
do
for threshold in 0.4
do
for lr in 1e-5
do
CUDA_VISIBLE_DEVICES=$device python src/baseline/train_interpretcc.py \
    --data enrico \
    --tau $tau \
    --hard True \
    --threshold $threshold \
    --lr $lr \
    --batch_size 32 \
    --hidden_dim $hidden_dim \
    --patch False \
    --dropout 0.5 \
    --train_epochs 50 \
    --modality SW \
    --n_runs 3 \
    --save False \
    --use_common_ids True
done
done
done
done