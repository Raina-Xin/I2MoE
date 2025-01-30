export device=0


python src/baseline/train_m3care.py \
    --data adni \
    --modality IGCB 


python src/baseline/train_m3care.py \
    --data mimic \
    --modality LNC

python src/baseline/train_m3care.py \
    --data mosi \
    --modality TVA 


python src/baseline/train_m3care.py \
    --data enrico \
    --modality SW

python src/baseline/train_m3care.py \
    --data mmimdb \
    --modality LI
