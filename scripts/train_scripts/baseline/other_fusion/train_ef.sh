export device=0


python src/baseline/train_ef.py \
    --data adni \
    --modality IGCB


python src/baseline/train_ef.py \
    --data mimic \
    --modality LNC

python src/baseline/train_ef.py \
    --data mosi_regression \
    --modality TVA


python src/baseline/train_ef.py \
    --data enrico \
    --modality SW

python src/baseline/train_ef.py \
    --data mmimdb \
    --modality LI
