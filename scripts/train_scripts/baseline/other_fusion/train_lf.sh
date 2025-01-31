export device=0


python src/baseline/train_lf.py \
    --data adni \
    --modality IGCB


python src/baseline/train_lf.py \
    --data mimic \
    --modality LNC

python src/baseline/train_lf.py \
    --data mosi_regression \
    --modality TVA


python src/baseline/train_lf.py \
    --data enrico \
    --modality SW

python src/baseline/train_lf.py \
    --data mmimdb \
    --modality LI
