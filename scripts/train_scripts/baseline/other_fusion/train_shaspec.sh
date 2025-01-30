export device=0


python src/baseline/train_shaspec.py \
    --data adni \
    --modality IGCB 


python src/baseline/train_shaspec.py \
    --data mimic \
    --modality LNC

python src/baseline/train_shaspec.py \
    --data mosi \
    --modality TVA 


python src/baseline/train_shaspec.py \
    --data enrico \
    --modality SW

python src/baseline/train_shaspec.py \
    --data mmimdb \
    --modality LI
