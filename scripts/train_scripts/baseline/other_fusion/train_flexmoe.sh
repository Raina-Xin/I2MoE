export device=0


python src/baseline/train_flexmoe.py \
    --data adni \
    --modality IGCB


python src/baseline/train_flexmoe.py \
    --data mimic \
    --modality LNC

python src/baseline/train_flexmoe.py \
    --data mosi \
    --modality TVA

python src/baseline/train_flexmoe.py \
    --data mosi_regression \
    --modality TVA


python src/baseline/train_flexmoe.py \
    --data enrico \
    --modality SW

python src/baseline/train_flexmoe.py \
    --data mmimdb \
    --modality LI
