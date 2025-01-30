export device=0


python src/baseline/train_lrtf.py \
    --data adni \
    --modality IGCB


python src/baseline/train_lrtf.py \
    --data mimic \
    --modality LNC

python src/baseline/train_lrtf.py \
    --data mosi \
    --modality TVA


python src/baseline/train_lrtf.py \
    --data enrico \
    --modality SW

python src/baseline/train_lrtf.py \
    --data mmimdb \
    --modality LI
