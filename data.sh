BASE_ROOT=.
DATASET_ROOT=/mnt/data/lab_datasets

IMAGE_ROOT=${DATASET_ROOT}/CUHK-PEDES/imgs
JSON_ROOT=${DATASET_ROOT}/CUHK-PEDES/reid_raw.json
OUT_ROOT=${BASE_ROOT}/cuhkpedes/processed_data

echo "Process CUHK-PEDES dataset and save it as pickle form"

uv run python ${BASE_ROOT}/datasets/preprocess.py \
        --img_root=${IMAGE_ROOT} \
        --json_root=${JSON_ROOT} \
        --out_root=${OUT_ROOT} \
        --min_word_count 3
