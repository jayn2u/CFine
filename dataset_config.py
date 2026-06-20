import os

DATASET_ROOT = '/mnt/data/lab_datasets'

CUHK_PEDES_DIR = os.path.join(DATASET_ROOT, 'CUHK-PEDES')
CUHK_PEDES_IMAGE_DIR = os.path.join(CUHK_PEDES_DIR, 'imgs')
CUHK_PEDES_JSON_ROOT = os.path.join(CUHK_PEDES_DIR, 'reid_raw.json')
CUHK_PEDES_ANNO_DIR = os.path.join(os.path.dirname(__file__), 'cuhkpedes', 'processed_data')

ICFG_PEDES_DIR = os.path.join(DATASET_ROOT, 'ICFG-PEDES')
ICFG_PEDES_IMAGE_DIR = os.path.join(ICFG_PEDES_DIR, 'imgs')
ICFG_PEDES_JSON_ROOT = os.path.join(ICFG_PEDES_DIR, 'ICFG-PEDES.json')

RSTPREID_DIR = os.path.join(DATASET_ROOT, 'RSTPReid')
RSTPREID_IMAGE_DIR = os.path.join(RSTPREID_DIR, 'imgs')
RSTPREID_JSON_ROOT = os.path.join(RSTPREID_DIR, 'data_captions.json')
