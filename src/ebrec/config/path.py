import pathlib

PROJECT_ROOT = (pathlib.Path(__file__) / ".." / ".." / "..").resolve()

OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "model"
LOG_OUTPUT_DIR = OUTPUT_DIR / "log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# DATASET_DIR = pathlib.Path('/data/badou/new_data/ebnerd')
DATASET_DIR = pathlib.Path('/home/dev/ebnerd-benchmark/data')
#/home/dev/ebnerd-benchmark/data


CACHE_DIR = OUTPUT_DIR / ".cache"


EBREC_SMALL_DATASET_DIR = DATASET_DIR / "ebnerd_small"
EBREC_SMALL_VAL_DATASET_DIR = EBREC_SMALL_DATASET_DIR / "validation"
EBREC_SMALL_TRAIN_DATASET_DIR = EBREC_SMALL_DATASET_DIR / "train"

EBREC_TEST_DATASET_DIR = DATASET_DIR / "ebnerd_testset/test"


EBREC_LARGE_DATASET_DIR = DATASET_DIR / "ebnerd_large"
EBREC_LARGE_VAL_DATASET_DIR = EBREC_LARGE_DATASET_DIR / "validation"

EBREC_LARGE_TRAIN_DATASET_DIR = EBREC_LARGE_DATASET_DIR / "train"


EBREC_GENERATED_DATASET_DIR = DATASET_DIR / "generated"
EBREC_GENERATED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
