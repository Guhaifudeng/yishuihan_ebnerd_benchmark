from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf
import polars as pl
import os

from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
)

from ebrec.utils._behaviors import (
    create_binary_labels_column,
    sampling_strategy_wu2019,
    add_known_user_column,
    add_prediction_scores,
    truncate_history,
)
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore
from ebrec.utils._articles import convert_text2encoding_with_transformers
from ebrec.utils._polars import concat_str_columns, slice_join_dataframes, split_df
from ebrec.utils._articles import create_article_id_to_value_mapping
from ebrec.utils._nlp import get_transformers_word_embeddings
from ebrec.utils._python import write_submission_file, rank_predictions_by_score

from ebrec.models.newsrec.dataloader import NRMSDataLoader, NRMSDataLoaderPretransform
from ebrec.models.newsrec.model_config import hparams_nrms
from ebrec.models.newsrec import NRMSModel

# python examples/00_quick_start/nrms_ebnerd.py


def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.DataFrame:
    print('ebnerd_from_path start. ', path)
    """
    Load ebnerd - function
    """
    df_history = (
        pl.scan_parquet(path.joinpath("history.parquet"))
        .select(DEFAULT_USER_COL, DEFAULT_HISTORY_ARTICLE_ID_COL)
        .pipe(
            truncate_history,
            column=DEFAULT_HISTORY_ARTICLE_ID_COL,
            history_size=history_size,
            padding_value=0,
            enable_warning=False,
        )
    )
    df_behaviors = (
        pl.scan_parquet(path.joinpath("behaviors.parquet"))
        .collect()
        .pipe(
            slice_join_dataframes,
            df2=df_history.collect(),
            on=DEFAULT_USER_COL,
            how="left",
        )
    )
    print('ebnerd_from_path finished. ' , path)
    return df_behaviors


PATH = Path("/home/dev/ebnerd-benchmark/data")
DATASPLIT = "ebnerd_large"
DATATEST = "ebnerd_testset"

COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
]
TEST_COLUMNS = [
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    # DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
]
HISTORY_SIZE = 30
FRACTION = 1.0

df_train = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "train"), history_size=HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=4,
        shuffle=True,
        with_replacement=True,
        seed=123,
    )
    .pipe(create_binary_labels_column)
    .sample(fraction=FRACTION)
)

df_val = (
    ebnerd_from_path(PATH.joinpath(DATASPLIT, "validation"), history_size=HISTORY_SIZE)
    .select(COLUMNS)
    .pipe(
        sampling_strategy_wu2019,
        npratio=4,
        shuffle=False,
        with_replacement=True,
        seed=123,
    )
    .pipe(create_binary_labels_column)
    .sample(fraction=0.3)
)
# df_train, df_val = split_df(df_training, 0.8)

df_test = (
    ebnerd_from_path(PATH.joinpath(DATATEST, "test"), history_size=HISTORY_SIZE)
    .select(TEST_COLUMNS)
    # .sample(fraction=0.01)
    # .pipe(create_binary_labels_column)
)
print('start load articles')
df_articles = pl.read_parquet(PATH.joinpath("articles.parquet"))

# df_train = df_train[:50]
# df_val = df_val[:50]
# df_test = df_test[:50]

# =>
os.environ["TOKENIZERS_PARALLELISM"] = "True"  # or "true"
TRANSFORMER_MODEL_NAME = "/home/dev/models/bce-embedding-base_v1"
TEXT_COLUMNS_TO_USE = [DEFAULT_TITLE_COL, DEFAULT_SUBTITLE_COL]
MAX_TITLE_LENGTH = 30

# => LOAD HUGGINGFACE:
print('articles process-1')
transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME,use_fast=True)

word2vec_embedding = get_transformers_word_embeddings(transformer_model)
#
print('articles process-2')
df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
df_articles, token_col_title = convert_text2encoding_with_transformers(
    df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
)
# =>
print('articles process-3')
article_mapping = create_article_id_to_value_mapping(
    df=df_articles, value_col=token_col_title
)
print('end load articles')
COLUMNS_DATALOAD = [
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_LABELS_COL,
]
TEST_COLUMNS_DATALOAD = [
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    # DEFAULT_LABELS_COL,
]
print('start build loader')
train_loader = NRMSDataLoaderPretransform(
    behaviors=df_train.select(COLUMNS_DATALOAD),
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=False,
    batch_size=32,
)
print('build loader-train')
val_loader = NRMSDataLoaderPretransform(
    behaviors=df_val.select(COLUMNS_DATALOAD),
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    eval_mode=True,
    batch_size=32,
)
print('build loader-val')
test_loader = NRMSDataLoaderPretransform(
    behaviors=df_test.select(TEST_COLUMNS_DATALOAD),
    article_dict=article_mapping,
    unknown_representation="zeros",
    history_column=DEFAULT_HISTORY_ARTICLE_ID_COL,
    test_mode=True,
    batch_size=32,
)
print('build loader-test')
print('finishe build loader')
MODEL_NAME = "NRMS"
LOG_DIR = f"downloads/runs/{MODEL_NAME}"
MODEL_WEIGHTS = f"downloads/data/state_dict/{MODEL_NAME}/weights"
print('finished build loader')
# => CALLBACKS
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
# modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(
#     filepath=MODEL_WEIGHTS, save_best_only=True, save_weights_only=True, verbose=1
# )
print('starting build model')
hparams_nrms.history_size = HISTORY_SIZE
model = NRMSModel(
    hparams=hparams_nrms,
    word2vec_embedding=word2vec_embedding,
    seed=42,
)
print('finished build model')
print('start training')
hist = model.model.fit(
    train_loader,
    validation_data=val_loader,
    epochs=2,
    callbacks=[tensorboard_callback, early_stopping],
)
print('finish training')
# =>
print('start val predict')
pred_validation = model.scorer.predict(val_loader)
df_val = add_prediction_scores(df_val, pred_validation.tolist())
df_val.head(2)
print('finish val predict')
# =>
metrics = MetricEvaluator(
    labels=df_val["labels"].to_list(),
    predictions=df_val["scores"].to_list(),
    metric_functions=[AucScore(), MrrScore(), NdcgScore(k=5), NdcgScore(k=10)],
)
print(metrics.evaluate())

print('start test predict')
pred_validation = model.scorer.predict(test_loader)
df_test = add_prediction_scores(df_test, pred_validation.tolist()).pipe(
    add_known_user_column, known_users=df_train[DEFAULT_USER_COL]
)
df_test.head(2)
print('finish test predict')

print('start save test predict')
df_test = df_test.with_columns(
    pl.col("scores")
    .map_elements(lambda x: list(rank_predictions_by_score(x)))
    .alias("ranked_scores")
)
df_test.head(2)

write_submission_file(
    impression_ids=df_test[DEFAULT_IMPRESSION_ID_COL],
    prediction_scores=df_test["ranked_scores"],
    path=Path("downloads/predictions.txt"),
)

print('done')