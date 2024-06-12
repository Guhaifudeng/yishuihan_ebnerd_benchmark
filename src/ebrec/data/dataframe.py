import hashlib
import inspect
import json
import pickle
from pathlib import Path
from typing import Callable

import pandas as pd
import polars as pl

# from const.path import CACHE_DIR
from ebrec.utils.log_util import init_logger
logging = init_logger(__name__)
UNKNOWN_USER_IDX = -1
# from const.mind import UNKNOWN_USER_IDX
#
#
# def _cache_dataframe(fn: Callable) -> Callable:
#     def read_df_function_wrapper(*args: tuple, **kwargs: dict) -> pl.DataFrame:
#         # inspect **kwargs
#         bound = inspect.signature(fn).bind(*args, **kwargs)
#         bound.apply_defaults()
#
#         d = bound.arguments
#         d["function_name"] = fn.__name__
#         d["path_to_tsv"] = str(bound.arguments["path_to_tsv"])
#
#         # if file exist in cache path, then load & return it.
#         cache_filename = hashlib.sha256(json.dumps(d).encode()).hexdigest()
#         cache_path = CACHE_DIR / f"{cache_filename}.pth"
#         if cache_path.exists() and (not d["clear_cache"]):
#             with open(cache_path, "rb") as f:
#                 df = pickle.load(f)
#             return df
#
#         df = fn(*args, **kwargs)
#
#         cache_path.parent.mkdir(parents=True, exist_ok=True)
#
#         with open(cache_path, "wb") as f:
#             pickle.dump(df, f)
#
#         return df
#
#     return read_df_function_wrapper





from ebrec.utils._constants import (
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_SUBTITLE_COL,
    DEFAULT_LABELS_COL,
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
    DEFAULT_CATEGORY_COL
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


def read_behavior_df(data_path: Path, history_size=30, mode = None) -> pl.DataFrame:
    def ebnerd_from_path(path: Path, history_size: int = 30) -> pl.DataFrame:
        logging.info('ebnerd_from_path start. {}'.format(path))
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
        logging.info(df_behaviors[0])
        logging.info('ebnerd_from_path finished. {}'.format(path))

        return df_behaviors
    logging.info('mode {}'.format(mode))
    if mode == 'train':
        behavior_df = (ebnerd_from_path(data_path, history_size=history_size).select(COLUMNS))
        behavior_df = (behavior_df.pipe(
            sampling_strategy_wu2019,
            npratio=4,
            shuffle=True,
            with_replacement=True,
            seed=123,).pipe(create_binary_labels_column))
    elif mode =='test':
        behavior_df = (ebnerd_from_path(data_path, history_size=history_size).select(TEST_COLUMNS))
    else:
        behavior_df = (ebnerd_from_path(data_path, history_size=history_size).select(COLUMNS))
        behavior_df = (behavior_df.pipe(create_binary_labels_column))

    # behavior_df = behavior_df.rename(
    #     {
    #         "column_1": "impression_id",
    #         "column_2": "user_id",
    #         "column_3": "time",
    #         "column_4": "history_str",
    #         "column_5": "impressions_str",
    #     }
    # )


    logging.info(behavior_df[0])
    return behavior_df

def read_news_df(path_dir: Path,  clear_cache: bool = False) -> pl.DataFrame:
    logging.info('start load articles {}'.format(path_dir))
    df_articles = pl.read_parquet(path_dir.joinpath("articles.parquet"))

    # TRANSFORMER_MODEL_NAME = "/home/dev/models/bce-embedding-base_v1"
    TEXT_COLUMNS_TO_USE = [DEFAULT_TITLE_COL, DEFAULT_SUBTITLE_COL,DEFAULT_CATEGORY_COL]
    MAX_TITLE_LENGTH = 30

    # => LOAD HUGGINGFACE:
    logging.info('articles process-1')
    # transformer_model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME)
    # transformer_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME, use_fast=True)

    # word2vec_embedding = get_transformers_word_embeddings(transformer_model)
    #
    logging.info('articles process-2')
    df_articles, cat_cal = concat_str_columns(df_articles, columns=TEXT_COLUMNS_TO_USE)
    df_articles = df_articles.select([DEFAULT_ARTICLE_ID_COL,cat_cal]).with_columns((pl.col(cat_cal).alias(DEFAULT_TITLE_COL))).drop([cat_cal])
    # df_articles, token_col_title = convert_text2encoding_with_transformers(
    #     df_articles, transformer_tokenizer, cat_cal, max_length=MAX_TITLE_LENGTH
    # )
    # # =>
    # print('articles process-3')
    # article_mapping = create_article_id_to_value_mapping(
    #     df=df_articles, value_col=token_col_title
    # )
    logging.info('end load articles')
    logging.info(df_articles[0])
    # print(df_articles.head(2))
    return df_articles

def create_user_ids_to_idx_map(train_behavior_df: pl.DataFrame, val_behavior_df: pl.DataFrame) -> dict[str, int]:
    user_ids_in_train_set = list(set(train_behavior_df[DEFAULT_USER_COL].to_list()))

    d: dict[str, int] = {}
    for i, user_id in enumerate(user_ids_in_train_set):
        d[user_id] = i + 1  # idx = 0

    user_ids_in_val_set = list(set(val_behavior_df[DEFAULT_USER_COL].to_list()))

    for i, user_id in enumerate(user_ids_in_val_set):
        if user_id not in d:
            d[user_id] = UNKNOWN_USER_IDX

    return d
