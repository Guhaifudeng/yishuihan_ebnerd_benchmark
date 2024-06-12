import random
from typing import Callable

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from ebrec.utils.log_util import init_logger
from ebrec.utils.list import uniq

logging = init_logger(__name__)

from ebrec.utils._constants import EMPTY_IMPRESSION_IDX, EMPTY_NEWS_ID

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
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_USER_COL,
    DEFAULT_CATEGORY_COL
)

class EbrecTrainDataset(Dataset):
    def __init__(
        self,
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        user_ids_to_idx_map: dict[str, int],
        batch_transform_texts: Callable[[list[str]], torch.Tensor],
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        self.behavior_df: pl.DataFrame = behavior_df
        self.news_df: pl.DataFrame = news_df
        self.batch_transform_texts: Callable[[list[str]], torch.Tensor] = batch_transform_texts
        self.device: torch.device = device

        # self.behavior_df = self.behavior_df.with_columns(
        #     [
        #         pl.col("impressions")
        #         .apply(lambda v: [i for i, imp_item in enumerate(v) if imp_item["clicked"] == 1])
        #         .alias("clicked_idxes"),
        #         pl.col("impressions")
        #         .apply(lambda v: [i for i, imp_item in enumerate(v) if imp_item["clicked"] == 0])
        #         .alias("non_clicked_idxes"),
        #     ]
        # )

        self.__news_id_to_title_map: dict[str, str] = {
            self.news_df[i][DEFAULT_ARTICLE_ID_COL].item(): self.news_df[i][DEFAULT_TITLE_COL].item() for i in range(len(self.news_df))
        }
        self.__news_id_to_title_map[EMPTY_NEWS_ID] = ""
        self.__user_ids_to_idx_map = user_ids_to_idx_map

    def __getitem__(self, behavior_idx: int) -> dict:
        """
        Returns:
            torch.Tensor: history_news
            torch.Tensor: candidate_news
            torch.Tensor: labels
        """
        behavior_item = self.behavior_df[behavior_idx]
        history_news_ids = behavior_item[DEFAULT_HISTORY_ARTICLE_ID_COL].to_list()[0]
        candidate_news_ids = behavior_item[DEFAULT_INVIEW_ARTICLES_COL].to_list()[0]
        labels = behavior_item[DEFAULT_LABELS_COL].to_list()[0]
        # News ID to News Title
        candidate_news_titles, history_news_titles = (
            [self.__news_id_to_title_map[news_id] for news_id in candidate_news_ids],
            [self.__news_id_to_title_map[news_id] for news_id in history_news_ids],
        )

        # Convert to Tensor
        candidate_news_tensor, history_news_tensor = (
            self.batch_transform_texts(candidate_news_titles),
            self.batch_transform_texts(history_news_titles),
        )
        labels_tensor = torch.Tensor(labels).argmax()

        # user_id
        user_id = behavior_item[DEFAULT_USER_COL].to_list()[0]

        return {
            "news_histories": history_news_tensor,
            "candidate_news": candidate_news_tensor,
            "user_id": user_id,
            "target": labels_tensor,
        }

    def __len__(self) -> int:
        return len(self.behavior_df)

    def get_user_num(self, only_train: bool = False) -> int:
        if only_train:
            return len(uniq(self.behavior_df["user_id"].to_list()))
        return len(self.__user_ids_to_idx_map)

class EbrecValDataset(Dataset):
    def __init__(
        self,
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        user_ids_to_idx_map: dict[str, int],
        batch_transform_texts: Callable[[list[str]], torch.Tensor],
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        self.behavior_df: pl.DataFrame = behavior_df
        self.news_df: pl.DataFrame = news_df
        self.batch_transform_texts: Callable[[list[str]], torch.Tensor] = batch_transform_texts
        self.device: torch.device = device

        self.__news_id_to_title_map: dict[str, str] = {
            self.news_df[i][DEFAULT_ARTICLE_ID_COL].item(): self.news_df[i][DEFAULT_TITLE_COL].item() for i in
            range(len(self.news_df))
        }
        self.__news_id_to_title_map[EMPTY_NEWS_ID] = ""
        self.__user_ids_to_idx_map = user_ids_to_idx_map

    def __getitem__(self, behavior_idx: int) -> dict:
        """
        Returns:
            torch.Tensor: history_news
            torch.Tensor: candidate_news
            torch.Tensor: one-hot labels
        """
        behavior_item = self.behavior_df[behavior_idx]
        history_news_ids = behavior_item[DEFAULT_HISTORY_ARTICLE_ID_COL].to_list()[0]
        candidate_news_ids = behavior_item[DEFAULT_INVIEW_ARTICLES_COL].to_list()[0]
        labels = behavior_item[DEFAULT_LABELS_COL].to_list()[0]

        # News ID to News Title
        candidate_news_titles, history_news_titles = (
            [self.__news_id_to_title_map[news_id] for news_id in candidate_news_ids],
            [self.__news_id_to_title_map[news_id] for news_id in history_news_ids],
        )

        # Convert to Tensor
        candidate_news_tensor, history_news_tensor = (
            self.batch_transform_texts(candidate_news_titles),
            self.batch_transform_texts(history_news_titles),
        )
        one_hot_label_tensor = torch.Tensor(labels)

        # user_id
        user_id = behavior_item[DEFAULT_USER_COL].to_list()[0]

        return {
            "news_histories": history_news_tensor,
            "candidate_news": candidate_news_tensor,
            "user_id": user_id,
            "target": one_hot_label_tensor,
            # "labels":labels
        }

    def __len__(self) -> int:
        return len(self.behavior_df)



class EbrecTestDataset(Dataset):
    def __init__(
        self,
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        user_ids_to_idx_map: dict[str, int],
        batch_transform_texts: Callable[[list[str]], torch.Tensor],
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        self.behavior_df: pl.DataFrame = behavior_df
        self.news_df: pl.DataFrame = news_df
        self.batch_transform_texts: Callable[[list[str]], torch.Tensor] = batch_transform_texts
        self.device: torch.device = device

        self.__news_id_to_title_map: dict[str, str] = {
            self.news_df[i][DEFAULT_ARTICLE_ID_COL].item(): self.news_df[i][DEFAULT_TITLE_COL].item() for i in
            range(len(self.news_df))
        }
        self.__news_id_to_title_map[EMPTY_NEWS_ID] = ""
        self.__user_ids_to_idx_map = user_ids_to_idx_map

    def __getitem__(self, behavior_idx: int) -> dict:
        """
        Returns:
            torch.Tensor: history_news
            torch.Tensor: candidate_news
            torch.Tensor: one-hot labels
        """
        behavior_item = self.behavior_df[behavior_idx]
        history_news_ids = behavior_item[DEFAULT_HISTORY_ARTICLE_ID_COL].to_list()[0]
        candidate_news_ids = behavior_item[DEFAULT_INVIEW_ARTICLES_COL].to_list()[0]
        # labels = behavior_item[DEFAULT_LABELS_COL]

        # News ID to News Title
        candidate_news_titles, history_news_titles = (
            [self.__news_id_to_title_map[news_id] for news_id in candidate_news_ids],
            [self.__news_id_to_title_map[news_id] for news_id in history_news_ids],
        )

        # Convert to Tensor
        candidate_news_tensor, history_news_tensor = (
            self.batch_transform_texts(candidate_news_titles),
            self.batch_transform_texts(history_news_titles),
        )
        # one_hot_label_tensor = torch.Tensor(labels)

        # user_id
        user_id = behavior_item[DEFAULT_USER_COL].to_list()[0]

        return {
            "news_histories": history_news_tensor,
            "candidate_news": candidate_news_tensor,
            "user_id": user_id,
            #"target": one_hot_label_tensor,
        }

    def __len__(self) -> int:
        return len(self.behavior_df)


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from ebrec.utils.random_seed import set_random_seed

    from ebrec.data.dataframe import read_behavior_df, read_news_df, create_user_ids_to_idx_map

    set_random_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("/home/badou/openapp/passage_retrieval/BCE/bce-embedding-base_v1")
    #
    # # logging.info()
    def transform(texts: list[str]) -> torch.Tensor:
        return tokenizer(texts, return_tensors="pt", max_length=64, padding="max_length", truncation=True)["input_ids"]


    Ebrec_SMALL_TRAIN_DATASET_DIR = '/data/badou/new_data/ebnerd/ebnerd_small/train'
    Ebrec_SMALL_VAL_DATASET_DIR =  '/data/badou/new_data/ebnerd/ebnerd_small/validation'
    Ebrec_SMALL_News_DATASET_DIR = '/data/badou/new_data/ebnerd/ebnerd_small/'
    logging.info("Load Data")
    train_behavior_df = read_behavior_df(Path(Ebrec_SMALL_TRAIN_DATASET_DIR),mode= 'train')
    news_df = read_news_df(Path(Ebrec_SMALL_News_DATASET_DIR))
    # train_behavior_df, train_news_df = (
    #     read_behavior_df(Ebrec_SMALL_TRAIN_DATASET_DIR),
    #     read_news_df(Ebrec_SMALL_TRAIN_DATASET_DIR / "news.tsv"),
    # )
    # val_behavior_df = read_behavior_df(Ebrec_SMALL_VAL_DATASET_DIR / "behaviors.tsv")
    #
    val_behavior_df = read_behavior_df(Path(Ebrec_SMALL_VAL_DATASET_DIR),mode='test')


    user_ids_to_idx_map = create_user_ids_to_idx_map(train_behavior_df, val_behavior_df)
    #
    # logging.info("Init EbrecTrainDataset")
    train_dataset = EbrecTrainDataset(
        train_behavior_df,
        news_df,
        user_ids_to_idx_map,
        batch_transform_texts=transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    logging.info("Start Iteration")
    for batch in train_dataloader:
        logging.info(f"{batch}")
        break
