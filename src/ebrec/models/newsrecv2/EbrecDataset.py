import random
from typing import Callable

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from utils.list import uniq

from ebrec.utils._constants import EMPTY_IMPRESSION_IDX, EMPTY_NEWS_ID

from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import tensorflow as tf
import polars as pl
import os

class EbrecTrainDataset(Dataset):
    def __init__(
        self,
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        user_ids_to_idx_map: dict[str, int],
        batch_transform_texts: Callable[[list[str]], torch.Tensor],
        npratio: int,
        history_size: int,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        self.behavior_df: pl.DataFrame = behavior_df
        self.news_df: pl.DataFrame = news_df
        self.batch_transform_texts: Callable[[list[str]], torch.Tensor] = batch_transform_texts
        self.npratio: int = npratio
        self.history_size: int = history_size
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
            self.news_df[i]["news_id"].item(): self.news_df[i]["title"].item() for i in range(len(self.news_df))
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
        # Extract Values
        behavior_item = self.behavior_df[behavior_idx]

        history: list[str] = (
            behavior_item["history"].to_list()[0] if behavior_item["history"].to_list()[0] is not None else []
        )  # TODO: Consider Remove if "history" is None
        poss_idxes, neg_idxes = (
            behavior_item["clicked_idxes"].to_list()[0],
            behavior_item["non_clicked_idxes"].to_list()[0],
        )
        EMPTY_IMPRESSION = {"news_id": EMPTY_NEWS_ID, "clicked": 0}
        impressions = np.array(
            behavior_item["impressions"].to_list()[0] + [EMPTY_IMPRESSION]
        )  # NOTE: EMPTY_IMPRESSION_IDX = -1なので最後尾に追加する。

        poss_idxes, neg_idxes = (
            behavior_item["clicked_idxes"].to_list()[0],
            behavior_item["non_clicked_idxes"].to_list()[0],
        )

        # Sampling Positive(clicked) & Negative(non-clicked) Sample
        sample_poss_idxes, sample_neg_idxes = (
            random.sample(poss_idxes, 1),
            self.__sampling_negative(neg_idxes, self.npratio),
        )

        sample_impression_idxes = sample_poss_idxes + sample_neg_idxes
        random.shuffle(sample_impression_idxes)

        sample_impressions = impressions[sample_impression_idxes]

        # Extract candidate_news & history_news based on sample idxes
        candidate_news_ids = [imp_item["news_id"] for imp_item in sample_impressions]
        labels = [imp_item["clicked"] for imp_item in sample_impressions]
        history_news_ids = history[: self.history_size]  # TODO: diverse
        if len(history) < self.history_size:
            history_news_ids = history_news_ids + [EMPTY_NEWS_ID] * (self.history_size - len(history))

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
        user_id = self.__user_ids_to_idx_map[behavior_item["user_id"].to_list()[0]]

        # ref: NRMS.forward in src/recommendation/nrms/NRMS.py
        return {
            "news_histories": history_news_tensor,
            "candidate_news": candidate_news_tensor,
            "user_id": user_id,
            "target": labels_tensor,
        }

    def __len__(self) -> int:
        return len(self.behavior_df)

    def __sampling_negative(self, neg_idxes: list[int], npratio: int) -> list[int]:
        if len(neg_idxes) < npratio:
            return neg_idxes + [EMPTY_IMPRESSION_IDX] * (npratio - len(neg_idxes))

        return random.sample(neg_idxes, self.npratio)

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
        history_size: int,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        self.behavior_df: pl.DataFrame = behavior_df
        self.news_df: pl.DataFrame = news_df
        self.batch_transform_texts: Callable[[list[str]], torch.Tensor] = batch_transform_texts
        self.history_size: int = history_size
        self.device: torch.device = device

        self.__news_id_to_title_map: dict[str, str] = {
            self.news_df[i]["news_id"].item(): self.news_df[i]["title"].item() for i in range(len(self.news_df))
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
        # Extract Values
        behavior_item = self.behavior_df[behavior_idx]

        history: list[str] = (
            behavior_item["history"].to_list()[0] if behavior_item["history"].to_list()[0] is not None else []
        )  # TODO: Consider Remove if "history" is None
        EMPTY_IMPRESSION = {"news_id": EMPTY_NEWS_ID, "clicked": 0}
        impressions = np.array(
            behavior_item["impressions"].to_list()[0] + [EMPTY_IMPRESSION]
        )  # NOTE: EMPTY_IMPRESSION_IDX = -1なので最後尾に追加する。

        # Extract candidate_news & history_news based on sample idxes
        candidate_news_ids = [imp_item["news_id"] for imp_item in impressions]
        labels = [imp_item["clicked"] for imp_item in impressions]
        history_news_ids = history[: self.history_size]  # TODO: diverse
        if len(history) < self.history_size:
            history_news_ids = history_news_ids + [EMPTY_NEWS_ID] * (self.history_size - len(history))

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
        user_id = self.__user_ids_to_idx_map[behavior_item["user_id"].to_list()[0]]

        return {
            "news_histories": history_news_tensor,
            "candidate_news": candidate_news_tensor,
            "user_id": user_id,
            "target": one_hot_label_tensor,
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
        history_size: int,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        self.behavior_df: pl.DataFrame = behavior_df
        self.news_df: pl.DataFrame = news_df
        self.batch_transform_texts: Callable[[list[str]], torch.Tensor] = batch_transform_texts
        self.history_size: int = history_size
        self.device: torch.device = device

        self.__news_id_to_title_map: dict[str, str] = {
            self.news_df[i]["news_id"].item(): self.news_df[i]["title"].item() for i in range(len(self.news_df))
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
        # Extract Values
        behavior_item = self.behavior_df[behavior_idx]

        history: list[str] = (
            behavior_item["history"].to_list()[0] if behavior_item["history"].to_list()[0] is not None else []
        )  # TODO: Consider Remove if "history" is None
        EMPTY_IMPRESSION = {"news_id": EMPTY_NEWS_ID, "clicked": 0}
        impressions = np.array(
            behavior_item["impressions"].to_list()[0] + [EMPTY_IMPRESSION]
        )  # NOTE: EMPTY_IMPRESSION_IDX = -1

        # Extract candidate_news & history_news based on sample idxes
        candidate_news_ids = [imp_item["news_id"] for imp_item in impressions]
        #labels = [imp_item["clicked"] for imp_item in impressions]
        history_news_ids = history[: self.history_size]  # TODO: diverse
        if len(history) < self.history_size:
            history_news_ids = history_news_ids + [EMPTY_NEWS_ID] * (self.history_size - len(history))

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
        #one_hot_label_tensor = torch.Tensor(labels)

        # user_id
        user_id = self.__user_ids_to_idx_map[behavior_item["user_id"].to_list()[0]]

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
    from utils.logger import logging
    from utils.random_seed import set_random_seed

    from ebrec.models.newsrecv2.dataframe import read_behavior_df, read_news_df, create_user_ids_to_idx_map

    set_random_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("/home/badou/openapp/passage_retrieval/BCE/bce-embedding-base_v1")
    #
    # # logging.info()
    def transform(texts: list[str]) -> torch.Tensor:
        return tokenizer(texts, return_tensors="pt", max_length=64, padding="max_length", truncation=True)["input_ids"]


    Ebrec_SMALL_TRAIN_DATASET_DIR = '/data/badou/new_data/ebnerd/ebnerd_small/train'
    Ebrec_SMALL_VAL_DATASET_DIR =  ''
    Ebrec_SMALL_News_DATASET_DIR = '/data/badou/new_data/ebnerd/ebnerd_small/'
    logging.info("Load Data")
    train_behavior_df = read_behavior_df(Path(Ebrec_SMALL_TRAIN_DATASET_DIR,is_test= False))
    news_df = read_news_df(Path(Ebrec_SMALL_News_DATASET_DIR))
    # train_behavior_df, train_news_df = (
    #     read_behavior_df(Ebrec_SMALL_TRAIN_DATASET_DIR),
    #     read_news_df(Ebrec_SMALL_TRAIN_DATASET_DIR / "news.tsv"),
    # )
    # val_behavior_df = read_behavior_df(Ebrec_SMALL_VAL_DATASET_DIR / "behaviors.tsv")
    #
    val_behavior_df = read_behavior_df(Path(Ebrec_SMALL_VAL_DATASET_DIR, is_test=True))
    user_ids_to_idx_map = create_user_ids_to_idx_map(train_behavior_df, val_behavior_df)
    #
    # logging.info("Init EbrecTrainDataset")
    train_dataset = EbrecTrainDataset(
        train_behavior_df,
        news_df,
        user_ids_to_idx_map,
        batch_transform_texts=transform,
        npratio=4,
        history_size=20,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    logging.info("Start Iteration")
    for batch in train_dataloader:
        logging.info(f"{batch}")
        break
