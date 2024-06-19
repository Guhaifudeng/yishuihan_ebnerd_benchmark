import json
import os
os.environ['POLARS_MAX_THREADS'] = "8"
from pathlib import Path

import hydra
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.modeling_outputs import ModelOutput
from datetime import datetime

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from ebrec.config.config import TrainConfig
from ebrec.config.path import LOG_OUTPUT_DIR, EBREC_SMALL_TRAIN_DATASET_DIR, EBREC_SMALL_VAL_DATASET_DIR, \
    MODEL_OUTPUT_DIR, EBREC_SMALL_DATASET_DIR, EBREC_LARGE_DATASET_DIR, EBREC_TEST_DATASET_DIR
from ebrec.evaluation.RecEvaluator import RecEvaluator, RecMetrics
from ebrec.data.EbrecDataset import EbrecTrainDataset,EbrecValDataset,EbrecTestDataset

from ebrec.models.newsrecv2.recommendation.nrms import NRMS, PLMBasedNewsEncoder as NRMSNewsEncoder, UserEncoder as NRMSUserEncoder
from ebrec.models.newsrecv2.recommendation.npa import NPA, PLMBasedNewsEncoder as NPANewsEncoder, UserEncoder as NPAUserEncoder
from ebrec.models.newsrecv2.recommendation.naml import NAML, PLMBasedNewsEncoder as NAMLNewsEncoder, UserEncoder as NAMLUserEncoder
from ebrec.models.newsrecv2.recommendation import NewsRecommendationModel

from ebrec.data.dataframe import read_behavior_df, read_news_df, create_user_ids_to_idx_map

from ebrec.utils.log_util import init_logger
logging = init_logger(__name__)
from ebrec.utils.path import generate_folder_name_with_timestamp
from ebrec.utils.random_seed import set_random_seed
from ebrec.utils.text import create_transform_fn_from_pretrained_tokenizer

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



def predict(
    pretrained: str,
    news_recommendation_model: NewsRecommendationModel,
    npratio: int,
    history_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    conv_kernel_num: int,
    kernel_size: int,
    user_emb_dim: int,
    query_dim: int,
    max_len: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    logging.info("Start")
    setting_info = {
        "pretrained": pretrained,
        "news_recommendation_model": news_recommendation_model,
        "npratio": npratio,
        "history_size": history_size,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "conv_kernel_num": conv_kernel_num,
        "kernel_size": kernel_size,
        "user_emb_dim": user_emb_dim,
        "query_dim": query_dim,
        "max_len": max_len
    }

    logging.info(setting_info)

    """
    0. Definite Parameters & Functions
    """
    EVAL_BATCH_SIZE = 1
    hidden_size: int = AutoConfig.from_pretrained(pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    transform_fn = create_transform_fn_from_pretrained_tokenizer(AutoTokenizer.from_pretrained(pretrained), max_len)
    model_save_dir = generate_folder_name_with_timestamp(MODEL_OUTPUT_DIR)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    """
    1. Load Data & Create Dataset
    """
    logging.info("Initialize Dataset")

    train_news_df = read_news_df(Path(EBREC_LARGE_DATASET_DIR))
    val_news_df = train_news_df
    test_behavior_df = read_behavior_df(Path(EBREC_TEST_DATASET_DIR),mode='test',history_size=history_size)


    # test_behavior_df = test_behavior_df[:50]


    user_ids_to_idx_map = {}

    test_dataset = EbrecTestDataset(test_behavior_df, val_news_df, user_ids_to_idx_map, transform_fn,device)

    """
    2. Init Model
    """
    logging.info("Initialize Model")
    newsrec_net = None

    if news_recommendation_model == NewsRecommendationModel.NRMS:
        news_encoder = NRMSNewsEncoder(pretrained)
        user_encoder = NRMSUserEncoder(hidden_size=hidden_size)
        newsrec_net = NRMS(
            news_encoder=news_encoder, user_encoder=user_encoder, hidden_size=hidden_size, loss_fn=loss_fn
        ).to(device, dtype=torch.bfloat16)
    elif news_recommendation_model == NewsRecommendationModel.NPA:
        news_encoder = NPANewsEncoder(
            pretrained=pretrained,
            conv_kernel_num=conv_kernel_num,
            kernel_size=kernel_size,
            user_emb_dim=user_emb_dim,
            query_dim=query_dim,
        )
        user_encoder = NPAUserEncoder(conv_kernel_num=conv_kernel_num, user_emb_dim=user_emb_dim, query_dim=query_dim)
        newsrec_net = NPA(
            news_encoder=news_encoder,
            user_encoder=user_encoder,
            user_num=100,
            user_emb_dim=user_emb_dim,
            loss_fn=loss_fn,
        ).to(device, dtype=torch.bfloat16)
    elif news_recommendation_model == NewsRecommendationModel.NAML:
        news_encoder = NAMLNewsEncoder(
            pretrained=pretrained,
            conv_kernel_num=conv_kernel_num,
            kernel_size=kernel_size,
            query_dim=query_dim,
        )
        user_encoder = NAMLUserEncoder(conv_kernel_num=conv_kernel_num, query_dim=query_dim)
        newsrec_net = NAML(
            news_encoder=news_encoder,
            user_encoder=user_encoder,
            loss_fn=loss_fn,
        ).to(device, dtype=torch.bfloat16)
    else:
        raise Exception(f"Unknown news rec model: {news_recommendation_model}")

    """
    3. Train
    """
    logging.info("Training Start")

    # newsrec_net.load_state_dict(torch.load('/home/dev/ebnerd-benchmark/src/output/model/2024-06-12_20-08-38/nrms_bce-embedding-base_v1.pth'))
    #newsrec_net.load_state_dict(torch.load('/home/dev/ebnerd-benchmark/src/output/model/2024-06-12_20-08-38/nrms_bce-embedding-base_v1.pth'))
    newsrec_net.load_state_dict(torch.load('/home/dev/ebnerd-benchmark/src/output/model/2024-06-15_19-48-30/nrms_bce-embedding-base_v1.pth'))
    newsrec_net.eval()

    test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True)


    logging.info({"device": device})
    pred_validation_list = []
    for batch in tqdm(test_dataloader, desc="Predict for EBRECValDataset"):
        # Inference
        for k in batch.keys():
            batch[k] = batch[k].to(device)

        with torch.no_grad():
            model_output: ModelOutput = newsrec_net(**batch)

        # Convert To Numpy
        y_score: torch.Tensor = model_output.logits.flatten().cpu().to(torch.float64).numpy()
        # logging.info('y_score {}'.format(y_score))

        pred_validation_list.extend(y_score.tolist())

    logging.info('start test predict')
    df_test = test_behavior_df
    df_test = add_prediction_scores(df_test, pred_validation_list)
    df_test.head(2)
    logging.info('finish test predict')

    logging.info('start save test predict')
    df_test = df_test.with_columns(
        pl.col("scores")
        .map_elements(lambda x: list(rank_predictions_by_score(x)))
        .alias("ranked_scores")
    )
    df_test.head(2)
    write_submission_file(
        impression_ids=df_test[DEFAULT_IMPRESSION_ID_COL],
        prediction_scores=df_test["ranked_scores"],
        path=Path("predictions.txt"),
    )

    logging.info('done')


@hydra.main(version_base=None, config_name="train_config")
def main(cfg: TrainConfig) -> None:
    try:
        set_random_seed(cfg.random_seed)
        predict(
            pretrained=cfg.pretrained,
            news_recommendation_model=cfg.news_recommendation_model,
            npratio=cfg.npratio,
            history_size=cfg.history_size,
            batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            conv_kernel_num=cfg.conv_kernel_num,
            kernel_size=cfg.kernel_size,
            user_emb_dim=cfg.user_emb_dim,
            query_dim=cfg.query_dim,
            max_len=cfg.max_len
        )
    except Exception as e:
        logging.error(e)


if __name__ == "__main__":
    main()
