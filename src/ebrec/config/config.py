from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from ebrec.models.newsrecv2.recommendation import NewsRecommendationModel


@dataclass
class TrainConfig:
    random_seed: int = 42
    pretrained: str = "/home/dev/models/bce-embedding-base_v1"
    npratio: int = 4
    history_size: int = 4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8  # batch_size = 16 x 8 = 128
    epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    conv_kernel_num: int = 300
    user_emb_dim: int = 200
    kernel_size: int = 3
    max_len: int = 64
    query_dim: int = 200
    news_recommendation_model: NewsRecommendationModel = NewsRecommendationModel.NRMS


cs = ConfigStore.instance()

cs.store(name="train_config", node=TrainConfig)
