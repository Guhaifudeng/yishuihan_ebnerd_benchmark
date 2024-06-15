# 参考项目
- https://github.com/yamanalab/gpt-augmented-news-recommendation (pytorch 版本的新闻推荐模型，使用了其中的models)
- https://github.com/ebanalyse/ebnerd-benchmark (官方baseline-tf keras实现)

# 核心代码
```
./src/
├── ebrec
│   ├── config # 配置文件
│   │   ├── config.py
│   │   ├── path.py
│   ├── data # 数据集构建
│   │   ├── dataframe.py
│   │   ├── EbrecDataset.py
│   │   ├── __init__.py
│   │   └── test_load_data.py
│   ├── evaluation # 评测代码
│   ├── models
│   │   ├── fastformer # 未使用
│   │   ├── newsrec # 官网baseline 
│   │   └── newsrecv2 # 基于其他开源项目写的版本2

│   ├── nrms_ebnerd_eval.py
│   ├── nrms_ebnerd.py # 官网baseline - 训练及预测脚本
│   ├── predict_v2.py # V2 - 训练及评测脚本
│   ├── train_v2.py # V2 - 预测脚本
│   └── utils # 工具类
└── __init__.py

```

# 比赛主页
RecSys Challenge 2024
- Home:https://recsys.eb.dk/
- dataset:https://recsys.eb.dk/dataset/
- demo:https://github.com/ebanalyse/ebnerd-benchmark
- register:https://www.codabench.org/competitions/2469/?secret_key=98314b2c-9237-471e-905c-2a88bf6a1d8a

# 数据
```
Ekstra Bladet News Recommendation Dataset - General License Terms
Thank you! 

Please use the following links to access the dataset. We recommend opening the ones you want to download in different tabs or to print the page.

—————— ——————

- ebnerd_demo (20MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip  
(*5,000 users)

—————— ——————

- ebnerd_small (80MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip
(*50,000 users)

—————— ——————

- ebnerd_large (3.0GB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip 

- Articles (140MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/articles_large_only.zip
(Only download the articles from Large)

—————— ——————

- ebnerd_testset (1.5GB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip 

- Example of full submission file (220MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/predictions_large_random.zip
(It’s all random predictions but this file will successfully upload to the leaderboard)

—————— ——————
Artifacts:
- Ekstra-Bladet-word2vec (133MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_word2vec.zip 

- Ekstra_Bladet_image_embeddings (372MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_image_embeddings.zip 

- Ekstra-Bladet-contrastive_vector (341MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip

- google-bert-base-multilingual-cased (344MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip (多语言预训练模型) 

- FacebookAI-xlm-roberta-base (341MB): https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/FacebookAI_xlm_roberta_base.zip (多语言预训练模型)
```
