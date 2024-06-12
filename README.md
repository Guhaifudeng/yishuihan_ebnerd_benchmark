# 参考项目
https://github.com/yamanalab/gpt-augmented-news-recommendation
https://github.com/ebanalyse/ebnerd-benchmark

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