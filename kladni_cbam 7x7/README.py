'''目录结构
my_chladni_project/
├── data/
│   ├── raw/                     ← 原始生成脚本（clean_data.py, noise.py...）
│   │   ├── clean_data.py
│   │   ├── noise.py
│   │   ├── random_color.py
│   │   └── random_matrix.py
│   ├── generated/               ← 自动生成的原始图像+标签（未划分）
│   │   ├── images/
│   │   │   ├── clean/
│   │   │   ├── noise/
│   │   │   ├── random_color/
│   │   │   └── random_matrix/
│   │   ├── combine_labels/
│   │   └── labels/
│   │       ├── clean_labels.json
│   │       ├── noise_label.json
│   │       ├── random_label.json
│   │       └── merged_all_labels.json
│   └── processed/               ← 划分后的最终数据集（供训练用）
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test_synthetic/
│       └── labels/
│           ├── train.json
│           ├── val.json
│           └── test_synthetic.json
│
├── scripts/
│   └── split_dataset.py         ← 数据划分脚本移至此处
│
├── model/                       ← 模型定义、训练、评估
│    ├── dataset.py        ← 自定义 Dataset
│    ├── network.py        ← 基础 CNN 模型（无 CBAM）
│    ├── train.py          ← 训练脚本
│    ├── evaluate.py       ← 评估脚本
│    └── utils.py          ← 辅助函数（可选，但推荐）
│
└── README.md
'''
