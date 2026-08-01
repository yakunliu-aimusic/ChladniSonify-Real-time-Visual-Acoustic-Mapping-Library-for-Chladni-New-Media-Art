'''Directory structure
my_chladni_project/
├── data/
│   ├── raw/                     # Raw generation scripts (clean_data.py, noise.py...)
│   │   ├── clean_data.py
│   │   ├── noise.py
│   │   ├── random_color.py
│   │   └── random_matrix.py
│   ├── generated/               # Auto-generated raw images and labels (not yet split)
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
│   └── processed/               # Final split dataset for model training
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
│   └── split_dataset.py         # Dataset partitioning script
│
├── model/                       # Model definition, training, and evaluation
│    ├── dataset.py        # Custom Dataset class
│    ├── network.py        # Basic CNN model (without CBAM)
│    ├── train.py          # Training script
│    ├── evaluate.py       # Evaluation script
│    └── utils.py          # Helper utilities (optional but recommended)
│
└── README.md
'''
