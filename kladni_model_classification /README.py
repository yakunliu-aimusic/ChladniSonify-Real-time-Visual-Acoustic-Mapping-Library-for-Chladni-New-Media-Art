'''
# Directory Structure
my_chladni_project/
├── data/
│   ├── raw/                     # Raw generation scripts (clean_data.py, noise.py...)
│   │   ├── clean_data.py
│   │   ├── noise.py
│   │   ├── random_color.py
│   │   └── random_matrix.py
│   ├── generated/               # Auto-generated raw images & labels (not split yet)
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
├── model/                       # Model definition, training & evaluation codes
│    ├── dataset.py        # Custom PyTorch Dataset class
│    ├── network.py        # Neural network architecture definition
│    ├── train.py          # Model training pipeline
│    ├── evaluate.py       # Model evaluation & metrics calculation
│    └── utils.py          # Common helper functions
│
└── README.md

'''