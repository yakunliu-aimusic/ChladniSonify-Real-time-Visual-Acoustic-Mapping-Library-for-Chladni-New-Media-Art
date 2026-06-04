 
# Chladni Visual-Acoustic Mapping Project
## Dataset Introduction
![Modeled and generated Chladni patterns](./gen_chladni.jpg)

> Raw simulated Chladni vibration samples generated based on thin-plate physical equations.

## Data Augmentation Samples
![Augmented Chladni patterns](./aug_chladni.jpg)

> Augmented dataset via color perturbation, image filtering and matrix transformation for model training.

## System Mapping Workflow
![Mapping mechanism workflow](./mapping_flow.jpg)

> End-to-end pipeline of the proposed visual-acoustic sonification mapping framework.

### 论文地址：https://arxiv.org/abs/2605.09846
## Abstract
Existing visual-audio mapping schemes for new media art suffer from subjective mapping rules, expensive physical simulation and unsatisfactory real-time performance. To tackle these limitations, this paper proposes a real-time Chladni-pattern sonification framework oriented to interactive artistic creation.

Guided by Kirchhoff-Love thin-plate vibration theory, paired Chladni image-frequency datasets are generated and verified via ANSYS finite-element simulation. Targeting the fine nodal-line features of Chladni figures, a lightweight CNN embedded with CBAM attention is constructed to implement efficient and precise mode classification. Cooperated with Python and Max/MSP, an end-to-end real-time mapping system is built to convert recognized vibration patterns into corresponding tonal audio.

Experimental results demonstrate that the proposed method achieves 99.33% classification accuracy with 7.03 ms per-frame inference. The predicted frequencies are fully consistent with theoretical benchmarks with zero relative deviation, and the average full-link latency is less than 50 ms. This physically interpretable sonification system completely meets the real-time interactive requirements of new media art practice.


# Directory Structure
my_chladni_project/
├── data/
│ ├── raw/ # Raw generation scripts for pattern simulation & data augmentation
│ │ ├── clean_data.py
│ │ ├── noise.py
│ │ ├── random_color.py
│ │ └── random_matrix.py
│ ├── generated/ # Raw generated images+labels before train/val/test split
│ │ ├── images/
│ │ │ ├── clean/
│ │ │ ├── noise/
│ │ │ ├── random_color/
│ │ │ └── random_matrix/
│ │ └── labels/
│ │ ├── clean_labels.json
│ │ ├── noise_label.json
│ │ ├── random_label.json
│ │ └── merged_all_labels.json
│ └── processed/ # Standard split dataset for formal model training
│ ├── images/
│ │ ├── train/
│ │ ├── val/
│ │ └── test_synthetic/
│ └── labels/
│ ├── train.json
│ ├── val.json
│ └── test_synthetic.json
├── scripts/
│ └── split_dataset.py # Script for automatic dataset partition
├── model/ # Core CNN model codes(CBAM(SP5)_CNN_3×3 & baselines)
│ ├── dataset.py # Custom PyTorch Dataset loading class
│ ├── network.py # Definition of CNN & optimized CBAM architecture
│ ├── train.py # End-to-end model training pipeline
│ ├── evaluate.py # Model accuracy & latency evaluation
│ └── utils.py # General reusable helper functions
└── README.md
