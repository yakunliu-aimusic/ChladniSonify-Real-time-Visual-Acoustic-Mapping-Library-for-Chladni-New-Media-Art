# ChladniSonify: Real-Time Visual-Acoustic Mapping Library for Chladni New Media Art

## Description

ChladniSonify is the dataset, model-training, and Python algorithm repository for the paper **"ChladniSonify: A Visual-Acoustic Mapping Method for Chladni Patterns in New Media Art Creation"**. It supports reproducible Chladni pattern generation, augmentation, model training, model comparison, and inference evaluation for physically grounded visual-acoustic mapping.

## Project Scope and Paper Alignment

The paper describes a full system with three layers:

1. physically consistent Chladni pattern dataset construction;
2. lightweight CNN-CBAM pattern mode recognition;
3. Python-JUCE real-time visual-acoustic mapping and audio rendering.

This repository implements layers 1 and 2, plus a lightweight Python mapping-server prototype. It contains:

- formula-driven Chladni pattern generation;
- augmentation scripts for noise, color variation, sand-distribution disturbance, matrix filtering, and geometric transforms;
- JSON label generation, merging, and dataset splitting;
- Basic CNN, AlexNet, VGG16, CBAM-CNN 5x5, and CBAM-related comparison experiments;
- evaluation scripts for accuracy, F1-score, classification reports, and inference latency.

The JUCE/VST3 plugin, shared-memory video transfer, UDP return pipeline, real-time audio oscillator rendering, compiled plugin artifacts, and interactive hardware installation belong to the companion project rather than this repository.

## Related Repositories and Archives

- Dataset and model repository: [https://github.com/yakunliu-aimusic/ChladniSonify-Real-time-Visual-Acoustic-Mapping-Library-for-Chladni-New-Media-Art](https://github.com/yakunliu-aimusic/ChladniSonify-Real-time-Visual-Acoustic-Mapping-Library-for-Chladni-New-Media-Art)
- Audio-visual mapping plugin repository: [https://github.com/yakunliu-aimusic/VisionAudioMapping](https://github.com/yakunliu-aimusic/VisionAudioMapping)
- Zenodo archive for weights and artifacts: [https://zenodo.org/records/21730609](https://zenodo.org/records/21730609)
- DOI: [10.5281/zenodo.21730609](https://doi.org/10.5281/zenodo.21730609)
- arXiv paper: [https://arxiv.org/abs/2605.09846](https://arxiv.org/abs/2605.09846)

## Dataset Information

### Visual Overview

### Generated Chladni Pattern Samples

![Modeled and generated Chladni patterns](./gen_chladni.jpg)

### Data Augmentation Samples

![Augmented Chladni patterns](./aug_chladni.jpg)

### Mapping Workflow

![Mapping mechanism workflow](./mapping_flow.jpg)

The workflow figure refers to the full paper system. The complete JUCE/VST3 implementation is maintained in the companion plugin repository.

## Methodology

### Physical Modeling and Dataset Construction

The generation scripts implement a physically informed approximation of Chladni nodal topology for a square stainless-steel plate with center excitation and four free edges. Default parameters follow the paper setting:

- plate side length: `0.16 m`;
- plate thickness: `0.0008 m`;
- Young's modulus: `200e9 Pa`;
- Poisson's ratio: `0.3`;
- density: `7850 kg/m^3`;
- image size: `224 x 224` RGB.

The scripts compute plate bending stiffness and map modal orders to calibrated benchmark frequencies through a modal coefficient table. The rendering pipeline includes antisymmetric modal combinations, center attenuation, edge damping, adaptive nodal-region thresholding, stochastic sand-particle placement, and textured metallic backgrounds.

### Data Augmentation

The paper describes multi-dimensional augmentation to reduce overfitting to ideal synthetic data. The repository implements this through:

- `clean.py`: clean formula-driven images;
- `noise .py`: Gaussian noise, occlusion, blur, sand-distribution variation, and geometric transforms;
- `random_color.py`: full-gamut plate/sand color randomization with contrast constraints;
- `random_matrix filter  .py`: local CNN-style matrix filters for texture variation.

All augmentation operations preserve the modal and frequency labels.

### Data Preprocessing Pipeline

The preprocessing pipeline is consistent with the paper and is implemented in the training/evaluation scripts:

1. convert each image to RGB;
2. resize to `224 x 224`;
3. convert pixel values to floating-point tensors;
4. normalize channels with mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`;
5. map each physical modal order to a discrete class index from `0` to `14`;
6. keep modal, frequency, and augmentation metadata in JSON labels where available.

Implemented transform:

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

The paper describes 15-dimensional one-hot labels at the dataset-construction level. The PyTorch implementation returns integer `modal_idx` targets, which is the expected target format for `torch.nn.CrossEntropyLoss`.

### Model Architecture

The repository contains the model families discussed in the paper:

- `kladni_model_classification /`: baseline CNN;
- `kladni_model_alex/`: AlexNet comparison;
- `kladni_model_vgg 3.0/`: VGG16 comparison;
- `kladni_cbam 5x5/`: paper-oriented CBAM-CNN with 5x5 spatial attention;
- `kladni_cbam 7x7/`: CBAM comparison module.

The `kladni_cbam 5x5` model follows the paper's core design: convolutional feature extraction, CBAM channel-spatial attention, adaptive average pooling to `4 x 4`, and a classifier head producing 15 modal logits.

## Assessment Metrics

The paper's Assessment Metrics section belongs logically before the experiment results because it defines the evaluation criteria used throughout the experiments. This repository now documents it here, before the running guide.

### Top-1 Classification Accuracy

Top-1 accuracy measures whether the predicted modal class exactly matches the ground-truth modal class. In Chladni visual-acoustic mapping, this is the primary correctness metric because any modal error maps to an incorrect benchmark frequency.

Code implementation:

```python
acc = np.mean(all_preds == all_labels)
```

### Macro F1-Score

Macro F1-score averages per-class F1 values across all modal classes. It checks whether the model performs consistently across both simple low-frequency patterns and more complex high-frequency patterns.

Code implementation:

```python
macro_f1 = f1_score(all_labels, all_preds, average='macro')
```

### Micro F1-Score

Micro F1-score is reported as an auxiliary metric. For single-label multi-class classification, it is usually close to overall accuracy, but it is useful as a sanity check.

Code implementation:

```python
micro_f1 = f1_score(all_labels, all_preds, average='micro')
```

### Classification Report

The evaluation scripts print per-class precision, recall, and F1-score through `sklearn.metrics.classification_report`, enabling class-level error inspection.

### Single-Image Inference Latency

The paper uses single-image inference latency to verify real-time deployability. The current scripts benchmark pure model inference with a synthetic tensor of shape `1 x 3 x 224 x 224` and report milliseconds per image and throughput. The current implementation averages 100 runs; change the benchmark loop to 1000 runs for exact paper-protocol reproduction.

### End-to-End Latency

The full-link latency reported in the paper includes camera/video acquisition, shared-memory transfer, Python inference, UDP return, JUCE parsing, and audio rendering. That benchmark belongs to the companion JUCE/VST3 repository and is not fully reproduced by this Python-only repository.

## Code Information

The codebase is organized as a set of parallel experiment modules. Each module generally contains data-generation scripts, generated labels, processed dataset folders, model definitions, training scripts, evaluation scripts, and dataset-splitting utilities. The `kladni_cbam 5x5` folder is the primary implementation aligned with the paper's optimized CNN-CBAM configuration, while the other folders provide baseline and comparative experiments.

## Repository Structure

```text
.
├── kladni_model_classification /
│   ├── data/
│   ├── model/
│   ├── scripts/
│   └── chladni_server.py
├── kladni_model_alex/
├── kladni_model_vgg 3.0/
├── kladni_cbam 5x5/
├── kladni_cbam 7x7/
├── gen_chladni.jpg
├── aug_chladni.jpg
├── mapping_flow.jpg
├── LICENSE
└── README.md
```

Each experiment folder generally contains `data/raw`, `data/generated`, `data/processed`, `model`, and `scripts` subdirectories.

## Requirements

Recommended Python version: Python 3.9.

Core dependencies:

- `numpy`
- `scipy`
- `matplotlib`
- `Pillow`
- `scikit-learn`
- `torch`
- `torchvision`

Suggested setup:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib pillow scikit-learn torch torchvision
```

On Windows:

```bash
.venv\Scripts\activate
```

## Usage Instructions

The examples below use `kladni_cbam 5x5`, which is closest to the paper's optimized CNN-CBAM configuration. Replace the folder name to run another model variant.

### 1. Generate Clean Samples

```bash
python "kladni_cbam 5x5/data/raw/clean.py"
```

### 2. Generate Noisy Augmentation

```bash
python "kladni_cbam 5x5/data/raw/noise .py"
```

### 3. Generate Color Augmentation

```bash
python "kladni_cbam 5x5/data/raw/random_color.py"
```

### 4. Generate Matrix-Filter Augmentation

```bash
python "kladni_cbam 5x5/data/raw/random_matrix filter  .py"
```

### 5. Merge Labels

```bash
python "kladni_cbam 5x5/data/generated/combine_labels.py"
```

### 6. Split Dataset

```bash
python "kladni_cbam 5x5/scripts/split_dataset.py"
```

Current code uses an image-level split ratio of `0.8/0.1/0.1` for train, validation, and synthetic test. The paper summarizes an `8:2` train/test split for the expanded dataset. Adjust `SPLIT_RATIO` if exact paper reproduction is required.

### 7. Train Model

```bash
python "kladni_cbam 5x5/model/train.py"
```

Outputs:

```text
kladni_cbam 5x5/model/best_model.pth
kladni_cbam 5x5/plots/training_curve.png
```

The current `kladni_cbam 5x5` script uses batch size `32`, `20` epochs, Adam, and learning rate `8e-4`. The paper reports batch size `32`, Adam, learning rate `1e-4`, `50` epochs, and early stopping after 10 epochs without validation-loss improvement. Modify the script if strict reproduction is needed.

### 8. Evaluate Model

```bash
python "kladni_cbam 5x5/model/evaluate.py"
```

The script reports inference speed, Top-1 accuracy, macro F1, micro F1, and per-class precision/recall/F1.

### 9. Run Python Mapping Server Prototype

```bash
python "kladni_model_classification /chladni_server.py"
```

Default ports:

- image input: `9999`;
- frequency response: `9998`.

This is a Python socket prototype, not the final shared-memory and UDP JUCE/VST3 system described in the paper.

## Implementation Notes for Strict Paper Reproduction

The codebase is consistent with the paper's core algorithmic direction, but several implementation parameters should be checked before exact reproduction:

- `evaluate.py` uses 100 pure-inference timing runs; the paper describes 1000 runs.
- `kladni_cbam 5x5/model/train.py` uses 20 epochs and learning rate `8e-4`; the paper reports 50 epochs and learning rate `1e-4` with early stopping.
- `split_dataset.py` uses `0.8/0.1/0.1`; the paper summarizes an `8:2` train/test split.
- The full JUCE/VST3 shared-memory and UDP audio-rendering pipeline is maintained in the companion repository.

## Citations

If you use this repository, dataset, or model artifacts, cite the paper and Zenodo archive.

```bibtex
@misc{chladnisonify_zenodo_2026,
  title        = {ChladniSonify-Dataset},
  author       = {Liu, Yakun},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21730609},
  url          = {https://doi.org/10.5281/zenodo.21730609}
}
```

## License & Contribution Guidelines

See `LICENSE` for the repository license. The Zenodo record states that the archived dataset package is released under Creative Commons Attribution 4.0 International.

Contributions are welcome when they improve reproducibility, documentation quality, or experimental clarity. Recommended contribution rules:

- keep all source-code comments, console messages, exceptions, and documentation text in English;
- clearly document any change that affects dataset generation, model training, evaluation metrics, or reported performance;
- avoid committing large generated artifacts unless they are required for reproducibility and are explicitly documented;
- use issues or pull requests to discuss substantial methodological changes before merging;
- cite the associated paper and Zenodo archive when reusing the dataset, code, or trained weights.
