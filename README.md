# industry_code_labeling

This project builds and deploys a hierarchical machine learning model for predicting 2022 NAICS industry codes from company descriptions.

The work began with a word-token CNN approach, then moved to a stronger sentence-embedding pipeline using pretrained transformer embeddings. The current best model uses MPNet sentence embeddings with a hierarchical PyTorch classifier that predicts NAICS codes from 2 digits through the full 6-digit code.

A live dashboard utilizing the API that you can use to interact with my model is available [here](https://019db3be-81fa-ce38-110c-3c40da46c791.share.connect.posit.cloud/).

## Project goal

The goal is to classify free-form business descriptions into 2022 NAICS codes using a strict hierarchy:

- predict 2-digit NAICS
- condition on that to predict 3-digit
- then 4-digit
- then 5-digit
- then full 6-digit

This structure helps the model learn broad industry routing first, then increasingly detailed subclasses.

## Current best model

The current best model uses:

- sentence embedder: `sentence-transformers/all-mpnet-base-v2`
- hierarchical classifier: PyTorch MLP with multi-level output heads
- hidden dimension: `768`
- parent embedding dimension: `64`
- dropout: `0.1`
- optimizer: Adam
- early stopping on validation 6-digit accuracy

Best test performance from the final selected model:

- `acc_y2 = 0.5431`
- `acc_y3 = 0.3900`
- `acc_y4 = 0.2513`
- `acc_y5 = 0.1749`
- `acc_y6 = 0.1303`
- `top5_y6 = 0.1766`

## Repository structure

```text
industry_code_labeling/
├── api/                         # Flask API for deployed inference
├── dashboard/                   # Python Shiny dashboard code
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── training/
│   ├── notebooks/
│   ├── scripts/
│   └── artifacts/
├── outputs/
│   └── predictions/
├── README.md
└── environment.yml
```

## Main workflow

### 1. Data acquisition
- `training/scripts/download_data.py`
- pulls the source dataset and saves a raw local copy

### 2. Data preparation
- `training/scripts/prepare_data.py`
- cleans 2022 NAICS labels
- derives `y2` through `y6`
- creates train / validation / test splits
- ensures all evaluated classes exist in training

### 3. Embedding generation
- `training/scripts/embed_data.py`
- generates sentence embeddings for company descriptions
- saves embedded train / validation / test arrays
- saves label maps and hierarchy masks

### 4. Model training
- `training/scripts/train_hierarchical_embed.py`
- trains the hierarchical MPNet-based classifier
- uses teacher forcing during training
- uses strict masked hierarchical decoding at evaluation time

### 5. Evaluation
- `training/scripts/evaluate_hierarchical_embed.py`
- exports row-level predictions
- creates grouped summaries by 2-digit through 6-digit NAICS

### 6. Deployment
- `api/`
- Flask API for inference
- deployed separately to Hugging Face Spaces as a Docker app

## Deployment architecture

This project uses two separate deployment targets:

### GitHub
This repository is the full project source repository, including:
- training code
- evaluation code
- dashboard code
- API code
- documentation

### Hugging Face Spaces
A separate Docker Space hosts the inference API only. That deployment includes only the files required for serving predictions.

The Space exposes endpoints such as:

- `GET /`
- `GET /health`
- `POST /predict`

## Example API request

```json
{
  "company_description": "Commercial roofing contractor specializing in industrial and warehouse roofing systems."
}
```

Example response structure:

```json
{
  "pred_prob_y2": 0.4753499925136566,
  "pred_prob_y3": 0.8068565726280212,
  "pred_prob_y4": 0.660828709602356,
  "pred_prob_y5": 0.49590933322906494,
  "pred_prob_y6": 1.0,
  "pred_top5_y6": [238160, 111120, 111140, 111150...],
  "pred_y2": 23,
  "pred_y3": 238,
  "pred_y4": 2381,
  "pred_y5": 23816,
  "pred_y6": 238160
}
```

## Environment setup

This project is currently run from a Conda environment.

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate naics_torch
```

## Notes on model development

Several model families were explored during development:

- token-based CNN hierarchy
- pretrained sentence-embedding hierarchy
- multiple head sizes and hierarchy embedding sizes
- grouped evaluation by NAICS branch

The strongest improvement came from replacing the earlier tokenizer/CNN representation with pretrained sentence embeddings. This substantially improved both broad routing and detailed 6-digit classification.
