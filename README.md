
# EvoBot: Advancing LLM-based Social Bot Generation and Its Detection through an Adversarial Learning Framework


🎉 **2025/08/21** We are thrilled to announce that `EvoBot` has been accepted to the **EMNLP 2025 Main Conference**!

This repository contains the code for EvoBot, a novel adversarial learning framework where an LLM-based social bot (EvoBot) and a graph-based detector are co-adapted. EvoBot learns to generate increasingly human-like content, while the Detector simultaneously improves its ability to distinguish bot-generated content from human text.

The training process consists of two main phases:
1.  **Supervised Fine-Tuning (SFT):** EvoBot's base LLM is fine-tuned on human social media data to learn basic human expression patterns.
2.  **Adversarial Training:** EvoBot and the Detector are iteratively trained. EvoBot uses feedback from the Detector (via Direct Preference Optimization - DPO) to refine its content generation, while the Detector is updated using EvoBot's evolving outputs.

## File Structure

Here is an overview of the main directories and important files in this project:

```

.
├── config.json              # Configuration file for paths, hyperparameters, etc.
├── data/
│   ├── processed_data/      # Stores processed data for communities (embeddings, etc.)
│   ├── raw_data/            # Stores raw community data (tweets, user info, labels, graph edges)
│   └── sft_data/            # Stores generated data for Supervised Fine-Tuning
├── environment.yml          # Conda environment specification for reproducibility
├── models/
│   ├── discriminator.py     # Contains the BotRGCN detector model architecture
│   ├── feature_extractor.py # Utility for feature extraction (e.g., RoBERTa embeddings)
│   ├── SFT/                 # Default directory for saving SFT models
│   ├── DPO/                 # Default directory for saving DPO models
│   └── Detector/            # Directory for saving detector models/checkpoints
├── README.md                # This file
├── sft.py                   # Script for Phase 1: Supervised Fine-Tuning of EvoBot
├── train.py                 # Script for Phase 2: Adversarial training of EvoBot and Detector
└── utils/                   # Utility functions

```

## Setup

### 1. Environment
It is recommended to use a Conda environment. You can create and activate the environment using the provided `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate evobot_env # Or your chosen environment name
```

### 2\. Dependencies

Key dependencies include:

  * PyTorch (`torch`)
  * Transformers (`transformers`)
  * TRL (`trl`) for SFT and DPO trainers
  * PEFT (`peft`) for LoRA fine-tuning
  * Datasets (`datasets`)
  * Pandas (`pandas`)
  * NumPy (`numpy`)

The `environment.yml` file should handle the installation of all necessary packages.

### 3\. Configuration

The main configuration file for the adversarial training phase is `config.json`. Before running `train.py`, review and update this file with appropriate paths, model parameters, training hyperparameters, precision settings, etc.

Key settings in `train.py` to be aware of (may also be configurable via `config.json` or directly in the script):

  * `COMM`: Index of the community to process (e.g., `COMM = 11`).
  * Device allocation (`generator_device`, `other_device`).

## Data Preparation

1.  **Raw Data:**
    Place your raw community data (e.g., from TwiBot-22) into subdirectories like `data/raw_data/community_{COMM}/`. This typically includes:

      * `user_summary.json` (or similar for user profile information)
      * `tweet.json` (user tweets)
      * `edge.csv` (follower/followee relationships)
      * `label.csv` (user labels: human/bot)

2.  **SFT Data Generation:**
    The `sft.py` script will automatically attempt to generate SFT data if it's not found. This process is handled by `utils/sft_data_generation.py` using the `instruction_tune_instance` function, which creates `sft_data.jsonl` in `data/sft_data/community_{COMM}/`.

3.  **Preprocessing for Adversarial Training:**
    The `train.py` script expects certain processed data, such as tweet embeddings (`tweets_tensor.pt`). Some preprocessing steps like `tweets_embedding` (from `data/raw_data/preprocess.py`, though this specific script isn't listed in `utils`, it's imported in `train.py`) are called within `train.py` if output files don't exist. Ensure your `config.json` and paths in `train.py` point to the correct data locations.

## Training Process

The training is divided into two main phases:

### Phase 1: Supervised Fine-Tuning (SFT) - `sft.py`

This phase fine-tunes the base Large Language Model (EvoBot's generator) on human-generated text to learn basic user representation and language style.

**To run SFT:**

```bash
python sft.py \
    --model <base_model_name> \
    --comm <community_index> \
    --epochs <num_epochs> \
    --data-num <sft_data_samples> \
    --input <output_prefix> \
    --parent_directory <sft_model_save_path>
```

**Arguments:**

  * `--model`: Name of the base model to use (e.g., `llama2_7b`, `mistral`). Defaults to `llama2_7b`.
  * `--comm`: Community index for data loading and saving. Defaults to `0`.
  * `--epochs`: Number of training epochs. Defaults to `5`.
  * `--data-num`: Number of SFT data samples to generate/use. Defaults to `1024`.
  * `--input`: Prefix for the output model directory name. Defaults to `result`.
  * `--parent_directory`: Directory where SFT models will be saved. Defaults to `./models/SFT`.

**Output:**
The script saves the fine-tuned model (including merged adapters) to a path like `<sft_model_save_path>/sft_merged_ckp_{community_index}`. This model serves as the initial EvoBot generator for Phase 2.

### Phase 2: Adversarial Training - `train.py`

This phase iteratively trains EvoBot (the generator) and the Detector. EvoBot learns to generate more human-like and evasive content using DPO with feedback from the Detector, while the Detector learns to better distinguish EvoBot's outputs from human text.

**Before running:**

  * Ensure you have a trained SFT model from Phase 1.
  * Update `COMM` variable in `train.py` to specify the target community.
  * Carefully review and configure `config.json` with paths (especially `model_path` which should initially point to your SFT model from Phase 1), hyperparameters for DPO, PEFT, and the discriminator.
  * Ensure correct device allocation in `train.py`.

**To run Adversarial Training:**

```bash
python train.py
```

The script will execute the following main stages:

1.  **Initial Detector Training:** Pre-trains the `BotRGCN` detector on the original dataset.
2.  **(Optional Baseline):** A section for training/evaluating with a vanilla LLM may run.
3.  **Main Adversarial Loop (for `training_epoch` iterations defined in `config.json`):**
      * **EvoBot Content Generation:** The current EvoBot generator model rewrites tweets for bot accounts in the dataset.
      * **Tweet Embedding Update:** Embeddings for the dataset are updated with the new bot tweets.
      * **Detector Training:** The Detector is re-trained on the updated dataset (including EvoBot's latest outputs).
      * **DPO Data Generation:** EvoBot generates pairs of candidate responses for selected bot profiles. These responses are classified by the current Detector, and (chosen, rejected) pairs are formed for DPO training based on these classifications.
      * **EvoBot DPO Training:** The EvoBot generator is fine-tuned using DPOTrainer on the newly generated preference dataset. Its weights are updated for the next iteration.

**Outputs:**

  * Detector model checkpoints (implicitly saved during training, path might be in `config.json` or `./models/Detector/`).
  * DPO datasets for each epoch (e.g., in `./data/dpo_data/community_{COMM}/`).
  * EvoBot (Generator) DPO model checkpoints for each epoch (e.g., in `./models/DPO/community_{COMM}/merged_ckp_{epoch}`). The `model_path` variable in `train.py` is updated to point to the latest merged DPO model.

## Models

Trained models are typically stored in subdirectories within the `models/` folder:

  * **SFT Models:** `models/SFT/sft_merged_ckp_{community_index}`
  * **DPO Models (EvoBot):** `models/DPO/community_{COMM}/merged_ckp_{epoch}`
  * **Detector Models:** Check training logs or `config.json` for specific paths; often saved within a structure like `models/Detector/`.
