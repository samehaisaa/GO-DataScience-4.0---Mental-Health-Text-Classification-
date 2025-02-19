# GO DS 4.0 - Mental Health Text Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](requirements.txt)
[![Zindi](https://img.shields.io/badge/Zindi-Platform-blue)](https://zindi.africa/competitions/go-data-science-40-mental-health-challenge/)

This repository contains my solution for the **[GO DATA SCIENCE 4.0 Hackathon](https://zindi.africa/competitions/go-data-science-40-mental-health-challenge/)** hosted on Zindi, where I achieved **[43rd place](https://zindi.africa/users/SamehAissa/competitions/certificate)** out of 194 participants. The challenge focused on classifying mental health-related text discussions into predefined categories using Natural Language Processing (NLP) techniques.

---

## 🏅 Competition Results

### Final Leaderboard Performance
- **Rank**: 43 out of 309 participants
- **Validation Accuracy**: 74.4%
- **Public Leaderboard Score**: 0.7528
- **Private Leaderboard Score**: 0.7371

### Top Performers (Excerpt)
| Rank | Team Name               | Public Score | Private Score |
|------|-------------------------|--------------|---------------|
| 1    | Recursive Duo           | 0.8189       | 0.7996        |
| ...  | ...                     | ...          | ...           |
| 9    | one crew        | 0.7786       | 0.7792        |
| 25   | Llama                     | 0.7610       | 0.7648        |
| 43   | **SamehAissa (Me)**     | **0.7686**   | **0.7528**    |
| 44   | ...                     | ...          | ...           |

---

## 📊 Analysis

### Key Observations
1. **Top Scores**:
   - The winning team achieved a public score of **0.8189** and a private score of **0.7996**.
2. **My Performance**:
   - Achieved a public score of **0.7686** and a private score of **0.7528**.
   - Ranked **43rd**, placing in the top 14% of participants.

3. **Leaderboard Insights**:
   - A small gap between public and private scores indicates robust models.
   - The competition was highly competitive, with close scores among top teams.

---

## 🏆 Competition Overview

### Problem Statement
The goal was to develop a model that accurately classifies text entries (titles and content) from online discussions into categories representing mental health issues. Each entry in the dataset included:
- `id`: Unique identifier
- `title`: Discussion title
- `content`: Main body of the text
- `target`: Mental health category (only in training data)

### Evaluation Metric
The model's performance was evaluated using **Accuracy** as the primary metric.

---

## 🚀 My Approach

### Key Steps
1. **Data Preprocessing**:
   - Combined `title` and `content` into a single text feature.
   - Handled missing values and cleaned text data.
   - Encoded target labels into numerical format.

2. **Modeling**:
   - Experimented with **BERT** and **RoBERTa** architectures.
   - Implemented **class weighting** to handle imbalanced data.
   - Used **Text Augmentation** (EDA) to improve generalization.

3. **Training**:
   - Fine-tuned transformer models using Hugging Face's `Trainer` API.
   - Applied **Focal Loss** to focus on hard-to-classify examples.
   - Used **Test-Time Augmentation (TTA)** for robust predictions.

4. **Evaluation**:
   - Achieved **~76.8% Public Accuracy** on the validation set.
   - Secured **43rd place** on the final leaderboard.

---
