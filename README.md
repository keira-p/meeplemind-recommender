# 🧠 MeepleMind

Understanding what makes a great board game — and helping you find your next favourite.

*“Meeple” is a common term for player tokens in board games — representing the human element at the centre of play.*

## 🎯 Overview

MeepleMind explores the factors that drive board game quality and uses those insights to power a recommendation system for discovering new games.

Starting with structured data from BoardGameGeek, this project:

- Analyses what influences game ratings and popularity
- Builds models to predict game quality
- Evolves into a system that recommends games based on user preferences

## 🧩 Problem

With thousands of board games available, discovering the right one is difficult:

- Ratings are noisy and biased towards popular games
- Similar games are not always easy to identify
- New or niche games are hard to surface

MeepleMind aims to combine data-driven insight with personalised recommendations to improve discovery.

## ⚙️ Approach

The project is structured in two stages:

**1. Understanding game quality**

- Explore relationships between mechanics, themes, and ratings
- Identify features that drive highly rated games
- Build a model to predict BayesAvgRating

**2. Recommending games**

- Use user ratings and game features
- Build similarity-based and collaborative recommendation approaches
- Combine insights into a hybrid recommender system

## 🚀 Goals

- Build a clear understanding of what makes games successful
- Develop a recommendation system grounded in both data and behaviour
- Demonstrate end-to-end ML thinking: EDA → modelling → system design → product application

## 📦 Data

The dataset is based on BoardGameGeek and includes:

- Game metadata (ratings, complexity, player count, etc.)
- Encoded features (mechanics, themes, categories)
- User ratings and interaction data

### 📦 Data Setup
This project uses board game data sourced from BoardGameGeek.

#### 🔽 Download the data
Kaggle: https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek

Download the dataset and place the files into the following folder:
raw_data/

#### 📁 Expected structure

```plaintext
project/
├── raw_data/          # raw, unprocessed data (not tracked in Git)
│   └── .gitkeep
├── notebooks/         # EDA and analysis
├── src/               # reusable code (if applicable)
```

#### ⚠️ Notes
- The raw_data/ folder is ignored by Git to avoid committing large files
- Only a placeholder (.gitkeep) is tracked to preserve structure
- If you'd like to run this project locally, ensure all required data files are placed in raw_data/

## 🛠️ Tech stack

Python (pandas, numpy, scikit-learn)

Jupyter notebooks for exploration

SQL for data extraction

(Later) recommendation models and evaluation

## 💡 Why this project

This project combines:

- Product thinking (discovery, recommendation)
- Machine learning (prediction + modelling)
- Real-world data challenges (bias, sparsity, scale)
