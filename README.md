# 🧠 MeepleMind

Understanding what makes a great board game - and helping you find your next favourite.

*“Meeple” is a common term for player tokens in board games, short for “my people” - representing the human element at the centre of play.*

🎯 **Goals**

- Build a clear understanding of what makes games successful
- Develop a recommendation system grounded in both data and behaviour
- Demonstrate end-to-end ML thinking: EDA → modelling → system design → product application

## 🚀 Live demo

👉 https://meeplemind-143304324442.europe-west2.run.app/

## 🎯 Overview

MeepleMind explores the factors that drive board game quality and uses those insights to power a recommendation system for discovering new games.

Starting with structured data from BoardGameGeek, this project:

- Analyses what influences game ratings and popularity
- Builds models to predict game quality
- Evolves into a production-ready recommender system

## 🧩 Problem

With thousands of board games available, discovering the right one is difficult:

- Ratings are noisy and biased towards popular games
- Similar games are not always easy to identify
- New or niche games are hard to surface

MeepleMind aims to combine data-driven insight with personalised recommendations to improve discovery.

## ⚙️ Approach

The project developed in two stages:

### 1. Understanding game quality

- Explored relationships between mechanics, themes, complexity and ratings
- Identified features associated with highly rated games
- Built models to predict `BayesAvgRating`

### 2. Recommending games

The final recommender uses an **item-based collaborative filtering** approach based on user ratings.

Rather than storing a full dense similarity matrix, it uses a **K-Nearest Neighbours (KNN) sparse neighbours representation**, which is more efficient and better suited to deployment.

At inference time, the app:

1. Takes 3–5 favourite games from the user
2. Retrieves similar neighbours for each one
3. Filters to sufficiently strong matches
4. Aggregates scores across the selected games
5. Returns ranked recommendations

## 🖥️ Application

MeepleMind is available as a **Streamlit app** where users can:

- choose 3–5 favourite games
- get ranked recommendations
- view simple match labels
- click through to BoardGameGeek for more detail

The interface is designed to be lightweight, scannable and easy to use.

## ☁️ Deployment

The app is:

- Containerised with **Docker**
- Deployed on **Google Cloud Run**
- Powered by precomputed **parquet artefacts** for faster inference

For MVP simplicity, the processed artefacts are bundled inside the container image.


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
├── data/
│   ├── raw/           # raw downloaded files, not tracked in Git
│   └── processed/     # processed artefacts used by the app
├── notebooks/         # EDA, modelling and recommender development
├── app.py             # Streamlit app
├── recommender.py     # recommender logic
├── requirements.txt
└── Dockerfile
```

#### ⚠️ Notes
- data/raw/ is ignored by Git and Docker
- The deployed app uses processed artefacts from data/processed/
- The recommender uses a processed dataset and a KNN-based sparse neighbours representation, allowing it to scale beyond the limitations of a dense similarity matrix

## 🛠️ Tech stack

Python

pandas, numpy, scikit-learn

Jupyter notebooks

Streamlit

Docker

Google Cloud Run

parquet

## 💡 Why this project

This project brings together:

- Product thinking around discovery and recommendation UX
- Machine learning for modelling and similarity-based recommendation
- Engineering decisions around sparsity, inference and deployment
- Real-world data challenges including bias, sparsity and scale

## 🔮 Potential next steps
- Scale the recommender to a larger/fuller dataset
- Add filters such as player count, play time and complexity
- Improve recommendation evaluation
- Strengthen explanation and transparency in the UI
- Move artefact storage outside the container if scale or update frequency increases
