# Aspect-Based Sentiment Analysis (ABSA) Project

## Overview
This project implements an ABSA pipeline for movie reviews using a hybrid approach (LDA, keyword fallback, BERT). It analyzes the IMDb 50K dataset to extract aspects (e.g., acting, plot) and predict sentiments (positive/negative).

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd absa-project


# Dataset Instructions

The IMDb 50K Movie Reviews dataset is not included due to its size. To run the pipeline:

1. Download the dataset from Kaggle:
   - URL: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Place the `IMDB Dataset.csv` file in the `data/` directory.
3. Update the file path (default: `data/IMDB Dataset.csv`).