# Movie-Recommendation-Engine

## **Project Overview**
This project builds a **Movie Recommendation System** using various algorithms to predict how a user will rate a movie based on historical data. It uses models ranging from simple baselines to advanced machine learning techniques to make accurate recommendations.

---

## **Key Features**
1. Predicts movie ratings for users.
2. Compares multiple models to find the best-performing one.
3. Incorporates user preferences (explicit ratings) and hidden patterns (implicit feedback).

---

## **Overview**
1. **Data Preparation**:
   - User-movie interaction data is converted into a sparse matrix.
   - Features like user averages, movie averages, and global averages are calculated.
   
2. **Model Training**:
   Different algorithms are trained on this data:
   - **BaselineOnly**: Predicts ratings using the average of user and movie ratings.
   - **KNN Baseline**: Uses similarity between users to predict ratings.
   - **SVD and SVDpp**: Breaks down the user-movie matrix to uncover hidden preferences.
   - **SlopeOne**: Predicts ratings based on relationships between items.
   - **XGBoost**: A machine learning model that predicts ratings based on extracted features.

3. **Evaluation**:
   - Models are evaluated using **Root Mean Square Error (RMSE)** and **Mean Absolute Percentage Error (MAPE)**.
   - The best model is chosen based on performance.

4. **Recommendations**:
   - Top movie recommendations are generated for each user using the trained models.

---

## **Project Steps**
1. **Data Loading and Preparation**:
   - User and movie IDs are mapped.
   - Data is split into training and testing sets.

2. **Feature Extraction**:
   - Extract features like user averages, movie averages, and ratings of similar users/movies.

3. **Model Training**:
   - Train models like BaselineOnly, KNN Baseline, SVD, SVDpp, and XGBoost on the training data.

4. **Model Evaluation**:
   - Compare models using RMSE and MAPE.
   - Use **GridSearchCV** to optimize hyperparameters for better performance.

5. **Make Recommendations**:
   - Generate the top 10 movies for each user using the best model.

---

## **Technologies Used**
- **Python**: For coding.
- **Libraries**:
  - `pandas`, `numpy`, `scipy`: For data manipulation.
  - `scikit-learn`: For evaluation and preprocessing.
  - `surprise`: For recommendation algorithms like KNN, SVD, SVDpp.
  - `xgboost`: For machine learning predictions.
  - `matplotlib`, `seaborn`: For visualizations.

---

## **Results**
The system evaluates multiple models to select the best one based on RMSE and MAPE scores. It then uses this model to provide personalized movie recommendations.

---
