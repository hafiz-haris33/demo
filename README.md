# 🏡 House Price Prediction - California Housing

This project predicts the **median house value** in different districts of California using **machine learning techniques**. It uses the **California Housing dataset**, performs data preprocessing, feature engineering, model selection, evaluation, and deployment as a **Streamlit web application**.

---

## 📊 Dataset

- **Source**: [StatLib Repository (California Housing)](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)
- **Attributes**:
  - Longitude, Latitude
  - Housing median age
  - Total rooms, Total bedrooms
  - Population, Households
  - Median income
  - Median house value (target)

---

## 🧠 Machine Learning Workflow

### 🔹 Step 1: Data Collection
- Loaded the dataset using `fetch_california_housing()` function from `sklearn.datasets`.

### 🔹 Step 2: Train-Test Split (Early)
- Split the data using **stratified sampling** based on income category to avoid data leakage.

### 🔹 Step 3: Exploratory Data Analysis (EDA)
- Checked dataset shape, null values, data types, and distribution.
- Visualized:
  - Correlation matrix
  - Geographic plots (latitude vs longitude)
  - Histogram plots for numeric features
  - Scatter plots to observe relationships with `median_house_value`

### 🔹 Step 4: Feature Engineering
- Created new features:
  - `rooms_per_household`
  - `bedrooms_per_room`
  - `population_per_household`
- Binned median income into categories for stratified sampling.

### 🔹 Step 5: Data Preprocessing
- Used `Pipeline` and `ColumnTransformer` from `sklearn`:
  - Separate pipelines for:
    - Numerical features
    - Categorical features (e.g., one-hot encoding)
    - Derived ratio features
  - Custom transformer for clustering similarity

### 🔹 Step 6: Model Selection & Training
Trained multiple models:
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- HistGradientBoosting Regressor  
- XGBoost Regressor  

### 🔹 Step 7: Model Evaluation
- Used **Root Mean Squared Error (RMSE)** for evaluation
- Performed **cross-validation** with `cross_val_score`
- Visualized model performance comparison

### 🔹 Step 8: Hyperparameter Tuning
- Tuned top 3 models using:
  - `GridSearchCV`
  - `RandomizedSearchCV`
- Best model: **Random Forest Regressor (tuned using Random Search)**

### 🔹 Step 9: Final Model Testing
- Evaluated final model on **test set** (previously untouched)
- RMSE on test set showed excellent generalization
- Saved model using `joblib`

---

## 🚀 Streamlit Web App

### Features:
- Interactive UI built using **Streamlit**
- Allows user to input housing attributes
- Predicts median house value instantly
- Download prediction as PDF

---

## ⚙️ Tech Stack

- **Python 3.10+**
- **Pandas, NumPy, Scikit-learn**
- **XGBoost, Matplotlib, Seaborn**
- **Streamlit (App Interface)**
- **Joblib (Model saving/loading)**
- **FPDF (Generate PDF Reports)**

---

## 🖥️ How to Run the App Locally

### 1. Clone this repository
```bash
git clone https://github.com/your-username/house-price-prediction
cd house-price-prediction

2. Create a virtual environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install requirements

pip install -r requirements.txt
4. Run Streamlit App

streamlit run app.py
🌐 Deployment
App can be deployed on Streamlit Cloud:

Push your code to GitHub

Go to https://streamlit.io/cloud

Connect your GitHub repo and deploy

## 👨‍💻 Author
Hafiz Muhammad Haris Attique
Machine Learning | Data Science Enthusiast
📫 LinkedIn Profile

⭐ If you liked this project, feel free to fork and star this repo. Happy coding!






# 🏠 House Price Prediction System

This is a **Machine Learning project** that predicts the **median house value** in different districts of **California** using various features such as location, population, income, etc.

---

## 📊 Problem Statement

Predict the **median house value** using California housing dataset by applying machine learning models and data preprocessing techniques.

---

## 🧠 ML Workflow

### 1. **Data Collection**

* Dataset obtained from `fetch_california_housing()` using `sklearn.datasets`.
* Initial data exploration done using Pandas.

### 2. **Data Splitting**

* Used `train_test_split()` from Scikit-learn to split data early to avoid **data leakage**.

### 3. **Data Exploration & Visualization**

* Plotted geographical maps using `matplotlib` and `scatter_plot`.
* Studied the distribution of features like **housing\_median\_age**, **population**, and **median\_income**.
* Used correlation matrix to check feature relationships with the target.

### 4. **Feature Engineering**

* Created new attributes such as:

  * `rooms_per_household`
  * `bedrooms_per_room`
  * `population_per_household`
* Added a **custom transformer** for cluster similarity.

### 5. **Data Preprocessing**

* Built separate **pipelines** using `ColumnTransformer`:

  * Numerical pipeline
  * Categorical pipeline
  * Combined using `Pipeline` object

### 6. **Model Training**

Trained and evaluated the following models:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Histogram-based Gradient Boosting Regressor
* XGBoost Regressor

### 7. **Model Evaluation**

* Used **cross-validation** and performance metrics like **RMSE**.
* Compared training vs validation scores.
* Tuned hyperparameters using:

  * `GridSearchCV`
  * `RandomizedSearchCV`

### 8. **Final Model Selection**

* Best model: **Random Forest Regressor with Random Search CV**
* Evaluated on test set and found it performed better than on training set (no overfitting).

### 9. **Model Saving & Loading**

* Used `joblib` to save the final model as `final_model.pkl`.

---

## 🚀 Deployment with Streamlit

We converted the model into a **Streamlit App** for easy interaction.

### Steps to Run Locally

```bash
# 1. Clone the Repository
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

# 2. Create a virtual environment
python -m venv venv
# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Run Streamlit App
streamlit run app.py

# 5. Deployment on Streamlit Cloud
You can deploy this app for free using Streamlit Cloud:
Push your code to GitHub
Go to https://streamlit.io/cloud
Connect your GitHub repo
Select app.py as the main file and deploy!
```

