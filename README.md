# Housing Price Prediction – Complete ML Pipeline

**Author:** Manan Verma  
**Project Type:** End-to-End Machine Learning Regression Project  
**Dataset:** King County, Washington Housing Data (2014–2015)

## Project Overview
This project develops a complete machine learning pipeline to predict housing prices using detailed property and location features. It covers every step from data loading and cleaning to feature engineering, model comparison, and final model deployment.[file:11]

## Business Objective
- Build a robust regression model to predict house prices.  
- Understand key drivers of price such as size, quality, location, and luxury indicators.  
- Provide a reusable ML pipeline for real estate pricing and valuation use cases.[file:11]

## Dataset Description
- **Location:** King County, Washington (including Seattle)  
- **Time Period:** May 2, 2014 to May 27, 2015  
- **Total Records:** 21,613 property transactions  
- **Original Features:** 21  

### Key Original Features
- `price` – Sale price (target variable)  
- `bedrooms`, `bathrooms` – Room counts  
- `sqft_living`, `sqft_lot` – Living area and lot size  
- `floors`, `waterfront`, `view`, `condition`, `grade` – Structural and quality attributes  
- `sqft_above`, `sqft_basement` – Above/below ground area  
- `yr_built`, `yr_renovated` – Construction and renovation year  
- `zipcode`, `lat`, `long` – Location  
- `sqft_living15`, `sqft_lot15` – Neighborhood averages for living area and lot size.[file:11]

## Tools and Libraries
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn, Plotly  
- SciPy  
- Scikit-learn  
- XGBoost  
- Joblib  
- Jupyter Notebook.[file:11]

## ML Pipeline Steps

### 1. Data Loading
- Loaded `housing.csv` into a Pandas DataFrame.  
- Shape: 21,613 rows × 21 columns.[file:11]

### 2. Data Cleaning & Initial Exploration
- No missing values found.  
- No duplicate rows detected.  
- Converted `date` to datetime and extracted `year`, `month`, `day`, and `dayofweek`.  
- Verified data types and basic statistics for all features.[file:11]

### 3. Exploratory Data Analysis (EDA)
- **Target variable (`price`):**  
  - Mean ≈ 540,088, Median 450,000, Std ≈ 367,127.  
  - Highly right-skewed with heavy tails; log-transform improves normality.[file:11]  
- **Numerical features:** Analyzed distributions for area, rooms, year built, etc.  
- **Categorical-like features:** Examined distributions and average price for bedrooms, bathrooms, floors, waterfront, view, condition, and grade.[file:11]

### 4. Data Visualization
- Histograms, boxplots, and Q–Q plots for price.  
- Distribution plots for all numerical features.  
- Bar plots for categorical distributions and average price by category.  
- Correlation heatmap and bar chart of feature–price correlations.  
- Scatter plots and pairplots for top correlated features.  
- Geographic scatter plot of price by latitude and longitude.[file:11]

### 5. Feature Engineering
Created 20 additional features, including:[file:11]

- **Age & renovation:**
  - `house_age`, `years_since_renovation`, `is_renovated`  
- **Room and size features:**
  - `total_rooms`, `bath_per_bed`, `total_sqft`  
- **Space ratios and structure:**
  - `living_lot_ratio`, `has_basement`, `basement_ratio`, `above_ratio`  
- **Quality and segment:**
  - `price_per_sqft` (for analysis only), `grade_category`, `condition_category`, `has_good_view`, `is_luxury` (top 10% by price or grade ≥ 11)  
- **Time & location:**
  - `quarter`, `season`, `distance_from_center`  
- **Neighborhood comparison:**
  - `living_space_change`, `lot_size_change`.[file:11]

Resulting dataset after feature engineering: 21,613 rows × 45 columns.[file:11]

### 6. Outlier Detection & Treatment
- Used both IQR and Z-score methods on key features (`price`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_basement`).  
- Removed extreme price outliers beyond 3 standard deviations.  
- Capped extreme values for `sqft_living`, `sqft_lot`, `sqft_above` using IQR bounds.  
- Removed unrealistic entries: 0 bedrooms and properties with >10 bedrooms.  
- Final dataset size after treatment: 21,189 rows (≈1.96% rows removed).[file:11]

### 7. Feature Selection
- Dropped non-predictive or leakage-prone columns: `id`, `date`, `zipcode`, `price_per_sqft`, `grade_category`, `condition_category`, `season`.[file:11]  
- Removed highly correlated features (correlation > 0.9): `house_age`, `is_renovated`, `basement_ratio`, `above_ratio`, `has_good_view`, `quarter`.[file:11]  
- Applied `SelectKBest` with F-regression and chose the top 20 features, including:  
  - `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `view`, `grade`, `sqft_above`, `sqft_basement`, `yr_renovated`, `lat`, `sqft_living15`, `total_rooms`, `bath_per_bed`, `total_sqft`, `living_lot_ratio`, `has_basement`, `is_luxury`, `distance_from_center`, `living_space_change`.[file:11]

### 8. Data Splitting & Scaling
- Train–test split: 80% train (16,951 samples), 20% test (4,238 samples).  
- Applied `StandardScaler` on features (fit on train, transform on train and test).  
- Saved scaled matrices as DataFrames for readability.[file:11]

### 9. Model Building & Evaluation
Trained and evaluated 10 regression models:[file:11]

- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- ElasticNet  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  
- AdaBoost Regressor  
- Extra Trees Regressor  

**Evaluation metrics:**
- R² (Train and Test)  
- RMSE  
- MAE  
- MAPE.[file:11]

### 10. Model Comparison
Best-performing models on test data:[file:11]

- **Random Forest**
  - Test R² ≈ 0.8946  
  - Test RMSE ≈ 84,875  
  - Test MAE ≈ 58,941  
  - Test MAPE ≈ 12.74%  
- **XGBoost**
  - Test R² ≈ 0.8926  
- **Extra Trees**
  - Test R² ≈ 0.8924  

Linear models (Linear, Ridge, Lasso) reached around 0.79 R², confirming the importance of non-linear methods for this problem.[file:11]

### 11. Best Model Analysis – Random Forest
- **Chosen best model:** Random Forest Regressor.  
- Explains ~89.46% of price variance.  
- Residuals approximately normal with slightly larger errors for luxury properties.  
- Feature importances show:
  - `is_luxury`, `lat`, and `sqft_living` as the most influential predictors.  
- 5-fold cross-validation:
  - Mean CV R² ≈ 0.8920  
  - Very low standard deviation (≈0.0031), indicating high stability.[file:11]

### 12. Hyperparameter Tuning
- Used `RandomizedSearchCV` on Random Forest with a search space over `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.  
- Best parameters included:
  - `n_estimators` = 300  
  - `min_samples_split` = 5  
  - `min_samples_leaf` = 2  
  - `max_depth` = None.[file:11]  
- Optimized model slightly improved Test R² to ≈ 0.8951 and reduced RMSE and MAE marginally, showing that the baseline Random Forest was already close to optimal.[file:11]

### 13. Final Predictions & Examples
- Evaluated sample predictions on test data with reported actual price, predicted price, absolute difference, and percentage error.  
- Typical errors around 5–15%, with better performance in the 300K–800K price range and higher variance for luxury homes.[file:11]

### 14. Model Saving & Deployment
Saved all key artifacts using Joblib:[file:11]

- `best_housing_model.pkl` – Trained Random Forest model  
- `feature_scaler.pkl` – Fitted `StandardScaler`  
- `selected_features.pkl` – List of 20 final input features  

Usage example (Python):

```python
import joblib
import pandas as pd

# Load artifacts
model = joblib.load('best_housing_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
features = joblib.load('selected_features.pkl')

# Prepare new data (DataFrame with the same 20 features)
new_property = pd.DataFrame({...})  # must match 'features' order
new_property_scaled = scaler.transform(new_property[features])

# Predict
predicted_price = model.predict(new_property_scaled)
print(f"Predicted Price: {predicted_price:,.2f}")
