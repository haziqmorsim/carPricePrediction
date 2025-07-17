from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from bert_embedding import bert_embedding

# Load your dataset
df = pd.read_csv('used_cars_rich_data.csv')

# Define target and features
X = df.drop('Price', axis=1)
y = df['Price']

# Identify categorical columns
cat_features = X.select_dtypes(include=['object']).columns.tolist()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoost Regressor
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='MAE',
    cat_features=cat_features,
    verbose=200
)

# Train model
cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

# Predict and evaluate
preds = cat_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"\nCatBoost Regressor MAE: RM {mae:.2f}")

# Save model
joblib.dump(cat_model, 'catboost_model.pkl')

# Plot feature importance
cat_model.plot_importance()
plt.show()

# Merge BERT features with tabular features
X_tabular = df.drop(columns=['Price'])
X_combined = np.concatenate([X_tabular.values, bert_embedding], axis=1)

# Train Regression model on combined features
reg = GradientBoostingRegressor()
reg.fit(X_combined, y_train)