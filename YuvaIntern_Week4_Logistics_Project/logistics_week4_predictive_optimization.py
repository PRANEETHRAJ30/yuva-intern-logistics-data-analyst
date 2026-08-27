"""
Yuva Intern - Logistics Data Analyst - Week 4
Predictive Modeling and Optimization
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import linprog

# 1. Load hypothetical logistics data
df = pd.read_csv("hypothetical_logistics_week4_dataset.csv")

# 2. Define prediction target and features
target = "Delivery_Time_days"
features = ["Region","Transport_Mode","Priority","Distance_km","Shipment_Volume_kg",
            "Traffic_Index","Weather_Risk_Index","Warehouse_Load_pct",
            "Vehicle_Utilization_pct","Daily_Orders"]
X = df[features]
y = df[target]

categorical = ["Region","Transport_Mode","Priority"]
numerical = [c for c in features if c not in categorical]
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numerical)
])

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Baseline model: Linear Regression
linear = Pipeline([
    ("pre", preprocess),
    ("model", LinearRegression())
])
linear.fit(X_train, y_train)
linear_pred = linear.predict(X_test)

# 5. Non-linear model: Random Forest
rf = Pipeline([
    ("pre", preprocess),
    ("model", RandomForestRegressor(
        n_estimators=250, random_state=42, min_samples_leaf=2
    ))
])
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 6. Evaluation helper
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

evaluate("Linear Regression", y_test, linear_pred)
evaluate("Random Forest", y_test, rf_pred)

# 7. 5-fold cross-validation for the selected model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = np.sqrt(-cross_val_score(rf, X, y, scoring="neg_mean_squared_error", cv=kf))
print("CV RMSE mean:", cv_rmse.mean())
print("CV RMSE std:", cv_rmse.std())

# 8. Hyperparameter tuning (secondary candidate)
param_grid = {
    "model__n_estimators": [150, 250],
    "model__max_depth": [None, 10],
    "model__min_samples_leaf": [1, 2]
}
grid = GridSearchCV(rf, param_grid, cv=3, scoring="neg_root_mean_squared_error")
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
rf_tuned_pred = best_model.predict(X_test)
evaluate("Tuned Random Forest", y_test, rf_tuned_pred)
print("Best parameters:", grid.best_params_)
print("Final selected model: Linear Regression")

# 9. Simple vehicle allocation optimization example
regions = ["North", "South", "East", "West"]
demand = np.array([82, 98, 76, 64])
risk = np.array([0.72, 0.86, 0.91, 0.58])
cost = np.array([1.00, 1.15, 1.22, 0.92])
capacity_per_vehicle = 30
available_vehicles = 9
benefit = risk * demand / capacity_per_vehicle
objective = cost - 1.3 * benefit / benefit.max()

result = linprog(
    objective,
    A_ub=np.array([np.ones(4)]),
    b_ub=np.array([available_vehicles]),
    bounds=[(0, int(np.ceil(demand[i]/capacity_per_vehicle))) for i in range(4)],
    method="highs"
)

if result.success:
    allocation = np.floor(result.x + 1e-9).astype(int)
    while allocation.sum() < available_vehicles:
        gains = benefit - cost
        choices = [i for i in range(4) if allocation[i] < np.ceil(demand[i]/capacity_per_vehicle)]
        if not choices:
            break
        best_i = max(choices, key=lambda i: gains[i])
        allocation[best_i] += 1
    print(dict(zip(regions, allocation)))
