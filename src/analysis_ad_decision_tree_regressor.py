# -- Decision Tree Regressor on ad.csv dataset -- ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Select features and target variable from cleaned dataset
path = "data/processed/"
df = pd.read_csv(f"{path}ad.csv")
safe_df =  df[df["is_flagged"] != True]

features = [
    'brand', 'colour', 'bodytype', 'mileage',
    'engine', 'gearbox', 'fuel', 'power',
    'tax', 'mpg', 'speed', 'seats', 'doors',
    'age', 'usage_intensity_norm'
]

X = safe_df[features]
y = safe_df['price']

# Split test and train data before encoding to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=10 ) # don't stratisfy y as it's continuous

# Encode categoricals and pass numeric through
categorical_features = [
    'brand', 'colour', 'bodytype', 
    'gearbox','fuel'
]

numeric_features = [
    'mileage', 'engine', 'power', 'tax', 
    'mpg', 'speed', 'seats', 'doors',
    'age', 'usage_intensity_norm'
]

# Use sckit-learn pipeline for preprocessing
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), # ignores cateories not seen during training
         categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

# # Show feature names (uncomment)
# preprocess.fit(X)

# # Get names from the OneHotEncoder to check features
# ohe = preprocess.named_transformers_['cat']
# ohe_features = ohe.get_feature_names_out(categorical_features)

# # Access numeric features
# num_features = numeric_features

# # print all features
# all_features = list(ohe_features) + num_features
# print(all_features)


# Full pipeline - preprocessing + model
model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('tree', DecisionTreeRegressor(max_depth=10, random_state=0))  # or classifier
])

model.fit(X_train, y_train)

preds = model.predict(X_test)


# -- Plot tree graphically -- #

# Extract tree to plot graphically
tree_model = model.named_steps['tree']

# Get feature names
ohe = model.named_steps['preprocess'].named_transformers_['cat']
ohe_features = ohe.get_feature_names_out(categorical_features)
all_features = list(ohe_features) + numeric_features

# Format and plot tree
plt.figure(figsize=(20,10))
plot_tree(
    tree_model, 
    feature_names = all_features,
    filled=True,
    max_depth = 3)
plt.tight_layout()
#plt.show()

# Show error metrics
mae = mean_absolute_error(y_test, preds) # How many pounds I am off by on average
mse = mean_squared_error(y_test, preds) # Like mae but Penalises very bad predictions
rmse = np.sqrt(mse) # Like mae but Penalises very bad predictions (close to mae = stable model)
r2 = r2_score(y_test, preds) # How much variantion in price my model explains - close to 1 = good, close to 0 = bad

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R²:", r2)

# # Define cross-validation
# kf = KFold(n_splits=9, shuffle=True, random_state=0)

# cv_mae = cross_val_score(model, X, y, cv=kf, 
#                           scoring= 'neg_mean_absolute_error' # mae - average error in pounds, cross validate returns r^2 as default for regressors
#                           )
# cv_mae *= -1 # convert to positive

# print("MAE scores:", cv_mae)
# print("Mean MAE:", np.mean(cv_mae))
# print("Best MAE:", np.min(cv_mae)) # lowest error 
# print("Worst MAE:", np.max(cv_mae)) # highest error

# Find feature importances
feature_names = all_features
importances = tree_model.feature_importances_

# Create a DataFrame for readability
feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feat_imp = feat_imp.sort_values(by="Importance", ascending=True)
#print(feat_imp)

# Show feature importance graphically
feat_imp_nonzero = feat_imp[feat_imp["Importance"] > 0] # only include features that have some importance
plt.figure(figsize=(8,5))
plt.barh(feat_imp_nonzero["Feature"], feat_imp_nonzero["Importance"])

plt.xlabel("Importance")
plt.title("Feature Importances (Horizontal)")
plt.tight_layout()
plt.show()

#print (f"Most important features:{clf.feature_importances_}")



# # Plot error graphs
# errors = np.abs(y_test - preds)
# relative_error = errors / y_test

# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, errors, alpha=0.4)

# plt.xlabel("True Price (£)")
# plt.ylabel("Absolute Error (£)")
# plt.title("Absolute Error vs True Price")

# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()

# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, relative_error, alpha=0.4)

# plt.xlabel("True Price (£)")
# plt.ylabel("Relative Error (fraction)")
# plt.title("Relative Error vs True Price")

# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()
