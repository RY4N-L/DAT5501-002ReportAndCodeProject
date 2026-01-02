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

def get_safe_data():
    # Access processed data that has not been flagged and is less than £100,000
    path = "data/processed/"
    df = pd.read_csv(f"{path}ad.csv")
    print (df.columns)
    safe_df =  df[(~df["is_flagged"]) & (df["price"] < 50000)]

        
    print(safe_df['price'].min())
    print(safe_df['price'].median())
    print(safe_df['price'].max())
    print(safe_df['price'].shape)        
    return safe_df

def feature_selection(df: pd.DataFrame, feature_names: list, target: str):
    # Select features and target variable from cleaned dataset
    
    missing = [f for f in feature_names + [target] if f not in df.columns]
    
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df[feature_names], df[target]

def split_and_encode(X, y, cat_features:list, num_features:list ):
    # Split test and train data before encoding to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=10 ) # don't stratify y as it's continuous

    # Use sckit-learn pipeline for preprocessing/encoding
    preprocess = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), # ignores categories not seen during training
            cat_features),
            ('num', 'passthrough', num_features)
        ]
    )

    return preprocess, X_train, X_test, y_train, y_test

def show_feature_names(preprocess: ColumnTransformer, cat_features:list, num_features: list, X_train):
    # Show feature names (uncomment)
    preprocess.fit(X_train)

    # Get names from the OneHotEncoder to check features
    ohe = preprocess.named_transformers_['cat']
    ohe_features = ohe.get_feature_names_out(cat_features)

    # print all features
    all_features = list(ohe_features) + num_features
    
    return all_features

def run_pipeline(preprocess: ColumnTransformer, X_train, y_train):
    # Full pipeline - preprocessing + model
    model = Pipeline(steps=[
        ('preprocess', preprocess),
        #('rf', RandomForestRegressor( n_estimators=10, random_state=0, max_depth=None ))
        ('tree', DecisionTreeRegressor(max_depth=10, random_state=0))  # or classifier
    ])

    model.fit(X_train, y_train)

    return model

def test_model(model, X_test, y_test):
    preds = model.predict(X_test)
    
    # Calculate error metrics
    mae = mean_absolute_error(y_test, preds) # How many pounds I am off by on average
    mse = mean_squared_error(y_test, preds) # Like mae but Penalises very bad predictions
    rmse = np.sqrt(mse) # Like mae but Penalises very bad predictions (close to mae = stable model)
    r2 = r2_score(y_test, preds) # How much variation in price my model explains - close to 1 = good, close to 0 = bad

    return preds, mae, rmse, r2

def plot_decision_tree(model, all_features):
    # -- Plot tree graphically -- #

    # Extract tree to plot graphically
    tree_model = model.named_steps['tree']

    # Format and plot tree
    plt.figure(figsize=(20,10))
    plot_tree(
        tree_model, 
        feature_names = all_features,
        filled=True,
        max_depth = 3)
    plt.tight_layout()
    #plt.show()

def cross_validate(model, X, y, score_type = 'neg_mean_absolute_error' ):
    
    tree_model = model.named_steps['tree']
    # Define cross-validation
    kf = KFold(n_splits=9, shuffle=True, random_state=0)

    cv_mae = cross_val_score(model, X, y, cv=kf, 
                            scoring= score_type # mae default parameter - average error in pounds, cross_val_score returns r^2 as default for regressors
                            )
    cv_mae *= -1 # convert to positive

    print("MAE scores:", cv_mae)
    print("Mean MAE:", np.mean(cv_mae))
    print("Best MAE:", np.min(cv_mae)) # lowest error 
    print("Worst MAE:", np.max(cv_mae)) # highest error

def plot_feature_importance(all_features:list, model):
    
    tree_model = model.named_steps['tree']

    # Find feature importances
    feature_names = all_features
    importances = tree_model.feature_importances_

    # Create a DataFrame for readability
    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    feat_imp = feat_imp.sort_values(by="Importance", ascending=True)
    print(feat_imp)

    # Show feature importance graphically
    feat_imp_nonzero = feat_imp[feat_imp["Importance"] > 0] # only include features that have some importance
    plt.figure(figsize=(8,5))
    plt.barh(feat_imp_nonzero["Feature"], feat_imp_nonzero["Importance"])

    plt.xlabel("Importance")
    plt.title("Feature Importances (Horizontal)")
    plt.tight_layout()
    plt.show()

    print (f"Most important features:{tree_model.feature_importances_}")


if __name__ == "__main__":

    df = get_safe_data()


    # List all categorical and numeric features
    categorical_features= [
        'brand', 'genmodel', 'colour', 'bodytype', 
        'gearbox','fuel'
    ]
    numeric_features = [
        'adv_year', 'adv_month', 'reg_year', 'mileage', 
        'engine', 'price', 'power', 'tax', 
        'wheelbase', 'height', 'width', 'length'
        'mpg', 'speed', 'seats', 'doors',
        'age', 'usage_intensity_norm'
    ]

    # Define features and target for the model
    features = [
        'brand', 'colour', 'bodytype', 
        'gearbox','fuel', 'adv_month', 'mileage', 
        'mpg', 'seats', 'doors', 'power', 'engine',
        'speed', 'age', 'usage_intensity_norm'
    ]
    target = 'price'

    X, y = feature_selection(df, features, target)

    # Encode categoricals and pass numeric through
    categorical_features_to_train = [f for f in features if f in categorical_features]
    numeric_features_to_train = [f for f in features if f in numeric_features]

    preprocess, X_train, X_test, y_train, y_test = split_and_encode(X, y, categorical_features_to_train, numeric_features_to_train)
    all_features = show_feature_names(preprocess, categorical_features_to_train, numeric_features_to_train, X_train) # Get feature names
    
    # Train model
    model = run_pipeline(preprocess, X_train, y_train)

    # Test model
    preds, mae, rmse, r2 = test_model(model, X_test, y_test)
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)
    print("R²:", r2)

    # Plot feature importance
    plot_feature_importance(all_features, model)

    # Plot tree
    plot_decision_tree(model, all_features)

    # Cross validate
    cross_validate(model, X, y)

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