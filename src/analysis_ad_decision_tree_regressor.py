# -- Decision Tree Regressor on ad.csv dataset -- ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.tree import plot_tree, DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def get_safe_data():
    # Access processed data that has not been flagged and is within the specified price
    path = "data/processed/"
    df = pd.read_csv(f"{path}ad.csv")
    print (df.columns)
    safe_df =  df[(~df["is_flagged"]) & (df["price"]<500000)]

        
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

def run_pipeline(preprocess: ColumnTransformer, X_train, y_train, model_name:str):
    # Full pipeline - preprocessing + model

    match model_name:
        case "tree":
            model = Pipeline(steps=[
                ('preprocess', preprocess),
                ('tree', DecisionTreeRegressor(
                    # max_depth=20, 
                    # max_features=None, 
                    # min_samples_leaf=1, 
                    # min_samples_split=10, 
                    random_state=10
                    )
                )
            ])
        case "rf":
            model = Pipeline(steps=[
                ('preprocess', preprocess),
                ('rf', RandomForestRegressor(random_state=10))
            ])

        case "gb":
            model = Pipeline(steps=[
                ('preprocess', preprocess),
                ('gb', GradientBoostingRegressor(random_state=10))
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
    plt.figure(figsize=(15,5), dpi=600)
    plot_tree(
        tree_model, 
        feature_names = all_features,
        filled=True,
        max_depth = 3,
        fontsize=6)
    
    plt.savefig("figures/decision_tree_regressor.png", dpi=600, bbox_inches='tight')
    #plt.show()

def cross_validate(model, model_type:str, X, y, score_type = 'neg_mean_absolute_error' ):
    
    type_model = model.named_steps[model_type]
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
    feat_imp_nonzero = feat_imp[feat_imp["Importance"] > 0.01] # only include features that have some importance
    plt.figure(figsize=(8,5))
    plt.barh(feat_imp_nonzero["Feature"], feat_imp_nonzero["Importance"])

    plt.xlabel("Importance")
    plt.title("Feature Importances (Horizontal)")
    plt.tight_layout()
    plt.savefig("figures/feature_importance.png", dpi=300, bbox_inches='tight')

    print (f"Most important features:{tree_model.feature_importances_}")

def plot_error_graphs(errors, relative_error):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, errors, alpha=0.4)

    plt.xlabel("True Price (£)")
    plt.ylabel("Absolute Error (£)")
    plt.title("Absolute Error vs True Price")

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("figures/absolute_error.png", dpi=300, bbox_inches='tight')
    #plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, relative_error, alpha=0.4)

    plt.xlabel("True Price (£)")
    plt.ylabel("Relative Error (fraction)")
    plt.title("Relative Error vs True Price")

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("figures/relative_error.png", dpi=300, bbox_inches='tight')
    #plt.show()

def plot_model_analysis(X_test, y_test, preds, errors, relative_error):
    
    # Check if model performs worse on high or low priced cars
    bins = [0, 5000, 10000, 20000, 40000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 2000000]
    labels = [ f"{int(bins[i]/1000)}-{int(bins[i+1]/1000)}k" for i in range(len(bins)-1) ]

    df_errors = pd.DataFrame({
        "true_price": y_test,
        "error": errors,
        "relative_error": relative_error
    })

    df_errors["price_bin"] = pd.cut(df_errors["true_price"], bins=bins, labels=labels)

    error_by_bin = df_errors.groupby("price_bin")[["error", "relative_error"]].mean()
   
    # --- Plot error by price bin --- #
    plt.figure(figsize=(12, 6))
    plt.bar(error_by_bin.index.astype(str), error_by_bin["relative_error"], color="steelblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Relative Error")
    plt.title("Relative Error by Price Range")
    plt.tight_layout()
    plt.savefig("figures/relative_error_binned.png", dpi=300, bbox_inches='tight')
    #plt.show()

    plt.figure(figsize=(12, 6))
    plt.bar(error_by_bin.index.astype(str), error_by_bin["error"], color="darkorange")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Absolute Error (£)")
    plt.title("Absolute Error by Price Range")
    plt.tight_layout()
    plt.savefig("figures/absolute_error_binned.png", dpi=300, bbox_inches='tight')
    #plt.show()


    # Brand analysis
    df_brand = pd.DataFrame({
    "brand": X_test["brand"],
    "true_price": y_test,
    "predicted_price": preds
    })

    # Check if brands get under or over predicted
    df_brand["error"] = df_brand["predicted_price"] - df_brand["true_price"]

    brand_errors = df_brand.groupby("brand")["error"].mean().sort_values()
    print(brand_errors)
    
    colors = ["red" if v < 0 else "green" for v in brand_errors.values]

    plt.figure(figsize=(10, 14))
    plt.barh(brand_errors.index, brand_errors.values, color=colors)
    plt.xlabel("Average Prediction Error (£)")
    plt.title("Brand-Level Prediction Bias")
    plt.axvline(0, color="black", linewidth=1)
    plt.tight_layout()
    plt.savefig("figures/brand_level_prediction_bias.png", dpi=300, bbox_inches='tight')
    #plt.show()

def plot_price_box_plots(df:pd.DataFrame):

    print(df['price'].describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999]))

    sns.boxplot(x=df['price'])
    plt.title("Boxplot of Car Prices") 
    plt.xlabel("Price (£)") 
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    sns.boxplot(x=df['price'], ax=axes[0])
    axes[0].set_title("Full Price Range")

    sns.boxplot(x=df[df['price'] < 100000]['price'], ax=axes[1])
    axes[1].set_title("Zoomed (< £100k)")

    axes[2].set_title("Zoomed (< 40k)")
    sns.boxplot(x=df[df['price'] < 40000]['price'], ax=axes[2])

    plt.show()

def tune_hyperparams(model, model_type:str, X_train, y_train, X_test, y_test):
    # Hyper parameter tuning based on model
    
    match(model_type):
        case 'tree':
            # Tune hyperparameters for decision tree regressor
            param_grid = {
                "tree__max_depth": [None, 5, 10, 20, 30],
                "tree__min_samples_split": [2, 5, 10, 20],
                "tree__min_samples_leaf": [1, 2, 4, 8],
                "tree__max_features": ["sqrt", "log2", None] 
            }
        case 'rf':
            param_grid = {
                "rf__n_estimators": [50, 100, 200],
                "rf__max_depth": [None, 10, 20],
                "rf__min_samples_split": [2, 5, 10],
                "rf__min_samples_leaf": [1, 2, 4]
            }


    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best MAE:", -grid.best_score_)


    best_model = grid.best_estimator_
    preds_tuned, mae_tuned, rmse_tuned, r2_tuned = test_model(best_model, X_test, y_test)

    print("Tuned MAE:", mae_tuned)
    print("Tuned RMSE:", rmse_tuned)
    print("Tuned R²:", r2_tuned)

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
        'wheelbase', 'height', 'width', 'length',
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
    print (all_features)
    print (len(all_features))
    
    # Train model
    # model = run_pipeline(preprocess, X_train, y_train, "tree")

    # Test model for baseline
    # preds, mae, rmse, r2 = test_model(model, X_test, y_test)
    # print("Mean Absolute Error:", mae)
    # print("Root Mean Squared Error:", rmse)
    # print("R²:", r2)

    # Plot feature importance
    # plot_feature_importance(all_features, model)

    # Plot tree
    # plot_decision_tree(model, all_features)

    # Cross validate
    #cross_validate(model, 'tree', X, y)

    # Tune hyper parameters
    #tune_hyperparams(model, 'tree', X_train, y_train, X_test, y_test)

    # -- Analyse Errors -- #
    # errors = np.abs(y_test - preds)
    # relative_error = errors / y_test

    # Plot error graphs
    #plot_error_graphs(errors, relative_error)
    #plot_model_analysis(X_test, y_test, preds, errors, relative_error)