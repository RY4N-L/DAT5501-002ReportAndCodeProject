# -- Decision Tree Regressor on final_dataset.csv dataset -- ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import time
from scipy.stats import randint

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.tree import plot_tree, DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted


def main():
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
        'age', 'usage_intensity_norm', 'original_sales', 'entry_price'
    ]

    # Check for proxy leakage
    plot_correlation_heatmap(df, numeric_features)

    # Define features and target for the model
    features = [
        'brand', 'colour', 'bodytype', 
        'gearbox','fuel', 'adv_month', 'mileage', 
        'mpg', 'seats', 'doors', 'power', 'engine',
        'speed', 'age', 'usage_intensity_norm', 'original_sales'
    ]

    target = 'price'

    X, y = feature_selection(df, features, target)

    # Create lists for numerical and catagorical features to train
    categorical_features_to_train = [f for f in features if f in categorical_features]
    numeric_features_to_train = [f for f in features if f in numeric_features]

    # Split data into test/train sets, encode categoricals using one hot encoding (ohe) and pass numeric through
    preprocess, X_train, X_test, y_train, y_test = split_and_encode(X, y, categorical_features_to_train, numeric_features_to_train)
    
    # Get all feature names (ohe + numerical) and count
    all_features = show_feature_names(preprocess, categorical_features_to_train, numeric_features_to_train, X_train) # Get feature names
    print (f"Number of ohe catagoric features + numeric features: {len(all_features)}")
    
    # Train models
    dtr_model = find_or_train_model("final_decision_tree.pkl", "decision_tree_regressor", preprocess, X_train, y_train, X_test, y_test)
    gb_model = find_or_train_model("final_gradient_boosting.pkl", "gradient_boosting", preprocess, X_train, y_train, X_test, y_test)
    rf_model = find_or_train_model("final_random_forest.pkl", "random_forest", preprocess, X_train, y_train, X_test, y_test)
    rf_tuned_model = find_or_train_model("final_random_forest_tuned.pkl", "random_forest_tuned", preprocess, X_train, y_train, X_test, y_test)

    # Plot feature importance
    plot_feature_importance(all_features, dtr_model, "decision_tree_regressor", "final_dtr_feature_importance.png")
    plot_feature_importance(all_features, gb_model, "gradient_boosting", "final_gb_feature_importance.png")
    plot_feature_importance(all_features, rf_model, "random_forest", "final_rf_feature_importance.png")
    plot_feature_importance(all_features, rf_tuned_model, "random_forest_tuned", "final_rf_tuned_feature_importance.png")

    # Analyse Errors 
    a_preds, a_mae, a_rmse, a_r2 = test_model(rf_tuned_model, X_test, y_test)
    a_errors = np.abs(y_test - a_preds)
    a_relative_error = a_errors / y_test

    # Plot error graphs
    plot_model_analysis(X_test, y_test, a_preds, a_errors, a_relative_error, "brand_level_bias.png", a_mae)

    # Plot tree
    #plot_decision_tree(dtr_model, "decision_tree_regressor", all_features, "final_decision_tree_plot.png")

    #Tune hyperparameters (tuned once then commented out)
    # values, rf_tuned, params_tuned, mae_tuned, rmse_tuned, r2_tuned = tune_hyperparams(rf_model, "random_forest", X_train, y_train, X_test, y_test)
    # save_model(rf_tuned, "final_random_forest_tuned.pkl")
    
    # # Save tuned metrics (uncomment when tuning model)
    # results = {
    #     "model_type": "random_forest_tuned",
    #     "mae": mae_tuned,
    #     "rmse": rmse_tuned,
    #     "r2": r2_tuned,
    #     "hyperparameters": params_tuned,
    # }
    # save_metrics(results, "final_rf_tuned.json")

def plot_correlation_heatmap(df, numeric_features, figsize=(15, 8), fig_name = "correlation_heatmap.png"):
    """
    Plots a correlation heatmap for the numeric features in the dataset
    and prints the full correlation matrix.
    """
    # Select only numeric columns
    numeric_df = df[numeric_features]

    # Compute correlation matrix
    corr_matrix = numeric_df.corr()

    # Print correlation values
    print("\n=== Correlation Matrix ===\n")
    print(corr_matrix)

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot_kws={"size": 12},
        annot=True,          # show values inside the heatmap
        fmt=".2f",           # format numbers
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig(f"figures/{fig_name}", dpi=600, bbox_inches='tight')
    #plt.show()

    return corr_matrix


def find_or_train_model(model_file: str, model_name: str, preprocess, X_train, y_train, X_test, y_test):
    # Check if model has already been trained, if not train the relevant model 
    
    model_path = os.path.join(f"models/{model_file}")
    metrics_path = f"models/final_{model_name}_metrics.json"
    if os.path.exists(model_path) and os.path.exists(metrics_path):
        model = load_model(model_file)
        print(f"Loaded existing baseline {model_name} model.")
    
    else:
        print(f"No saved model found — training baseline {model_name} model, saving to {model_file}...")

        # Decision Tree
        model = run_pipeline(preprocess, X_train, y_train, model_name)

        # Test model for baseline
        preds, mae, rmse, r2 = test_model(model, X_test, y_test)

        # Save model
        save_model(model, model_file)
        hyperparams = model.named_steps[model_name].get_params()

        # Save metrics
        metrics = {
            "model_type": model_name,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "hyperparameters": hyperparams
        }
        save_metrics(metrics, f"final_{model_name}_metrics.json")
    
    return model


def save_model(model, filename:str):
    os.makedirs("models", exist_ok=True)
    filepath = os.path.join("models", filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filename:str):
    filepath = os.path.join("models", filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved model found at {filepath}")
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

def save_metrics(metrics: dict, filename: str):
    os.makedirs("models", exist_ok=True)
    filepath = os.path.join("models", filename)

    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {filepath}")


def get_safe_data(path = "data/processed/final_dataset.csv"):
    # Access processed data that has not been flagged and is within the specified price
    df = pd.read_csv(path)
    
    price_filter = df["price"] < 500000

    safe_df =  df[(~df["is_flagged"]) & price_filter]

    #print(safe_df.max())

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
            ('num', 'passthrough', num_features) # models don't require scaling of numeric features
        ]
    )

    return preprocess, X_train, X_test, y_train, y_test

def show_feature_names(preprocess: ColumnTransformer, cat_features:list, num_features: list, X_train):
    # Fit only if not already fitted
    try:
        check_is_fitted(preprocess)
    except:
        preprocess.fit(X_train)

    # Get names from the OneHotEncoder to check features
    ohe = preprocess.named_transformers_['cat']
    ohe_features = ohe.get_feature_names_out(cat_features)

    # print all features
    all_features = list(ohe_features) + num_features
    
    return all_features

def run_pipeline(preprocess: ColumnTransformer, X_train, y_train, model_name:str):
    # Runs full pipeline based on the model selected - preprocessing + model
    start = time.time()

    match model_name:
        case "decision_tree_regressor":
            model = Pipeline(steps=[
                ('preprocess', preprocess),
                ('decision_tree_regressor', DecisionTreeRegressor(random_state=10)
                )
            ])
        case "random_forest":
            model = Pipeline(steps=[
                ('preprocess', preprocess),
                ('random_forest', RandomForestRegressor(n_jobs=-1, random_state=10))
            ])
        case "random_forest_tuned":
            model = Pipeline(steps=[
                ('preprocess', preprocess),
                ('random_forest_tuned', RandomForestRegressor(
                    n_jobs=-1, 
                    max_depth = 29,
                    max_features = None,
                    min_samples_leaf = 2,
                    min_samples_split = 6,
                    n_estimators = 47,
                    random_state = 10))
            ])

        case "gradient_boosting":
            model = Pipeline(steps=[
                ('preprocess', preprocess),
                ('gradient_boosting', GradientBoostingRegressor(random_state=10))
            ])


    model.fit(X_train, y_train)
    end = time.time()
    print(f"Training time for {model_name}: {end - start:.2f} seconds")

    return model

def test_model(model, X_test, y_test):
    preds = model.predict(X_test)
    
    # Calculate error metrics
    mae = mean_absolute_error(y_test, preds) # How many pounds I am off by on average
    mse = mean_squared_error(y_test, preds) # Like mae but Penalises very bad predictions
    rmse = np.sqrt(mse) # Like mae but Penalises very bad predictions (close to mae = stable model)
    r2 = r2_score(y_test, preds) # How much variation in price my model explains - close to 1 = good, close to 0 = bad

    return preds, mae, rmse, r2

def plot_decision_tree(model, model_name, all_features, fig_name:str):
    # Plot tree graphically

    # Extract tree to plot graphically
    model_step = model.named_steps[model_name]

    # Format and plot tree
    plt.figure(figsize=(15,5), dpi=600)
    plot_tree(
        model_step, 
        feature_names = all_features,
        filled=True,
        max_depth = 3,
        fontsize=6)
    
    plt.savefig(f"figures/{fig_name}", dpi=600, bbox_inches='tight')
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

def plot_feature_importance(all_features:list, model, model_name:str, fig_name:str):
    # Plot feature importance horizontal bar chart

    model_step = model.named_steps[model_name]

    # Find feature importances
    feature_names = all_features
    importances = model_step.feature_importances_

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

    plt.xlabel("Importance", fontweight="bold")
    plt.ylabel("Feature", fontweight="bold")
    plt.title("Feature Importances (Random Forest Tuned)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"figures/{fig_name}", dpi=300, bbox_inches='tight')

    print (f"Most important features:{model_step.feature_importances_}")

def plot_error_graphs(errors, relative_error, y_test):
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

def plot_model_analysis(X_test, y_test, preds, errors, relative_error, file_name:str, mae ):
    
    # Check if model performs worse on high or low priced cars
    bins = [0, 5000, 10000, 20000, 40000, 60000, 70000, 80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 2000000]
    labels = [ f"{int(bins[i]/1000)}-{int(bins[i+1]/1000)}" for i in range(len(bins)-1) ]

    df_errors = pd.DataFrame({
        "true_price": y_test,
        "error": errors,
        "relative_error": relative_error
    })

    df_errors["price_bin"] = pd.cut(df_errors["true_price"], bins=bins, labels=labels)

    error_by_bin = df_errors.groupby("price_bin")[["error", "relative_error"]].mean()*100
   
    # --- Plot error by price bin --- #
    plt.figure(figsize=(12, 6))
    plt.bar(error_by_bin.index.astype(str), error_by_bin["relative_error"], color="steelblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Relative Error (%)", fontweight="bold", fontsize = 15)
    plt.xlabel("Price Range (thousand, £)", fontweight="bold", fontsize = 15)
    plt.title("Relative Error by Price Range", fontweight="bold", fontsize = 15)
    plt.tight_layout()
    plt.savefig("figures/relative_error_binned.png", dpi=300, bbox_inches='tight')
    #plt.show()

    plt.figure(figsize=(12, 6))
    plt.bar(error_by_bin.index.astype(str), error_by_bin["error"], color="darkorange")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Absolute Error (£)", fontweight="bold")
    plt.title("Absolute Error by Price Range", fontweight="bold")
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
    # Filter significant over/under predictions 
    threshold = mae 
    significant = brand_errors[(brand_errors.abs() > threshold) & (brand_errors.abs() < 50000 )] # Filter for errors larger than the MAE and remove outliers (extreme errors)

    print(significant)
    
    colors = ["red" if v < 0 else "green" for v in significant.values]


    # Format Graph
    plt.figure(figsize=(15, 6))
    plt.barh(significant.index, significant.values, color=colors)
    plt.xlabel("Average Prediction Error (MAE, £)", fontsize =18, fontweight="bold")
    #plt.ylabel("Brand", fontsize =15, fontweight="bold")
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.title("Brand-Level Prediction Bias", fontsize = 18, fontweight="bold")
    plt.axvline(0, color="black", linewidth=1)

    # Add value labels
    for i, v in enumerate(significant.values):
        plt.text(v,i,f"{v:.0f}",va='center',ha='left' if v >= 0 else 'right', fontsize = 12)

    plt.tight_layout()
    plt.savefig(f"figures/{file_name}", dpi=300, bbox_inches='tight')
    #plt.show()

def plot_price_box_plots(df:pd.DataFrame):
    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    fig.suptitle("Entry Price Distribution", fontsize=16)
    
    # Set x-axis title for all plots
    for ax in axes:
     ax.set_xlabel("Price (£)")

    # Segments
    budget_max = 20000
    mid_range_max = 50000
    luxury_max = 100000
    exotic_max = 500000

    # Data subsets 
    full = df['entry_price'] 
    budget = df[df['entry_price'] < budget_max]['entry_price']
    mid_range = df[(df['entry_price'] >= budget_max) & (df['entry_price'] < mid_range_max) ]['entry_price'] 
    luxury = df[(df['entry_price'] >= mid_range_max) &  (df['entry_price'] < luxury_max)]['entry_price'] 
    exotic = df[(df['entry_price'] >= luxury_max) &  (df['entry_price'] < exotic_max)]['entry_price'] 

    # Plot 1
    sns.boxplot(x=full, ax=axes[0])
    axes[0].set_title(f"Full Price Range ({len(full)} points)")
    # Plot 2
    axes[1].set_title(f"Budget (Price<£{budget_max}) ({len(budget)} points)")
    sns.boxplot(x=budget, ax=axes[1])
    # Plot 3
    sns.boxplot(x=mid_range, ax=axes[2])
    axes[2].set_title(f"Mid-range (£{budget_max}<Price<£{mid_range_max}) ({len(mid_range)} points)")
    # Plot 4
    axes[3].set_title(f"Luxury (£{mid_range_max}<Price<£{luxury_max}) ({len(luxury)} points)")
    sns.boxplot(x=luxury, ax=axes[3])
    # Plot 5
    axes[4].set_title(f"Exotic (£{luxury_max}<Price<£{exotic_max}) ({len(exotic)} points)")
    sns.boxplot(x=exotic, ax=axes[4])

    plt.savefig("figures/entry_price_box_plot_comparison.png", dpi=300, bbox_inches='tight')
    plt.subplots_adjust(top=0.85) # overall title
    plt.show()

def tune_hyperparams(model, model_type:str, X_train, y_train, X_test, y_test):
    # Hyper parameter tuning based on model
    match(model_type):
        case 'decision_tree_regressor':
            # Tune hyperparameters for decision tree regressor
            param_grid = {
                "tree__max_depth": [None, 5, 10, 20, 30],
                "tree__min_samples_split": [2, 5, 10, 20],
                "tree__min_samples_leaf": [1, 2, 4, 8],
                "tree__max_features": ["sqrt", "log2", None] 
            }
        case 'random_forest':
            param_dist = {
                "random_forest__n_estimators": randint(20, 150),
                "random_forest__max_depth": [None] + list(range(5, 31)),
                "random_forest__min_samples_split": randint(2, 11),
                "random_forest__min_samples_leaf": randint(1, 5),
                "random_forest__max_features": [None, "sqrt", "log2"]
            }


    grid = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=10
    )

    # grid = GridSearchCV(
    #     model,
    #     param_grid,
    #     cv=5,
    #     scoring="neg_mean_absolute_error", # Tune using MAE to avoid instability from high‑value outliers while still improving RMSE performance
    #     n_jobs=-1
    # )
    
    print ("Tuning started...")
    start = time.time()
    grid.fit(X_train, y_train)
    end = time.time()
    print(f"Tuning time: {end - start:.2f} seconds")

    print("Best params:", grid.best_params_)
    print("Best MAE:", -grid.best_score_)


    best_model = grid.best_estimator_
    best_params = grid.best_params_
    preds_tuned, mae_tuned, rmse_tuned, r2_tuned = test_model(best_model, X_test, y_test)
    results = grid.cv_results_

    return results, best_model, best_params, mae_tuned, rmse_tuned, r2_tuned

if __name__ == "__main__":
    main()
