"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- Emma Adan
- Quinn Downey
- Allison Duran
- Nyleah Jones

Dataset: cubs_hitting_stats.csv
Predicting: Cubs hitters' OPS
Features: Batting average (BA), Walking percentage (BB%), and ISO
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def load_and_explore_data(filename):
    """
    Load your dataset and print basic information
    """
    print("=" * 70)
    print("LOADING AND EXPLORING DATA")
    print("=" * 70)
    
    data = pd.read_csv(filename)
    
    print("=== Cubs Hitting Stats Data ===")
    print(f"\nFirst 5 rows:")
    print(data.head())
    
    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    
    print(f"\nBasic statistics:")
    print(data.describe())
    
    print(f"\nColumn names: {list(data.columns)}")
    
    return data


def visualize_data(data):
    """
    Create visualizations to understand your data
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Hitting Stats vs. OPS', fontsize=16, fontweight='bold', fontfamily='sans-serif')
    
    # Plot 1: ISO vs OPS
    axes[0, 0].scatter(data['ISO'], data['OPS'], color='mediumturquoise', alpha=0.6)
    axes[0, 0].set_xlabel('Isolated Power')
    axes[0, 0].set_ylabel('On-Base + Slugging')
    axes[0, 0].set_title('ISO vs OPS', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: BA vs OPS
    axes[0, 1].scatter(data['BA'], data['OPS'], color='hotpink', alpha=0.6)
    axes[0, 1].set_xlabel('Batting Average')
    axes[0, 1].set_ylabel('On-Base + Slugging')
    axes[0, 1].set_title('BA vs OPS', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: BB% vs OPS
    axes[1, 0].scatter(data['BB%'], data['OPS'], color='crimson', alpha=0.6)
    axes[1, 0].set_xlabel('Walking Percentage')
    axes[1, 0].set_ylabel('On-Base + Slugging')
    axes[1, 0].set_title('BB% vs OPS', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hitting_features.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature plots saved as 'hitting_features.png'")
    plt.show()


def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)
    
    feature_columns = ['ISO', 'BA', 'BB%']
    feature_names = ['ISO', 'BA', 'BB%']  
    target_column = 'OPS'  
    X = data[feature_columns]
    y = data[target_column]
    
    print(f"\n=== Feature Preparation ===")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"\nFeature columns: {list(X.columns)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    print(f"\n=== Data Split ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test, feature_names  


def train_model(X_train, y_train, feature_names): 
    """
    Train the linear regression model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"\n=== Model Training Complete ===")
    print(f"Intercept: {model.intercept_:.4f}")  
    print(f"\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.4f}")
    
    print(f"\nEquation:")
    equation = f"OPS = "  
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        if i == 0:
            equation += f"{coef:.4f} × {name}"
        else:
            equation += f" + ({coef:.4f}) × {name}"
    equation += f" + {model.intercept_:.4f}"
    print(equation)
    
    return model


def evaluate_model(model, X_test, y_test, feature_names): 
    """
    Evaluate model performance
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2*100:.2f}% of OPS variation")  
    
    print(f"\nRoot Mean Squared Error: {rmse:.4f}")  
    print(f"  → On average, predictions are off by {rmse:.4f}") 
    
    # Feature importance (absolute value of coefficients)
    print(f"\n=== Feature Importance ===")
    feature_importance = list(zip(feature_names, np.abs(model.coef_)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.4f}")
    
    return predictions


def make_prediction(model, iso, ba, bb): 
    """
    Make a prediction for a new example
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    hitting_features = pd.DataFrame([[iso, ba, bb]],
                                  columns=['ISO', 'BA', 'BB%'])
    predicted_ops = model.predict(hitting_features)[0]
    print(f"\n=== New Prediction ===")
    print(f"Hitting specs: ISO={iso}, BA={ba}, BB%={bb}")
    print(f"Predicted OPS: {predicted_ops:.3f}")
    return predicted_ops


if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data('cubs_hitting_stats.csv')
    
    # Step 2: Visualize
    visualize_data(data)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test, feature_names = prepare_and_split_data(data) 
    
    # Step 4: Train
    model = train_model(X_train, y_train, feature_names)
    
    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test, feature_names)  
    
    # Step 6: Make a prediction with example values
    # Example: predict OPS for a player with ISO=0.200, BA=0.270, BB%=12.0
    make_prediction(model, iso=0.200, ba=0.270, bb=12.0) 
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")

