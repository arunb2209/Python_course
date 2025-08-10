# Linear Regression from Scratch--Main

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

class LinearRegressionFromScratch:
    def __init__(self):
        self.slope = None
        self.intercept = None
    
    def fit(self, X, y):
        """
        Fit the linear regression model using the normal equation.
        For simple linear regression: y = mx + b
        """
        n = len(X)
        
        # Calculate slope (m) and intercept (b)
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Slope = Σ((xi - x_mean)(yi - y_mean)) / Σ((xi - x_mean)²)
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean
        
        return self
    
    def predict(self, X):
        """Make predictions using the fitted model."""
        if self.slope is None or self.intercept is None:
            raise ValueError("Model has not been fitted yet.")
        
        return self.slope * X + self.intercept
    
    def get_params(self):
        """Return the learned parameters."""
        return {'slope': self.slope, 'intercept': self.intercept}

def generate_sample_data(n_samples=100, noise=0.1):
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.uniform(0, 10, n_samples)
    y = 2.5 * X + 1.5 + np.random.normal(0, noise * 10, n_samples)
    return X, y

def plot_regression_results(X, y, y_pred_custom, y_pred_sklearn, title="Linear Regression Comparison"):
    """Plot the data points and regression lines."""
    plt.figure(figsize=(12, 8))
    
    # Sort data for better line plotting
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_pred_custom_sorted = y_pred_custom[sort_idx]
    y_pred_sklearn_sorted = y_pred_sklearn[sort_idx]
    
    # Plot data points
    plt.scatter(X, y, alpha=0.6, color='blue', label='Data Points')
    
    # Plot regression lines
    plt.plot(X_sorted, y_pred_custom_sorted, color='red', linewidth=2, label='Custom Implementation')
    plt.plot(X_sorted, y_pred_sklearn_sorted, color='green', linewidth=2, linestyle='--', label='Scikit-learn')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2
    }

def main():
    print("=== Linear Regression Algorithm Implementation ===\n")
    
    # Generate sample data
    print("1. Generating sample data...")
    X, y = generate_sample_data(n_samples=100, noise=0.1)
    print(f"Generated {len(X)} data points")
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Custom implementation
    print("\n2. Training custom linear regression model...")
    custom_model = LinearRegressionFromScratch()
    custom_model.fit(X_train, y_train)
    
    # Make predictions with custom model
    y_pred_custom = custom_model.predict(X_test)
    custom_params = custom_model.get_params()
    
    print(f"Custom Model Parameters:")
    print(f"  Slope: {custom_params['slope']:.4f}")
    print(f"  Intercept: {custom_params['intercept']:.4f}")
    
    # Scikit-learn implementation
    print("\n3. Training scikit-learn linear regression model...")
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train.reshape(-1, 1), y_train)
    
    # Make predictions with sklearn model
    y_pred_sklearn = sklearn_model.predict(X_test.reshape(-1, 1))
    
    print(f"Scikit-learn Model Parameters:")
    print(f"  Slope: {sklearn_model.coef_[0]:.4f}")
    print(f"  Intercept: {sklearn_model.intercept_:.4f}")
    
    # Calculate metrics
    print("\n4. Model Performance Metrics:")
    print("\nCustom Implementation:")
    custom_metrics = calculate_metrics(y_test, y_pred_custom)
    for metric, value in custom_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nScikit-learn Implementation:")
    sklearn_metrics = calculate_metrics(y_test, y_pred_sklearn)
    for metric, value in sklearn_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Plot results
    print("\n5. Plotting results...")
    plot_regression_results(X_test, y_test, y_pred_custom, y_pred_sklearn)
    
    # Demonstrate prediction on new data
    print("\n6. Making predictions on new data:")
    new_X = np.array([2.5, 5.0, 7.5])
    custom_predictions = custom_model.predict(new_X)
    sklearn_predictions = sklearn_model.predict(new_X.reshape(-1, 1))
    
    for i, x_val in enumerate(new_X):
        print(f"  X = {x_val}: Custom = {custom_predictions[i]:.2f}, Scikit-learn = {sklearn_predictions[i]:.2f}")

if __name__ == "__main__":
    main()


