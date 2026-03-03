"""
Artificial Intelligence, Machine Learning and Neural Networks 
for Chemical Reaction Optimization

Using neural networks to predict optimal conditions for esterification reactions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

print("Chemical Reaction Optimization using Neural Networks")
print("Esterification: Acid + Alcohol → Ester + Water\n")

# Data Generation
np.random.seed(42)
n_samples = 600

# Generate random process conditions
temp = np.random.uniform(60, 120, n_samples)           # Temperature (°C)
catalyst = np.random.uniform(0.5, 3.0, n_samples)      # Catalyst (%)
time = np.random.uniform(1, 8, n_samples)              # Time (hours)
ratio = np.random.uniform(0.8, 1.2, n_samples)         # Acid:Alcohol ratio
agitation = np.random.uniform(200, 800, n_samples)     # Agitation (RPM)

# Realistic yield calculation based on chemical principles
yield_base = 50

# Temperature effect - optimal around 90°C
temp_effect = 0.5 * (temp - 90) - 0.01 * (temp - 90)**2

# Catalyst effect
catalyst_effect = 8 * catalyst - 1.5 * catalyst**2

# Time effect
time_effect = 5 * time - 0.3 * time**2

# Ratio effect - best at 1:1
ratio_effect = -10 * abs(1.0 - ratio)  # Penalty for deviation from 1:1

# Agitation effect
agitation_effect = 0.005 * agitation

# Combine all effects
yield_ideal = yield_base + temp_effect + catalyst_effect + time_effect + ratio_effect + agitation_effect

# Addition of random noise
noise = np.random.normal(0, 3, n_samples)
reaction_yield = yield_ideal + noise

# Clip reaction yield between realistic bounds
reaction_yield = np.clip(reaction_yield, 10, 98)

# Create data table
data = pd.DataFrame({
    'Temperature': temp,
    'Catalyst': catalyst,
    'Time': time,
    'Ratio': ratio,
    'Agitation': agitation,
    'Yield': reaction_yield
})

print(f"Created data for {n_samples} experiments")
print(f"Average yield: {reaction_yield.mean():.2f}%")
print(f"Yield range: {reaction_yield.min():.2f}% to {reaction_yield.max():.2f}%\n")

# Prepare data for neural network
X = data[['Temperature', 'Catalyst', 'Time', 'Ratio', 'Agitation']]
y = data['Yield']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training neural network...")

# Create and train neural network - increased iterations to avoid convergence warning
model = MLPRegressor(
    hidden_layer_sizes=(32, 16),  # 2 hidden layers
    max_iter=2000,  # Increased from 1000 to 2000
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Accuracy metrics
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

print("\n")
print("MODEL PERFORMANCE:")
print(f"Training R²:    {r2_train:.4f}")    
print(f"Test R²:        {r2_test:.4f}")
print(f"Training MSE:   {mse_train:.2f}")
print(f"Test MSE:       {mse_test:.2f}")
print(f"Training RMSE:  {rmse_train:.2f}")
print(f"Test RMSE:      {rmse_test:.2f}")
print(f"\nModel explains {r2_test*100:.2f}% of yield variation")
print(f"Predictions typically within ±{rmse_test:.2f} percentage points of actual yield")

# Calculate average error
errors = y_test - y_pred_test
avg_error = np.mean(np.abs(errors))
print(f"Average prediction error: {avg_error:.1f}%")

# Create results plot
plt.figure(figsize=(15, 4))

# Plot 1: Actual vs Predicted
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
plt.xlabel('Actual Yield (%)')
plt.ylabel('Predicted Yield (%)')
plt.title(f'Prediction Accuracy\nR² = {r2_test:.3f}')
plt.grid(True, alpha=0.3)

# Plot 2: Error distribution
plt.subplot(1, 3, 2)
plt.hist(errors, bins=15, alpha=0.7, color='orange', edgecolor='black')
plt.axvline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Prediction Error (%)')
plt.ylabel('Number of Tests')
plt.title('Prediction Errors')
plt.grid(True, alpha=0.3)

# Plot 3: Training convergence
plt.subplot(1, 3, 3)
if hasattr(model, 'loss_curve_'):
    plt.plot(model.loss_curve_)
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Convergence')
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'Loss curve not available', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Training Convergence')

plt.tight_layout()
plt.savefig('reaction_results.png', dpi=150, bbox_inches='tight')
plt.show()

# Find best conditions
print("\nSearching for optimal conditions...")

# Test many random combinations
test_combinations = np.random.uniform(
    low=[60, 0.5, 1, 0.8, 200],
    high=[120, 3.0, 8, 1.2, 800],
    size=(2000, 5)
)

# Convert to DataFrame to maintain feature names and avoid warning
test_df = pd.DataFrame(test_combinations, columns=X.columns)
test_scaled = scaler.transform(test_df)

predicted_yields = model.predict(test_scaled)

best_index = np.argmax(predicted_yields)
best_conditions = test_combinations[best_index]
best_yield = predicted_yields[best_index]

print("\nOPTIMAL CONDITIONS FOUND:")
print(f"Temperature:    {best_conditions[0]:.1f}°C")
print(f"Catalyst:       {best_conditions[1]:.2f}%")
print(f"Time:           {best_conditions[2]:.1f} hours")
print(f"Acid:Alcohol:   {best_conditions[3]:.2f}")
print(f"Agitation:      {best_conditions[4]:.0f} RPM")

print(f"Expected Yield: {min(best_yield, 100):.1f}%")
