# Chemical Reaction Optimization using Neural Networks

## Overview

This project builds an end-to-end machine learning pipeline to predict and optimize the yield of the esterification reaction. The system combines data simulation grounded in chemical principles with neural network modelling to create a data-driven framework for process optimization.

## Problem

In chemical process engineering, reaction yield depends on multiple interacting variables such as temperature, catalyst concentration, reaction time, reactant ratio, and agitation speed. Identifying optimal operating conditions experimentally and using traditional methods can be time-consuming and expensive.

The goal of this project was to:

- Model the relationship between process parameters and reaction yield
- Train a neural network to predict yield accurately
- Use the trained model to search for optimal operating conditions

## Methodology

### Data Simulation

- Generated 600 synthetic experiments
- Simulated yield behaviour based on reaction engineering principles
- Modelled measurement variability through Gaussian noise

### Machine Learning Pipeline

- Train-test split (80/20)
- Feature scaling using StandardScaler
- Neural network model using MLPRegressor
- Two hidden layers (32, 16 neurons)
- Model evaluation using R², MSE, and RMSE

### Optimization

- Tested 2000 new parameter combinations
- Predicted yield for each configuration
- Selected the condition that maximized expected yield

## Results

- Test R² ≈ 0.91
- Average prediction error ≈ 2.7%
- Typical prediction deviation ≈ ±3.4%
- Identified near-optimal operating conditions with predicted yield ≈ 94%

The model explains more than 90% of the variation in reaction yield and demonstrates how machine learning can support process optimization decisions.

## Key Takeaways

- Combining chemical engineering knowledge with ML improves the model's realism.
- Prediction models can be leveraged for decision-making and parameter optimization.
- Model reliability depends on disciplined validation and error analysis.
