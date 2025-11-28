# Gradient Descent Animation

A visual demonstration of Gradient Descent optimization, built with React, Three.js, and Tailwind CSS.

## Overview

This project visualizes how Gradient Descent minimizes loss by iteratively updating weights. It focuses on:

- **Loss Landscape**: A parabolic curve ($L = w^2$) representing the loss function.
- **Optimization Process**: Watching a "ball" roll down the curve to find the minimum.
- **Learning Rate Effects**: Seeing how different learning rates affect convergence speed and stability.

## Components

- **Gradient Descent Panel**: Animated loss landscape with a ball that moves according to gradient descent updates.
- **Loss History Panel**: Graph showing how loss decreases over iterations.
- **Practice Panel**: Interactive controls for learning rate and starting weight.

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

3. Open your browser to the URL shown (usually `http://localhost:5173`).

## Experiments

Try these learning rates to see different behaviors:
- **α = 0.01**: Slow but steady convergence
- **α = 0.1**: Fast, smooth convergence
- **α = 0.5**: Very fast (may overshoot)
- **α = 0.95**: Oscillation and potential divergence

## License

MIT
