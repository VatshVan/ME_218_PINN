# Inverse Elasticity PINN via Direct Hybrid Collocation (DHC)

This repository contains the implementation of a **Physics-Informed Neural Network (PINN)** framework designed for the inverse identification of spatially-varying Young's moduli in additively manufactured lattice structures. 
[Go to detailed Math about this project](https://vatshvan.github.io/ME_218_PINN/DHC_PINN_Lecture.pdf)

## Overview
Traditional Digital Image Correlation (DIC) data assimilation often relies on Radial Basis Function (RBF) pre-interpolation, which can lead to non-physical displacement gradients (e.g., strains exceeding 26,000%) due to a lack of mechanical constraints.

The **Direct Hybrid Collocation (DHC)** paradigm eliminates RBF pre-processing by employing the neural network itself as the direct spatial interpolator. Sparse DIC blocks act as **Dirichlet anchors**, while a dense **Sobol quasi-random collocation cloud** enforces the 2D Cauchy Momentum PDE across the domain.

## Key Features
* **Bifurcated Architecture**: Independent branches for kinematics (space-time) and constitutive behavior (space-only).
* **Adaptive Lagrangian Optimization**: Training is posed as a constrained saddle-point problem using simultaneous Adam primal descent and dual ascent over six learned Lagrange multipliers.
* **Spectral Bias Mitigation**: Utilizes Random Fourier Feature (RFF) encoding with distinct scales to capture fine kinematic details while maintaining a smooth modulus field.
* **Positivity Enforcement**: A softplus offset construction ensures the identified Young’s modulus $E(x,y)$ remains physically valid ($E > 0$).
* **Automated Pipeline**: Includes Savitzky-Golay temporal smoothing, rolling $R^2$ Hookean regime isolation, and dimensional restoration via the chain rule.

## Technical Specifications
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Epochs** | 10,000 | Total training iterations |
| **Collocation Points** | 10,000 | Sobol quasi-random samples |
| **Optimizer** | Adam | Primal/Dual learning rate of $10^{-3}$ |
| **Activation** | SiLU | Smooth $C^{\infty}$ function for higher-order derivatives |
| **Physics** | 2D Plane Stress | Cauchy Momentum Balance |

## Installation and Execution (GPU Only)

1. **Initialize and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

2. **Install Requirements:**
   ```bash
   pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
   pip install pandas numpy scipy matplotlib
   ```

3. **Run the Program:**
   ```bash
   python A_DHC.py
   ```

## Results
The framework was validated on 80% Lines, 60% Gyroid, and 80% Gyroid specimens. 
* **Scale Effect Identification**: PINN-recovered local stiffness ($E_{PINN}$) was found to be 1.4x to 1.9x higher than macroscopic UTM-averaged compliance ($E_{UTM}$), capturing intermediate-scale deformation physics.
* **Stability**: Successfully suppressed oscillatory artifacts and non-physical divergences inherent in standard interpolation techniques.

## Reference
Van, V., et al. (2026). *Direct Hybrid Collocation PINN (DHC-PINN/IE-PINN) for Inverse Identification of Spatially-Varying Elastic Moduli in Additively Manufactured Lattice Structures*.
