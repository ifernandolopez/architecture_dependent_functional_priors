# Implicit Functional Priors in PINN-Based Control

This repository contains the source code used in the numerical experiments of the paper:

**Neural Architectures as Functional Priors in Scientific Machine Learning Control Problems**

The code compares different neural architectures for learning controls in dynamical systems governed by ordinary differential equations. In particular, it studies whether the choice of neural architecture influences not only the accuracy of the learned solution, but also the functional structure of the learned control.

## Systems studied

The repository includes two controlled dynamical systems:

1. A linear RLC electrical circuit.
2. A nonlinear Duffing-type oscillator.

For each system, the code compares four PINN-based architectural combinations:

* State MLP / Control MLP
* State MLP / Control FourierKAN
* State FourierKAN / Control MLP
* State FourierKAN / Control FourierKAN

The linear RLC case also includes the classical minimum-energy control computed through the finite-time controllability Gramian. The nonlinear Duffing case uses a classical numerical control baseline obtained by direct optimization of a discretized control signal.

## Installation

```bash
git clone https://github.com/<user>/implicit_functional_priors_pinn_control.git
cd implicit_functional_priors_pinn_control
pip install -r requirements.txt
```

## Running the experiments

### Linear RLC experiment

```bash
python src/linear_rlc.py
```

This script generates the control profiles, phase-space trajectories, Fourier spectra, control energy, smoothness, spectral centroid, and training loss for the linear RLC system.

### Nonlinear Duffing experiment

```bash
python src/nonlinear_duffing.py
```

This script generates the corresponding figures and quantitative metrics for the nonlinear Duffing system.

## Outputs

The scripts generate figures in both PDF and PNG format, including:

* control profiles,
* phase-space trajectories,
* Fourier spectra,
* control energy,
* control smoothness,
* spectral centroid,
* training loss.

## Reproducibility

The experiments use fixed random seeds. The default configuration follows the numerical setup described in the paper:

* 15000 training epochs for the PINN models,
* 900 collocation points,
* Adam optimizer,
* MLP and FourierKAN-like architectures,
* separated state and control networks.

## Citation

If you use this code, please cite the associated paper:

Rubio Herranz, S., Lopez Hernandez, F. C., and Lopez Montes, A.
**Neural Architectures as Functional Priors in Scientific Machine Learning Control Problems**.

## Authors

Sonia Rubio Herranz
Fernando Carlos Lopez Hernandez
Antonio Lopez Montes
