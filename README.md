# pinn_burgers

This module implements the Physics Informed Neural Network (PINN) model for Burgers' equation. Burgers' equation is given by `du/dt + u(du/dx) = nu (d^2 u)/(dx^2)`, where `nu` is the kinematic viscosity. This is a good simplification of Navier-Stokes equation where the velocity is in one spatial dimension and the external force is neglected and without any pressure gradient. We give the initial condition `u(t=0, x) = -sin(x)` and the boundary condition `u(t, x=-1,+1) = 0`. The PINN model predicts `u(t, x)` for the input `(t, x)`.

## Description

The PINN is a deep learning approach to solve partial differential equations. Well-known finite difference, volume and element methods are formulated on discrete meshes to approximate derivatives. Meanwhile, the automatic differentiation using neural networks provides differential operations directly. The PINN is the automatic differentiation based solver and has an advantage of being meshless.

The effectiveness of PINNs is validated in the following works.

* [M. Raissi, et al., Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations, arXiv: 1711.10561 (2017).](https://arxiv.org/abs/1711.10561)
* [M. Raissi, et al., Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations, arXiv: 1711.10566 (2017).](https://arxiv.org/abs/1711.10566)

In addition, an effective convergent optimizer is required to solve the differential equations accurately using PINNs. The stochastic gradient dicent is generally used in deep learnigs, but it only depends on the primary gradient (Jacobian). In contrast, the quasi-Newton based approach such as the limited-memory Broyden-Fletcher-Goldfarb-Shanno method for bound constraints (L-BFGS-B) incorporates the quadratic gradient (Hessian), and gives a more accurate convergence.

Here we implement a PINN model with the L-BFGS-B optimization for Burgers' equation.
Scripts is given as follows.

* *lib : libraries to implement the PINN model for a projectile motion.*
    * `layer.py` : computing 1st and 2nd derivatives as a custom layer.
    * `network.py` : building a keras network model.
    * `optimizer.py` : implementing the L-BFGS-B optimization.
    * `pinn.py` : building a PINN model.
    * `tf_silent.py` : suppressing tensorflow warnings
* `main.py` : main routine to run and test the PINN solver.

## Requirement

You need Python 3.6 and the following packages.

| package    | version (recommended) |
| -          | -      |
| matplotlib | 3.2.1  |
| numpy      | 1.18.1 |
| scipy      | 1.3.1  |
| tensorflow | 2.1.0  |

GPU acceleration is recommended in the following environments.

| package        | version (recommended) |
| -              | -     |
| cuda           | 10.1  |
| cudnn          | 7.6.5 |
| tensorflow-gpu | 2.1.0 |

## Usage

An example of PINN solver for Burgers' equation is implemented in `main.py`. The PINN is trained by the following procedure.

1. Building the keras network model
    ```python
    from lib.network import Network
    network = Network.build().
    network.summary()
    ```
    The following table depicts layers in the default network.
    ```
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         [(None, 2)]               0
    _________________________________________________________________
    dense (Dense)                (None, 32)                96
    _________________________________________________________________
    dense_1 (Dense)              (None, 16)                528
    _________________________________________________________________
    dense_2 (Dense)              (None, 32)                544
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 33
    =================================================================
    Total params: 1,201
    Trainable params: 1,201
    Non-trainable params: 0
    _________________________________________________________________
    ```
2. Building the PINN model (`nu` is the kinematic viscosity).
    ```python
    from lib.pinn import PINN
    pinn = PINN(network, nu).build()
    ```
3. Building training data.
    ```python
    # create training input
    tx_eqn = np.random.rand(num_train_samples, 2)          # t_eqn =  0 ~ +1
    tx_eqn[..., 1] = 2 * tx_eqn[..., 1] - 1                # x_eqn = -1 ~ +1
    tx_ini = 2 * np.random.rand(num_train_samples, 2) - 1  # x_ini = -1 ~ +1
    tx_ini[..., 0] = 0                                     # t_ini =  0
    tx_bnd = np.random.rand(num_train_samples, 2)          # t_bnd =  0 ~ +1
    tx_bnd[..., 1] = 2 * np.round(tx_bnd[..., 1]) - 1      # x_bnd = -1 or +1
    # create training output
    u_eqn = np.zeros((num_train_samples, 1))               # u_eqn = 0
    u_ini = np.sin(-np.pi * tx_ini[..., 1, np.newaxis])    # u_ini = -sin(pi*x_ini)
    u_bnd = np.zeros((num_train_samples, 1))               # u_bnd = 0
    ```

4. Optimizing the PINN model for the training data.
    ```python
    from lib.optimizer import L_BFGS_B
    x_train = [tx_eqn, tx_ini, tx_bnd]
    y_train = [ u_eqn,  u_ini,  u_bnd]
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    lbfgs.fit()
    ```
    The progress is printed as follows. The optimization is terminated for loss ~ 6.7e-5.
    ```
    Optimizer: L-BFGS-B (maxiter=5000)
    4635/5000 [==========================>...] - ETA: 17s - loss: 6.7799e-05
    ```

An example result of `main.py` is shown below.

![result_img](result_img.png)
