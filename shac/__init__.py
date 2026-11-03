"""
SHAC: Short-Horizon Actor-Critic, from the ICLR 2022 paper

    "Accelerated Policy Learning with Parallel Differentiable Simulation"
    Jie Xu, Viktor Makoviychuk, Yashraj Narang, Fabio Ramos,
    Wojciech Matusik, Animesh Garg, Miles Macklin.

This package is a small, self-contained, heavily-commented re-implementation
designed for study. It contains:

    shac.envs           Differentiable, batched environments (CartPole Swing-Up)
    shac.models         Actor / critic neural networks
    shac.algorithms     The SHAC training algorithm (paper Algorithm 1)
    shac.utils          Observation normalization, logging helpers
    shac.train          Command-line training entry point

The paper's official implementation lives at https://github.com/NVlabs/DiffRL
(GPU differentiable simulator "dflex" + the same algorithm). This package
replaces the simulator with ~100 lines of pure-PyTorch cart-pole dynamics so
the whole system runs on CPU or GPU with no external simulation dependency.
"""
