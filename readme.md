# Source Code for NOM.

We provide all sorce code of dynamics appeared in the paper, as well as code for data collection and code for training a NN controller. 

### Prequest:
Training a NOM needs multiprocessing approach in Linux system. Please make sure your operation system is feasible for multi-processing.



## Control Objects:


### 1. Mathmatical Model
We use a mathmatical model:

$$
\begin{aligned}
\dot{x}_1&=x_2,\\
\dot{x}_2&=x_1^3+(x_2^2+1)u,
\end{aligned}
$$


which can be discretize using Euler's method as follows ($\Delta T$=0.1s):

$$
\begin{align}
x_1(t+1)&=x_1(t)+\Delta T x_2(t),\\
x_2(t+1)&=x_2(t)+\Delta T\big(x_1(t)^3+ (x_2(t)^2+1)\big)u(t),
\end{align}
$$

We set $\mathcal{X}=[-5,5]\times[-5,5]$ and $\mathcal{R}=0$. 

You can compare your controller with our NOM-NN controller via revising following code:

    NOM_IP_YALMIP_NOM_LQR.py


### 2. Drone (Real+Simulation)
The Bibop 2 drone dynamic model we are using is following:

$$
\begin{aligned}
& \dot{x}=\left[\begin{array}{cccccc}
0 & 1 & 0 & 0 & 0 & 0 \\
0 & -0.0527 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & -0.0187 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & -1.7873
\end{array}\right] x+\left[\begin{array}{ccc}
0 & 0 & 0 \\
-5.4779 & 0 & 0 \\
0 & 0 & 0 \\
0 & -7.0608 & 0 \\
0 & 0 & 0 \\
0 & 0 & -1.7382
\end{array}\right] u, \\
& y=\left[\begin{array}{llllll}
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0
\end{array}\right] x,
\end{aligned}
$$

We set $\mathcal{X}=[-1,1]\times[-1,1]\times[0.5,2]$ and $\mathcal{R}=[0,0,1.5]$.

We provide a simulation code for the drone as well. You can compare your controller with our NOM-NN controller via simulation:

    drone_simulation_random_point.m

#### Paper Figures for Drone:

You can obtain NOM paper figures directly by running follwoing code:

open 'PAPER_IMPORTANT_NOM' folder and run 

    cd PAPER_IMPORTANT_NOM
    python3 dataplot_NN_LQR.py

and:

    python3 NOM_IP_YALMIP_NOM_LQR.py

## One more thing
If you find this helpful, please cite our paper:

If you find any bugs, please feel free to publish issues in github.