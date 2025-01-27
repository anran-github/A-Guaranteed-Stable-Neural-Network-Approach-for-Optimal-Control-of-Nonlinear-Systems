Here's the revised Markdown file formatted for GitHub's README style:

```markdown
# Source Code for NOM

We provide all source code for the dynamics presented in the paper, as well as the code for data collection and training a neural network (NN) controller.

---

### Prerequisite

Training a NOM requires a multiprocessing approach on a Linux system. Please ensure your operating system supports multiprocessing.

---

## Control Objects

### 1. Mathematical Model

We use a mathematical model:

$$
\begin{aligned}
\dot{x}_1 &= x_2, \\
\dot{x}_2 &= x_1^3 + (x_2^2 + 1)u,
\end{aligned}
$$

which can be discretized using Euler's method as follows (with $\Delta T=0.1s$):

$$
\begin{aligned}
x_1(t+1) &= x_1(t) + \Delta T x_2(t), \\
x_2(t+1) &= x_2(t) + \Delta T \big(x_1(t)^3 + (x_2(t)^2 + 1)u(t)\big).
\end{aligned}
$$

We set the state space $\mathcal{X} = [-5,5] \times [-5,5]$ and the reference $\mathcal{R} = 0$. 

You can compare your controller with our NOM-NN controller by modifying the following code:

```
NOM_IP_YALMIP_NOM_LQR.py
```

---

### 2. Drone (Real + Simulation)

The Bebop 2 drone dynamic model we use is as follows:

$$
\begin{aligned}
\dot{x} &= 
\begin{bmatrix}
0 & 1 & 0 & 0 & 0 & 0 \\
0 & -0.0527 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & -0.0187 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & -1.7873
\end{bmatrix}
x +
\begin{bmatrix}
0 & 0 & 0 \\
-5.4779 & 0 & 0 \\
0 & 0 & 0 \\
0 & -7.0608 & 0 \\
0 & 0 & 0 \\
0 & 0 & -1.7382
\end{bmatrix}
u, \\
y &= 
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0
\end{bmatrix}
x.
\end{aligned}
$$

We set the state space $\mathcal{X} = [-1,1] \times [-1,1] \times [0.5,2]$ and the reference $\mathcal{R} = [0,0,1.5]$.

Simulation code for the drone is also provided. You can compare your controller with our NOM-NN controller via simulation:

```
drone_simulation_random_point.m
```

---

#### Paper Figures for Drone

You can reproduce the figures from the NOM paper directly by running the following code:

Navigate to the `PAPER_IMPORTANT_NOM` folder and run:

```bash
cd PAPER_IMPORTANT_NOM
python3 dataplot_NN_LQR.py
```

and:

```bash
python3 NOM_IP_YALMIP_NOM_LQR.py
```

---

## One More Thing

If you find this project helpful, please cite our paper.

If you encounter any bugs, feel free to open an issue on GitHub.
```

### Key Improvements:
1. Added proper headers (`#`, `##`, `###`) for better sectioning.
2. Used GitHub-supported code blocks (` ``` `) for file names and code snippets.
3. Removed typos and improved readability (e.g., "sorce" → "source", "follwoing" → "following").
4. Enhanced mathematical expressions for better rendering with GitHub's LaTeX support (`$...$` for inline and `$$...$$` for blocks).