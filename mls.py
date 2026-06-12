import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from ipywidgets import interact, FloatSlider, IntSlider


# =====================================================================
# 1. UNIVERSAL BASE ENGINE (Exact Analytical Derivatives)
# =====================================================================
class PIMLS_Vectorized_Engine:
    """
    Universal 1D Vectorized Moving Least Squares Engine.
    Computes Exact Analytical Shape Functions: Phi, dPhi, and d2Phi.
    """

    def __init__(self, dt, num_nodes=25, num_colloc=75, support_factor=3.5):
        self.dt = dt
        self.num_nodes = num_nodes
        self.num_colloc = num_colloc

        self.nodes = np.linspace(0, dt, num_nodes)
        self.colloc_pts = np.linspace(0, dt, num_colloc)
        self.d_support = support_factor * (self.nodes[1] - self.nodes[0])
        self.t0 = dt / 2.0

        # Precompute global matrices
        self.Phi, self.dPhi, _ = self.compute_shape_matrices(self.colloc_pts)

    def compute_shape_matrices(self, t_arr):
        """Vectorized Tensor Engine for exact Phi, dPhi, and d2Phi."""
        K = len(t_arr)

        # Static Taylor basis centered at t0: [1, dt, 0.5 * dt^2]
        dt_nodes = self.nodes - self.t0
        P_nodes = np.vstack([np.ones_like(self.nodes), dt_nodes, 0.5 * dt_nodes ** 2])

        t_diff = t_arr[:, None] - self.nodes[None, :]
        w = np.exp(-(t_diff / self.d_support) ** 2)

        # Exact Weight Derivatives
        dw = -2 * t_diff / (self.d_support ** 2) * w
        d2w = (4 * t_diff ** 2 / self.d_support ** 4 - 2 / self.d_support ** 2) * w

        # Moment Matrices
        A = np.einsum('kn,in,jn->kij', w, P_nodes, P_nodes)
        A += 1e-8 * np.eye(3)[None, :, :]
        dA = np.einsum('kn,in,jn->kij', dw, P_nodes, P_nodes)
        d2A = np.einsum('kn,in,jn->kij', d2w, P_nodes, P_nodes)

        # Invert Systems & Matrix Calculus Product Rule
        A_inv = np.linalg.inv(A)
        dA_inv = -A_inv @ dA @ A_inv
        d2A_inv = -(dA_inv @ dA @ A_inv + A_inv @ d2A @ A_inv + A_inv @ dA @ dA_inv)

        B = np.einsum('kn,in->kin', w, P_nodes)
        dB = np.einsum('kn,in->kin', dw, P_nodes)
        d2B = np.einsum('kn,in->kin', d2w, P_nodes)

        # Evaluation basis arrays
        dt_eval = t_arr - self.t0
        P_t = np.vstack([np.ones_like(t_arr), dt_eval, 0.5 * dt_eval ** 2])
        dP_t = np.vstack([np.zeros_like(t_arr), np.ones_like(t_arr), dt_eval])
        d2P_t = np.vstack([np.zeros_like(t_arr), np.zeros_like(t_arr), np.ones_like(t_arr)])

        # 1. State Matrix (Phi)
        p_A_inv = np.einsum('ik,kij->kj', P_t, A_inv)
        phi = np.einsum('kj,kjn->kn', p_A_inv, B)

        # 2. Velocity Matrix (dPhi)
        dp_A_inv = np.einsum('ik,kij->kj', dP_t, A_inv)
        p_dA_inv = np.einsum('ik,kij->kj', P_t, dA_inv)
        dphi = np.einsum('kj,kjn->kn', dp_A_inv, B) + \
               np.einsum('kj,kjn->kn', p_dA_inv, B) + \
               np.einsum('kj,kjn->kn', p_A_inv, dB)

        # 3. Acceleration Matrix (d2Phi)
        d2p_A_inv = np.einsum('ik,kij->kj', d2P_t, A_inv)
        dp_dA_inv = np.einsum('ik,kij->kj', dP_t, dA_inv)
        p_d2A_inv = np.einsum('ik,kij->kj', P_t, d2A_inv)

        d2phi = np.einsum('kj,kjn->kn', d2p_A_inv, B) + \
                2 * np.einsum('kj,kjn->kn', dp_dA_inv, B) + \
                2 * np.einsum('kj,kjn->kn', dp_A_inv, dB) + \
                np.einsum('kj,kjn->kn', p_d2A_inv, B) + \
                2 * np.einsum('kj,kjn->kn', p_dA_inv, dB) + \
                np.einsum('kj,kjn->kn', p_A_inv, d2B)

        return phi, dphi, d2phi


# =====================================================================
# 2. PHYSICS SUBCLASSES (Standardized to return N-Dimensional arrays)
# =====================================================================
class Logistic_Window(PIMLS_Vectorized_Engine):
    """1-Dimensional ODE: dy/dt = r*y*(1 - y/K)"""

    def __init__(self, init_state, dt, r, K, num_nodes=25, num_colloc=75):
        super().__init__(dt, num_nodes, num_colloc)
        self.y0 = init_state[0]
        self.r, self.K = r, K

    def residuals(self, u_flat, reg_weight=1e-3):
        y_pred, dy_pred = self.Phi @ u_flat, self.dPhi @ u_flat
        res_data = 100.0 * (y_pred[0] - self.y0)
        res_phys = dy_pred - self.r * y_pred * (1.0 - y_pred / self.K)
        return np.concatenate([[res_data], res_phys, reg_weight * np.diff(u_flat)])

    def solve(self):
        u_guess = np.full(self.num_nodes, self.y0)
        res = least_squares(self.residuals, u_guess, method='lm', xtol=1e-7, ftol=1e-7)
        return np.array([res.x])


class LotkaVolterra_Window(PIMLS_Vectorized_Engine):
    """2-Dimensional ODE: Coupled Predator-Prey"""

    def __init__(self, init_state, dt, alpha, beta, delta, gamma, num_nodes=25, num_colloc=75):
        super().__init__(dt, num_nodes, num_colloc)
        self.x0, self.y0 = init_state
        self.alpha, self.beta = alpha, beta
        self.delta, self.gamma = delta, gamma
        self.H_init = delta * self.x0 - gamma * np.log(self.x0) + beta * self.y0 - alpha * np.log(self.y0)

    def residuals(self, u_flat, reg_weight=1e-3, H_weight=10.0):
        ux, uy = np.split(u_flat, 2)
        x_pred, y_pred = self.Phi @ ux, self.Phi @ uy
        dx_pred, dy_pred = self.dPhi @ ux, self.dPhi @ uy

        res_data_x, res_data_y = 100.0 * (x_pred[0] - self.x0), 100.0 * (y_pred[0] - self.y0)
        res_phys_x = dx_pred - (self.alpha * x_pred - self.beta * x_pred * y_pred)
        res_phys_y = dy_pred - (self.delta * x_pred * y_pred - self.gamma * y_pred)

        H_pred = self.delta * x_pred - self.gamma * np.log(np.clip(x_pred, 1e-6, None)) + \
                 self.beta * y_pred - self.alpha * np.log(np.clip(y_pred, 1e-6, None))

        return np.concatenate([[res_data_x, res_data_y], res_phys_x, res_phys_y,
                               H_weight * (H_pred - self.H_init),
                               reg_weight * np.diff(ux), reg_weight * np.diff(uy)])

    def solve(self):
        u_guess = np.concatenate([np.full(self.num_nodes, self.x0), np.full(self.num_nodes, self.y0)])
        res = least_squares(self.residuals, u_guess, method='lm', xtol=1e-7, ftol=1e-7)
        return np.vstack(np.split(res.x, 2))


# =====================================================================
# 3. GENERALIZED N-DIMENSIONAL DASHBOARD AUTOMATOR
# =====================================================================
class PIMLS_Dashboard:
    """Constructs time-marching UI panels and extracts true kinematic profiles."""

    @staticmethod
    def run(SolverClass, init_state, T_max, state_names, true_ode_func=None, **physics_params):
        num_vars = len(init_state)
        target_dt = 2.5
        num_windows = int(np.ceil(T_max / target_dt))
        window_dts = np.full(num_windows, T_max / num_windows)

        global_t = []
        global_states = [[] for _ in range(num_vars)]

        # Track continuous true kinematics for Primary Variable (Var 0)
        global_kinematics = [[] for _ in range(3)]

        curr_state = np.array(init_state, dtype=float)
        accumulated_time = 0.0

        for w in range(num_windows):
            dt = window_dts[w]

            solver = SolverClass(init_state=curr_state, dt=dt, **physics_params)
            U_opt = solver.solve()

            t_local = np.linspace(0, dt, 40)

            # Request all 3 exact shape matrices
            phi, dphi, d2phi = solver.compute_shape_matrices(t_local)

            # Map parameters to exact continuous physical states
            window_states = U_opt @ phi.T

            # Map parameters to exact continuous kinematics for Var 0
            vel_v0 = U_opt[0] @ dphi.T
            acc_v0 = U_opt[0] @ d2phi.T

            # Boundary Stitching
            if w > 0:
                t_local = t_local[1:]
                window_states = window_states[:, 1:]
                vel_v0 = vel_v0[1:]
                acc_v0 = acc_v0[1:]

            global_t.extend(t_local + accumulated_time)
            for i in range(num_vars): global_states[i].extend(window_states[i])

            global_kinematics[0].extend(window_states[0])
            global_kinematics[1].extend(vel_v0)
            global_kinematics[2].extend(acc_v0)

            accumulated_time += dt
            curr_state = window_states[:, -1]

        # Baseline True Solver
        if true_ode_func is not None:
            t_true = np.linspace(0, T_max, 1000)
            true_sol = solve_ivp(true_ode_func, [0, T_max], init_state, t_eval=t_true, method='RK45')

        # Rendering
        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
        fig.suptitle(f'Universal PI-MLS Dashboard - {SolverClass.__name__}', fontsize=14, fontweight='bold')
        colors = ['dodgerblue', 'crimson', 'forestgreen', 'darkorchid']

        # Panel 1: Multi-variable Time Series
        for i in range(num_vars):
            axes[0].plot(global_t, global_states[i], color=colors[i % len(colors)], lw=2.5,
                         label=f'{state_names[i]} (MLS)')
            if true_ode_func is not None:
                axes[0].plot(t_true, true_sol.y[i], '--', color='black', alpha=0.5, label='True' if i == 0 else "")

        axes[0].set_title("Temporal State Trajectories", fontweight='bold')
        axes[0].set_xlabel("Time (t)")
        axes[0].grid(True, ls=':', alpha=0.5);
        axes[0].legend(loc='upper right')

        # Panel 2: Adaptive Phase Portrait Space
        if num_vars >= 2:
            axes[1].plot(global_states[0], global_states[1], color='indigo', lw=2.5, label='MLS Orbit')
            if true_ode_func: axes[1].plot(true_sol.y[0], true_sol.y[1], '--k', alpha=0.5)
            axes[1].set_xlabel(state_names[0]);
            axes[1].set_ylabel(state_names[1])
            axes[1].plot([init_state[0]], [init_state[1]], 'go', ms=8, label='Start')
        else:
            axes[1].plot(global_states[0], global_kinematics[1], color='indigo', lw=2.5, label='Phase Map')
            axes[1].set_xlabel(state_names[0]);
            axes[1].set_ylabel(f"Velocity d({state_names[0]})/dt")
            axes[1].plot([init_state[0]], [global_kinematics[1][0]], 'go', ms=8, label='Start')

        axes[1].set_title("Phase Space Analysis", fontweight='bold')
        axes[1].grid(True, ls=':', alpha=0.5);
        axes[1].legend(loc='upper right')

        # Panel 3: EXACT Kinematics (No longer Taylor Coefficients)
        axes[2].plot(global_t, global_kinematics[0], color='dodgerblue', lw=2, label=r'State $y(t)$')
        axes[2].plot(global_t, global_kinematics[1], color='darkorange', lw=2, label=r'Velocity $y^\prime(t)$')
        axes[2].plot(global_t, global_kinematics[2], color='crimson', lw=2, label=r'Accel $y^{\prime\prime}(t)$')
        axes[2].set_title(f"True Kinematics: {state_names[0]}", fontweight='bold')
        axes[2].set_xlabel("Time (t)")
        axes[2].grid(True, ls=':', alpha=0.5);
        axes[2].legend(loc='upper right')

        plt.tight_layout()
        plt.show()


# =====================================================================
# 4. IMPLEMENTATION
# =====================================================================
if __name__ == "__main__":
    # -------------------------------------------------------------
    # Example A: Binding the Lotka-Volterra System (2D)
    # -------------------------------------------------------------
    def interactive_lv_wrapper(alpha, beta, delta, gamma, x0, y0, T_max):
        def lv_true(t, state):
            return [alpha * state[0] - beta * state[0] * state[1], delta * state[0] * state[1] - gamma * state[1]]

        PIMLS_Dashboard.run(
            SolverClass=LotkaVolterra_Window,
            init_state=[x0, y0],
            T_max=T_max,
            state_names=['Prey', 'Predator'],
            true_ode_func=lv_true,
            alpha=alpha, beta=beta, delta=delta, gamma=gamma
        )


    print("--- Loading Lotka-Volterra Universal Dashboard ---")
    interact(
        interactive_lv_wrapper,
        alpha=FloatSlider(min=0.5, max=2.0, step=0.1, value=1.0),
        beta=FloatSlider(min=0.05, max=0.5, step=0.05, value=0.2),
        delta=FloatSlider(min=0.02, max=0.3, step=0.02, value=0.1),
        gamma=FloatSlider(min=0.5, max=2.0, step=0.1, value=1.0),
        x0=IntSlider(min=5, max=40, step=1, value=20),
        y0=IntSlider(min=1, max=15, step=1, value=2),
        T_max=FloatSlider(min=5.0, max=100.0, step=2.5, value=20.0)
    )


    # -------------------------------------------------------------
    # Example B: Binding the Logistic Growth System (1D)
    # -------------------------------------------------------------
    def interactive_logistic_wrapper(r, K, y0, T_max):
        def log_true(t, y):
            return [r * y[0] * (1 - y[0] / K)]

        PIMLS_Dashboard.run(
            SolverClass=Logistic_Window,
            init_state=[y0],
            T_max=T_max,
            state_names=['Population'],
            true_ode_func=log_true,
            r=r, K=K
        )


    print("\n--- Loading Logistic Growth Universal Dashboard ---")
    interact(
        interactive_logistic_wrapper,
        r=FloatSlider(min=0.5, max=3.0, step=0.1, value=1.2, description='Growth (r)'),
        K=FloatSlider(min=50.0, max=200.0, step=10.0, value=100.0, description='Capacity (K)'),
        y0=FloatSlider(min=1.0, max=50.0, step=1.0, value=5.0, description='Pop y(0)'),
        T_max=FloatSlider(min=2.0, max=20.0, step=1.0, value=8.0, description='Max Span T')
    )