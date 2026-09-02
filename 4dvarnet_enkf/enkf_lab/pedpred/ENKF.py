import yaml
import numpy as np
import os
import torch
from .utils import *

dir_path = os.path.dirname(os.path.realpath(__file__))
yaml_path = os.path.join(dir_path, 'Parameters.yaml')
with open(yaml_path, 'r') as file:
    params = yaml.safe_load(file)

class GeneratePartialObs:
    def __init__(self, GRID_SIZE, STATE_SHAPE, agent_pos, sensing_range ):
        self.GRID_SIZE = GRID_SIZE
        self.STATE_SHAPE =STATE_SHAPE
        self.agent_pos = agent_pos
        self.sensing_range = sensing_range

        self.num_features = STATE_SHAPE[0]
        self.total_cells = GRID_SIZE[0] * GRID_SIZE[1]
        self.state_dim = np.prod(STATE_SHAPE)


    def cells_within_range(self):
        rows, cols = self.GRID_SIZE
        cells = []
        r0, c0 = self.agent_pos
        for r in range(rows):
            for c in range(cols):
                if np.sqrt((r - r0) ** 2 + (c - c0) ** 2) <= self.sensing_range:
                    cells.append((r, c))
        return cells

    def get_observation_matrix(self):
        observed_cells = self.cells_within_range()
        m = self.num_features * len(observed_cells)
        C = np.zeros((m, self.state_dim))
        for i, (r, c) in enumerate(observed_cells):
            for f in range(self.num_features):
                row_idx = i * self.num_features + f
                col_idx = f * self.total_cells + r * self.GRID_SIZE[1] + c
                C[row_idx, col_idx] = 1.0
        return C, observed_cells

    def reconstruct_observation(self, obs_vector, C, fill_value=np.nan):
        """Rebuild (features, rows, cols) observation with missing entries filled,
           using the actual observation matrix C."""
        reconstructed = np.full(self.STATE_SHAPE, fill_value, dtype=float)

        # For each row in C, find which state variable it maps to
        obs_indices = C.nonzero()[1]  # column indices
        for row_idx, state_idx in enumerate(obs_indices):
            f = state_idx // self.total_cells
            rc = state_idx % self.total_cells
            r, c = divmod(rc, self.GRID_SIZE[1])
            reconstructed[f, r, c] = obs_vector[row_idx]

        return reconstructed

class MultiAgent:
    def __init__(self, grid_size, state_shape, sensing_range, num_agents=2):
        self.grid_size = grid_size
        self.state_shape = state_shape
        self.sensing_range = sensing_range
        self.num_agents = num_agents

        # Random initial positions and goals
        self.positions = [
            (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
            for _ in range(num_agents)
        ]
        self.goals = [
            (np.random.randint(0, grid_size[0]), np.random.randint(0, grid_size[1]))
            for _ in range(num_agents)
        ]

    def move_agents(self):
        """Move each agent one step toward its goal (Manhattan greedy)."""
        new_positions = []
        for (r, c), (gr, gc) in zip(self.positions, self.goals):
            dr = np.sign(gr - r)
            dc = np.sign(gc - c)
            new_r = min(max(r + dr, 0), self.grid_size[0] - 1)
            new_c = min(max(c + dc, 0), self.grid_size[1] - 1)
            new_positions.append((new_r, new_c))
        self.positions = new_positions

    def get_joint_observation(self):
        """Return combined observation from all agents."""
        all_cells = set()
        C_blocks = []
        for pos in self.positions:
            obs_gen = GeneratePartialObs(self.grid_size, self.state_shape, pos, self.sensing_range)
            C, cells = obs_gen.get_observation_matrix()
            all_cells.update(cells)
            C_blocks.append(C)

        # Merge by stacking (can also deduplicate if needed)
        C_joint = np.vstack(C_blocks)
        return C_joint, sorted(all_cells)

class EnsembleKalmanFilter:
    def __init__(self, grid_size, state_shape, ensemble_size=200,
                 proc_noise_std=(0.03435, 0.1899, 0.05144, 0.003886),
                 obs_noise_std=(0.05726, 0.3165, 0.08573, 0.006477),
                 init_perturb_std=(0.2290, 1.2660, 0.3429, 0.0259),
                 inflation=1.0, seed=0):

        self.grid_size = grid_size
        self.state_shape = state_shape
        self.ensemble_size = ensemble_size
        self.num_features = state_shape[0]
        self.state_dim = np.prod(state_shape)
        self.rng = np.random.RandomState(seed)
        self.inflation = inflation

        # Noise vectors
        self.proc_noise_vec = self._expand_noise(proc_noise_std)
        self.obs_noise_vec = self._expand_noise(obs_noise_std)
        self.init_std_vec = self._expand_noise(init_perturb_std)

        self.X = None

    def get_std(self):
        """Returns per-feature std across ensemble, reshaped to state shape."""
        if self.X is None:
            raise ValueError("Ensemble not initialized.")
        return np.std(self.X, axis=0).reshape(self.state_shape)

    def _expand_noise(self, std_per_feature):
        H, W = self.grid_size
        return np.concatenate([np.full(H * W, s) for s in std_per_feature])

    def initialize(self, init_state):
        """Initialize ensemble with perturbations + clipping"""
        mean_state = init_state.flatten()
        self.X = np.tile(mean_state, (self.ensemble_size, 1)) + \
                 self.rng.normal(0, self.init_std_vec * 0.5, size=(self.ensemble_size, self.state_dim))
        self.X = self._clip_bounds(self.X)

    def _clip_bounds(self, X):
        """Apply physical bounds: density [0,5], vx/vy [-5,5], variance [0,2]"""
        H, W = self.grid_size
        F = self.num_features
        feature_size = H * W
        X = X.copy()
        X[:, :feature_size] = np.clip(X[:, :feature_size], 0, 5)           # density
        X[:, feature_size:2*feature_size] = np.clip(X[:, feature_size:2*feature_size], -5, 5)  # vx
        X[:, 2*feature_size:3*feature_size] = np.clip(X[:, 2*feature_size:3*feature_size], -5, 5)  # vy
        X[:, 3*feature_size:] = np.clip(X[:, 3*feature_size:], 0, 2)       # variance
        return X

    def _f_model(self, x_np_batch, model):
        N = x_np_batch.shape[0]
        x_torch = torch.tensor(
            x_np_batch.reshape(N, 1, *self.state_shape), dtype=torch.float32
        )
        with torch.no_grad():
            y_torch = model(x_torch, horizon=1)
        return y_torch.cpu().numpy().reshape(N, -1)

    def forecast(self, model):
        if model is None:
            X_f = self.X.copy()
        else:
            X_f = self._f_model(self.X, model)

        # Add small process noise
        X_f += self.rng.normal(0, 0.01 * self.proc_noise_vec, size=X_f.shape)
        return self._clip_bounds(X_f)

    def update(self, X_f, C, y_obs, regularization=1e-3):
        Y_f = (C @ X_f.T).T
        y_mean = np.mean(Y_f, axis=0)

        X_mean = np.mean(X_f, axis=0)
        X_ano = X_f - X_mean
        Y_ano = Y_f - y_mean

        obs_idx = C.nonzero()[1]
        R = np.diag(self.obs_noise_vec[obs_idx] ** 2)

        P_xy = (X_ano.T @ Y_ano) / (self.ensemble_size - 1)
        P_yy = (Y_ano.T @ Y_ano) / (self.ensemble_size - 1) + R
        P_yy += regularization * np.eye(P_yy.shape[0])

        K = P_xy @ np.linalg.pinv(P_yy)  # safer

        X_a = X_f.copy()
        for i in range(self.ensemble_size):
            perturbed_y = y_obs + self.rng.normal(0, np.sqrt(np.diag(R)))
            X_a[i] = X_f[i] + K @ (perturbed_y - Y_f[i])

        # Inflation
        X_mean = np.mean(X_a, axis=0)
        X_a = X_mean + self.inflation * (X_a - X_mean)

        # Clip after inflation
        self.X = self._clip_bounds(X_a)
        return self.X

    def step(self, C, y_obs, model=None):
        X_f = self.forecast(model=model)
        X_a = self.update(X_f, C, y_obs)
        return X_a.mean(axis=0).reshape(self.state_shape)

class LocalizedEnsembleKalmanFilter:
    def __init__(self, grid_size, state_shape, ensemble_size=200,
                 proc_noise_std=(0.03435, 0.1899, 0.05144, 0.003886),
                 obs_noise_std=(0.05726, 0.3165, 0.08573, 0.006477),
                 init_perturb_std=(0.2290, 1.2660, 0.3429, 0.0259),
                 inflation=1.02, localization_radius=7, seed=0):

        self.grid_size = grid_size
        self.state_shape = state_shape
        self.ensemble_size = ensemble_size
        self.num_features = state_shape[0]
        self.state_dim = np.prod(state_shape)
        self.rng = np.random.RandomState(seed)
        self.inflation = inflation
        self.localization_radius = localization_radius

        # Noise vectors
        self.proc_noise_vec = self._expand_noise(proc_noise_std)
        self.obs_noise_vec = self._expand_noise(obs_noise_std)
        self.init_std_vec = self._expand_noise(init_perturb_std)

        self.X = None

        #estimatin NN model bias
        self.bias_estimate = np.zeros(int(self.state_dim))  # per-state bias estimate
        self.bias_ema_alpha = 0.05  # smoothing factor for EMA; tune 0.01-0.2
    ###########added for estimating NN model bias
    def _backproject_obs_bias(self, obs_idx, obs_bias):
        """
        obs_idx: array of column indices in state vector that each observed row corresponds to (C.nonzero()[1])
        obs_bias: vector of length m (observation-space bias = y_obs - predicted_obs)
        Returns: state_dim vector with non-zero entries at obs_idx (feature-level)
        """
        b = np.zeros(self.state_dim)
        # Each obs row corresponds to a single state variable (since C rows are selection rows)
        # So assign the obs_bias directly to the corresponding state index.
        for k, state_idx in enumerate(obs_idx):
            b[state_idx] = obs_bias[k]
        return b

    ###########
    def get_std(self):
        if self.X is None:
            raise ValueError("Ensemble not initialized.")
        return np.std(self.X, axis=0).reshape(self.state_shape)

    def _expand_noise(self, std_per_feature):
        H, W = self.grid_size
        return np.concatenate([np.full(H * W, s) for s in std_per_feature])

    def initialize(self, init_state):
        mean_state = init_state.flatten()
        self.X = np.tile(mean_state, (self.ensemble_size, 1)) + \
                 self.rng.normal(0, self.init_std_vec * 0.5, size=(self.ensemble_size, self.state_dim))
        self.X = self._clip_bounds(self.X)

    def _clip_bounds(self, X):
        H, W = self.grid_size
        F = self.num_features
        feature_size = H * W
        X = X.copy()
        X[:, :feature_size] = np.clip(X[:, :feature_size], 0, 5)           # density
        X[:, feature_size:2*feature_size] = np.clip(X[:, feature_size:2*feature_size], -5, 5)  # vx
        X[:, 2*feature_size:3*feature_size] = np.clip(X[:, 2*feature_size:3*feature_size], -5, 5)  # vy
        X[:, 3*feature_size:] = np.clip(X[:, 3*feature_size:], 0, 2)       # variance
        return X

    def _f_model(self, x_np_batch, model):
        N = x_np_batch.shape[0]
        x_torch = torch.tensor(
            x_np_batch.reshape(N, 1, *self.state_shape), dtype=torch.float32
        )
        with torch.no_grad():
            y_torch = model(x_torch, horizon=1)
        return y_torch.cpu().numpy().reshape(N, -1)

    def forecast(self, model):
        if model is None:
            X_f = self.X.copy()
        else:
            X_f = self._f_model(self.X, model)

        # Add small process noise
        X_f += self.rng.normal(0, 0.01 * self.proc_noise_vec, size=X_f.shape)

        ##added for bias correction
        X_f = X_f - self.bias_estimate[None, :]

        return self._clip_bounds(X_f)

    def _localization_matrix(self, obs_cells):
        """
        Create a diagonal localization matrix for observations.
        Only cells within `self.localization_radius` will influence updates.
        """
        H, W = self.grid_size
        num_obs = len(obs_cells)
        loc_matrix = np.zeros((num_obs, self.state_dim))

        for i, (r_obs, c_obs) in enumerate(obs_cells):
            for f in range(self.num_features):
                for r in range(H):
                    for c in range(W):
                        # Distance-based weight
                        dist = np.sqrt((r - r_obs)**2 + (c - c_obs)**2)
                        if dist <= self.localization_radius:
                            col_idx = f * H * W + r * W + c
                            loc_matrix[i, col_idx] = np.exp(-(dist**2)/(2*(self.localization_radius**2)))
        return loc_matrix

    def update(self, X_f, C, y_obs, observed_cells=None, regularization=1e-3):
        """
        X_f: f(ensemble) (N, state_dim)
        C: observation matrix
        y_obs: observation vector
        observed_cells: list of tuples (r,c) of observed positions
        """
        if observed_cells is None:
            # fallback: use C.nonzero()
            observed_cells = [(i // self.grid_size[1], i % self.grid_size[1]) for i in C.nonzero()[1]]

        # Apply localization
        Loc = self._localization_matrix(observed_cells)  # shape (num_obs, state_dim)

        Y_f = (C @ X_f.T).T
        y_mean = np.mean(Y_f, axis=0)
        X_mean = np.mean(X_f, axis=0)
        ##aded to remove bias
        obs_bias = y_mean - y_obs
        ##

        X_ano = X_f - X_mean
        Y_ano = Y_f - y_mean

        obs_idx = C.nonzero()[1]
        ####aded to remove bias
        state_bias = self._backproject_obs_bias(obs_idx, obs_bias)
        mask = state_bias != 0
        alpha = self.bias_ema_alpha
        self.bias_estimate[mask] = (1 - alpha) * self.bias_estimate[mask] + alpha * state_bias[mask]
        if observed_cells is None:
            observed_cells = [(i // self.grid_size[1], i % self.grid_size[1]) for i in C.nonzero()[1]]

        Loc = self._localization_matrix(observed_cells)
        y_mean = np.mean(Y_f, axis=0)
        X_mean = np.mean(X_f, axis=0)
        X_ano = X_f - X_mean
        Y_ano = Y_f - y_mean
        obs_idx = C.nonzero()[1]
        ###

        R = np.diag(self.obs_noise_vec[obs_idx] ** 2)

        P_xy = (X_ano.T @ Y_ano) / (self.ensemble_size - 1)
        P_yy = (Y_ano.T @ Y_ano) / (self.ensemble_size - 1) + R
        P_yy += regularization * np.eye(P_yy.shape[0])

        # Apply localization: multiply K by Loc.T element-wise
        K = (P_xy @ np.linalg.pinv(P_yy)) * Loc.T

        # Update ensemble
        X_a = X_f.copy()
        for i in range(self.ensemble_size):
            perturbed_y = y_obs + self.rng.normal(0, np.sqrt(np.diag(R)))
            X_a[i] = X_f[i] + K @ (perturbed_y - Y_f[i])

        # Inflation
        X_mean = np.mean(X_a, axis=0)
        X_a = X_mean + self.inflation * (X_a - X_mean)

        # Clip after inflation
        self.X = self._clip_bounds(X_a)
        return self.X

    def step(self, C, y_obs, observed_cells=None, model=None):
        X_f = self.forecast(model=model)
        X_a = self.update(X_f, C, y_obs, observed_cells=observed_cells)
        return X_a.mean(axis=0).reshape(self.state_shape)

def evaluate_enkf(localization_radius, data, model, GRID_SIZE, STATE_SHAPE,
                  PROC_STD, OBS_STD, ENSEMBLE_SIZE=50, NUM_AGENTS=3, SENSING_RANGE=2, NUM_STEPS=20):
    """
    Run one experiment with a given localization_radius and return RMSE over time.
    """
    enkf = LocalizedEnsembleKalmanFilter(
        grid_size=GRID_SIZE,
        state_shape=STATE_SHAPE,
        ensemble_size=ENSEMBLE_SIZE,
        proc_noise_std=PROC_STD,
        obs_noise_std=OBS_STD,
        localization_radius=localization_radius
    )

    # Initialize ensemble with prior
    STATE_DIM = np.prod(STATE_SHAPE)
    X_init = np.zeros((ENSEMBLE_SIZE, STATE_DIM))
    for f, std in enumerate(PROC_STD):
        start = f * GRID_SIZE[0] * GRID_SIZE[1]
        end = (f + 1) * GRID_SIZE[0] * GRID_SIZE[1]
        X_init[:, start:end] = np.random.normal(0, std, size=(ENSEMBLE_SIZE, GRID_SIZE[0] * GRID_SIZE[1]))
    enkf.X = X_init

    # Multi-agent observation setup
    agents = MultiAgent(GRID_SIZE, STATE_SHAPE, sensing_range=SENSING_RANGE, num_agents=NUM_AGENTS)

    rmse_per_step = []

    for step_idx, (x_in, _) in enumerate(data):
        if step_idx >= NUM_STEPS:
            break

        # Move agents one step
        agents.move_agents()

        # Build joint observation matrix
        C_joint, observed_cells = agents.get_joint_observation()

        # True state
        true_state = x_in.squeeze(0).squeeze(0).numpy().flatten()

        # Simulate noisy observations
        obs_vector = C_joint @ true_state
        obs_noise_vec = np.concatenate([np.full(GRID_SIZE[0]*GRID_SIZE[1], s) for s in OBS_STD])
        obs_vector += np.random.normal(0, obs_noise_vec[C_joint.nonzero()[1]], size=obs_vector.shape)

        # Forecast + update
        estimated_full = enkf.step(C_joint, obs_vector, model=model)

        # Compute RMSE for this step
        rmse = np.sqrt(np.mean((estimated_full.flatten() - true_state.flatten())**2))
        rmse_per_step.append(rmse)

    return rmse_per_step
import numpy as np
import torch
def estimate_obs_std_from_noisy_sim(data, GRID_SIZE, STATE_SHAPE, agents_factory,
                                    OBS_STD_assumed, num_steps=200, seed=0):
    """
    Estimate obs std per feature by simulating the measurement step with noise exactly
    the same way you do in main() and computing std of (noisy_obs - true_obs).
    - agents_factory: lambda returning MultiAgent(GRID_SIZE, STATE_SHAPE, sensing_range=..., num_agents=...)
    - OBS_STD_assumed: per-feature observer std you currently assume (used to generate noisy obs)
    Returns: obs_std_est per feature (length F)
    """
    rng = np.random.RandomState(seed)
    H, W = GRID_SIZE
    F = STATE_SHAPE[0]
    total_cells = H * W
    resid_lists = {f: [] for f in range(F)}

    steps = 0
    for step_idx, (x_in, _) in enumerate(data):
        if steps >= num_steps:
            break
        true_state = x_in.squeeze(0).squeeze(0).numpy().flatten()
        agents = agents_factory()
        C_joint, observed_cells = agents.get_joint_observation()
        y_true = C_joint @ true_state

        # Build per-cell obs noise vector (same as in main)
        obs_noise_vec = np.concatenate([np.full(total_cells, s) for s in OBS_STD_assumed])

        # Add noise exactly like main()
        obs_idx = C_joint.nonzero()[1]
        y_noisy = y_true + rng.normal(0, obs_noise_vec[obs_idx], size=y_true.shape)

        # Residuals per feature
        for k, state_idx in enumerate(obs_idx):
            f = state_idx // total_cells
            resid_lists[f].append(y_noisy[k] - (true_state[state_idx]))

        steps += 1

    obs_std_est = np.array([np.std(resid_lists[f]) if len(resid_lists[f])>0 else 0.0 for f in range(F)])
    return obs_std_est

def estimate_noise_from_data(data, model, GRID_SIZE, STATE_SHAPE, agents_factory,
                             num_steps=200, device="cpu"):
    """
    Returns:
      obs_std_est: array shape (num_features,) estimated obs std (global per-feature)
      proc_std_est: array shape (num_features,) estimated process std (global per-feature)
    agents_factory: function -> returns a MultiAgent instance (so you can vary sensing)
    """
    H, W = GRID_SIZE
    F = STATE_SHAPE[0]
    total_cells = H * W

    obs_resid_list = {f: [] for f in range(F)}
    model_err_list = {f: [] for f in range(F)}

    steps = 0
    for step_idx, (x_in, x_target) in enumerate(data):
        if steps >= num_steps:
            break

        # x_in: t, x_target: t+1 (depending on your loader)
        true_t = x_in.squeeze(0).squeeze(0).numpy().flatten()
        true_tp1 = x_target.squeeze(0).squeeze(0).numpy().flatten()

        # Build agents and obs for this timestep
        agents = agents_factory()
        C_joint, observed_cells = agents.get_joint_observation()

        # Simulate observation noisy (but we want residual wrt true)
        # here we use ideal observation (no noise) to measure sensor noise if dataset provides real obs.
        obs_vector = C_joint @ true_t

        # For obs residuals: compare obs_vector (simulated observation) with true_t at those indices
        obs_idx = C_joint.nonzero()[1]
        # Each obs row corresponds to a single state index
        for k, state_idx in enumerate(obs_idx):
            f = state_idx // total_cells
            val_obs = obs_vector[k]
            val_true = true_t[state_idx]
            obs_resid_list[f].append(val_obs - val_true)

        # For model/process error: predict from true_t using model
        with torch.no_grad():
            x_torch = torch.tensor(true_t.reshape(1, 1, *STATE_SHAPE), dtype=torch.float32).to(device)
            y_torch = model(x_torch, horizon=1)
            y_np = y_torch.cpu().numpy().reshape(-1)
        # model error per feature block (average over cells)
        for f in range(F):
            start = f * total_cells
            end = (f+1) * total_cells
            err_vec = y_np[start:end] - true_tp1[start:end]
            model_err_list[f].extend(err_vec.flatten().tolist())

        steps += 1

    # Aggregate to per-feature stds
    obs_std_est = np.array([np.std(obs_resid_list[f]) if len(obs_resid_list[f])>0 else 0.0 for f in range(F)])
    proc_std_est = np.array([np.std(model_err_list[f]) if len(model_err_list[f])>0 else 0.0 for f in range(F)])

    return obs_std_est, proc_std_est

def estimate_noise():
    params = {
        "NUM_AGENTS": 3,
        "SENSING_RANGE": 5,
        "ENSEMBLE_SIZE": 400,
        "NUM_STEPS": 20,
        "PROC_STD": [0.05, 0.3, 0.2, 0.01],
        "OBS_STD": [0.057, 0.317, 0.086, 0.0065],
        "GRID_SIZE": [36, 12],
        "STATE_SHAPE": [4, 36, 12],
        "Loc_radius": 5,
        "MODEL_PATH": 'pedpred/models/pretty-filly_train_model_28D_1step_20epoch.pth',
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "comment": "apply bias estimation + trust observation less (increase its var)"
    }

    # Save parameters to YAML
    # param_file = os.path.join(run_dir, "run_parameters.yaml")
    # with open(param_file, 'w') as f:
    #     yaml.dump(params, f)

    GRID_SIZE = tuple(params["GRID_SIZE"])
    STATE_SHAPE = tuple(params["STATE_SHAPE"])
    ENSEMBLE_SIZE = params["ENSEMBLE_SIZE"]
    PROC_STD = np.array(params["PROC_STD"])
    OBS_STD = np.array(params["OBS_STD"])
    SENSING_RANGE = params["SENSING_RANGE"]
    NUM_AGENTS = params["NUM_AGENTS"]
    NUM_STEPS = 30
    Loc_radius = params["Loc_radius"]
    PROC_STD = PROC_STD * 2
    OBS_STD = OBS_STD * 2
    # Load data & model
    data = get_data('test2', n_in=1, n_out=1)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model('pedpred/models/pretty-filly_train_model_28D_1step_20epoch.pth', DEVICE)
    # model = None

    # Initialize ensemble with prior
    STATE_DIM = np.prod(STATE_SHAPE)
    STATE_DIM = np.prod(STATE_SHAPE)

    X_init = np.zeros((ENSEMBLE_SIZE, STATE_DIM))
    for f, std in enumerate(PROC_STD):
        start = f * GRID_SIZE[0] * GRID_SIZE[1]
        end = (f + 1) * GRID_SIZE[0] * GRID_SIZE[1]
        X_init[:, start:end] = np.random.normal(0, std, size=(ENSEMBLE_SIZE, GRID_SIZE[0] * GRID_SIZE[1]))

    enkf = LocalizedEnsembleKalmanFilter(grid_size=GRID_SIZE, state_shape=STATE_SHAPE,
                                         ensemble_size=ENSEMBLE_SIZE, proc_noise_std=PROC_STD,
                                         obs_noise_std=OBS_STD, inflation=1, localization_radius=Loc_radius)
    enkf.X = X_init
    # X_init = np.zeros((ENSEMBLE_SIZE, STATE_DIM))
    # agents_factory = lambda: MultiAgent(GRID_SIZE, STATE_SHAPE, sensing_range=2, num_agents=3)
    # obs_std_est, proc_std_est = estimate_noise_from_data(data, model, GRID_SIZE, STATE_SHAPE, agents_factory,
    #                                                      num_steps=100, device=DEVICE)
    # print("Obs std per feature:", obs_std_est)
    # print("Proc std per feature:", proc_std_est)
    agents_factory = lambda: MultiAgent(GRID_SIZE, STATE_SHAPE, sensing_range=SENSING_RANGE, num_agents=NUM_AGENTS)
    # supply your assumed OBS_STD used to simulate sensors in main (if you used none, decide a plausible starting one)
    # OBS_STD_assumed = np.array([0.057, 0.317, 0.086, 0.0065])  # or change to sensor specs
    # obs_std_est = estimate_obs_std_from_noisy_sim(data, GRID_SIZE, STATE_SHAPE, agents_factory, OBS_STD_assumed,
    #                                               num_steps=100)
    # print("Estimated noisy obs std per feature:", obs_std_est)
    ratios = []
    for step_idx, (x_in, _) in enumerate(data):
        if step_idx >= NUM_STEPS: break
        Xf = enkf.forecast(model)
        agents = agents_factory()
        C, _ = agents.get_joint_observation()
        true_state = x_in.squeeze(0).squeeze(0).numpy().flatten()
        y = C @ true_state
        Yf = (C @ Xf.T).T  # (N, m)
        d = y - Yf.mean(axis=0)
        obs_idx = C.nonzero()[1]
        pred_var = np.var(Yf, axis=0) + enkf.obs_noise_vec[obs_idx] ** 2
        ratios.append(np.mean((d ** 2) / (pred_var + 1e-12)))
    print("innovation ratios (per step):", ratios)
    print("mean ratio:", np.mean(ratios))



def main():
    run_dir = create_run_folder(base_dir="runs")  # create folder for this run

    # Hyperparameters
    # --- Hyperparameters ---
    OBS_STD_use = np.array([0.05690936, 0.31472941, 0.08616199, 0.00644609])
    PROC_STD_est = np.array([0.02829307, 0.31263075, 0.12325809, 0.41680932])

    params = {
        "NUM_AGENTS": 3,
        "SENSING_RANGE": 5,
        "ENSEMBLE_SIZE": 400,
        "NUM_STEPS": 20,
        "PROC_STD": [0.02829307, 0.31263075, 0.12325809, 0.41680932],
        "OBS_STD": [0.05690936, 0.31472941, 0.08616199, 0.00644609],
        "GRID_SIZE": [36, 12],
        "STATE_SHAPE": [4, 36, 12],
        "Loc_radius": 5,
        "MODEL_PATH": 'pedpred/models/pretty-filly_train_model_28D_1step_20epoch.pth',
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "comment": "apply bias estimation + trust observation less (increase its var)"
    }

    # Save parameters to YAML
    param_file = os.path.join(run_dir, "run_parameters.yaml")
    with open(param_file, 'w') as f:
        yaml.dump(params, f)

    GRID_SIZE = tuple(params["GRID_SIZE"])
    STATE_SHAPE = tuple(params["STATE_SHAPE"])
    ENSEMBLE_SIZE = params["ENSEMBLE_SIZE"]
    PROC_STD = np.array(params["PROC_STD"])
    OBS_STD = np.array(params["OBS_STD"])
    SENSING_RANGE = params["SENSING_RANGE"]
    NUM_AGENTS = params["NUM_AGENTS"]
    NUM_STEPS = 30
    Loc_radius = params["Loc_radius"]


    # Load data & model
    data = get_data('test2', n_in=1, n_out=1)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model('pedpred/models/pretty-filly_train_model_28D_1step_20epoch.pth',DEVICE)
    # model = None



    # Initialize ensemble with prior
    STATE_DIM = np.prod(STATE_SHAPE)

    X_init = np.zeros((ENSEMBLE_SIZE, STATE_DIM))
    for f, std in enumerate(PROC_STD):
        start = f * GRID_SIZE[0] * GRID_SIZE[1]
        end = (f + 1) * GRID_SIZE[0] * GRID_SIZE[1]
        X_init[:, start:end] = np.random.normal(0, std, size=(ENSEMBLE_SIZE, GRID_SIZE[0] * GRID_SIZE[1]))

    enkf = LocalizedEnsembleKalmanFilter(grid_size=GRID_SIZE, state_shape=STATE_SHAPE,
                                ensemble_size=ENSEMBLE_SIZE, proc_noise_std=PROC_STD,
                                obs_noise_std=OBS_STD, inflation=1, localization_radius= Loc_radius)
    enkf.X = X_init  # set initial ensemble

    # --- Multi-agent simulation ---
    agents = MultiAgent(GRID_SIZE, STATE_SHAPE, sensing_range=SENSING_RANGE, num_agents=NUM_AGENTS)

    # Run filter for NUM_STEPS
    for step_idx, (x_in, _) in enumerate(data):
        print(step_idx,'***********')
        if step_idx >= NUM_STEPS:
            break

        # Move agents one step toward their goals
        agents.move_agents()

        # Build joint observation matrix
        C_joint, observed_cells = agents.get_joint_observation()

        # Simulate true observation vector with noise
        true_state = x_in.squeeze(0).squeeze(0).numpy().flatten()
        obs_vector = C_joint @ true_state
        obs_vector2 = C_joint @ true_state
        # obs_noise_vec = np.concatenate([np.full(GRID_SIZE, s) for s in OBS_STD])
        obs_noise_vec = np.concatenate([
            np.full(GRID_SIZE[0] * GRID_SIZE[1], s) for s in OBS_STD
        ])
        obs_vector += np.random.normal(0, obs_noise_vec[C_joint.nonzero()[1]], size=obs_vector.shape)

        # Forecast + update
        # Assume we have a MultiAgent or GeneratePartialObs instance for each agent
        obs_gen = GeneratePartialObs(GRID_SIZE, STATE_SHAPE, agent_pos=(0, 0), sensing_range=2)
        partial_obs = obs_gen.reconstruct_observation(obs_vector2, C_joint, fill_value=np.nan)

        estimated_full = enkf.step(C_joint, obs_vector, model = model)

        # -------------------- 🔸 DIAGNOSTIC BLOCK --------------------
        # MEAN_RESET_INTERVAL = 10  # how often to subtract mean drift
        #
        # if (step_idx + 1) % MEAN_RESET_INTERVAL == 0:
        #     # 1. compute current ensemble mean
        #     current_mean = np.mean(enkf.X, axis=0)
        #     # 2. subtract mean to recentre ensemble
        #     enkf.X = enkf.X - current_mean
        #     # 3. clip to keep within valid bounds
        #     enkf.X = enkf._clip_bounds(enkf.X)
        #     print(f"[Diagnostic] Subtracted ensemble mean at step {step_idx + 1}")
        # --------------------------------------------------------------

        estimated_spread = enkf.get_std().reshape(STATE_SHAPE)
        save_step_plot(partial_obs, true_state, estimated_full, estimated_spread, step_idx, run_dir)



if __name__ == "__main__":
    estimate_noise()

