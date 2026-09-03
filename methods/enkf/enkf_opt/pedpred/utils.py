import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from collections import namedtuple
from torch.utils.data import DataLoader

from .models import PedPred3
from .config import cfg
from .dataset import FileListDataset, GridFromH5Dataset, SeqDataset
from .grid import Grid

import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.pyplot as plt


def plot_generated_matrix_on_ax(example_matrix, ax, state_shape=(4, 36, 12)):
    if example_matrix.ndim == 1:
        if state_shape is None:
            raise ValueError("state_shape must be provided for flattened input")
        example_matrix = example_matrix.reshape(state_shape)
    density = example_matrix[0]
    velocity_x_matrix = example_matrix[1]
    velocity_y_matrix = example_matrix[2]
    velocity = np.sqrt(velocity_x_matrix ** 2 + velocity_y_matrix ** 2)
    heading = np.arctan2(velocity_y_matrix, velocity_x_matrix)

    x, y = np.meshgrid(np.arange(density.shape[1]), np.arange(density.shape[0]))
    u = np.cos(heading)
    v = np.sin(heading)

    mask = np.isnan(density) | (heading == 0)
    u[mask] = np.nan
    v[mask] = np.nan

    im = ax.imshow(density, cmap='Blues', aspect='auto', origin='upper', vmin=0, vmax=1)
    ax.quiver(x, y, u, v, color='black', scale=30, headwidth=3, headlength=4)
    ax.set_xticks([])
    ax.set_yticks([])

    return im  # return image for colorbar


import matplotlib.gridspec as gridspec

def save_step_plot(partial_obs, true_state, estimated_mean, estimated_spread, step_idx, run_dir):
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.15)

    ax0 = fig.add_subplot(gs[0])
    im0 = plot_generated_matrix_on_ax(partial_obs, ax0)
    ax0.set_title(f"Step {step_idx} - Partial Obs")

    ax1 = fig.add_subplot(gs[1])
    im1 = plot_generated_matrix_on_ax(true_state, ax1)
    ax1.set_title("True State")

    ax2 = fig.add_subplot(gs[2])
    im2 = plot_generated_matrix_on_ax(estimated_mean, ax2)
    ax2.set_title("Estimated Mean")

    ax3 = fig.add_subplot(gs[3])
    ax3.set_title("Estimated Spread")
    density_spread = estimated_spread.copy()
    density_spread[np.isnan(density_spread)] = 0
    im3 = ax3.imshow(density_spread[0], cmap="Reds", origin="upper", aspect='auto')
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Add colorbars
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    filename = os.path.join(run_dir, f"step_{step_idx:03d}.png")
    plt.savefig(filename)
    plt.close(fig)


def plot_generated_matrix_on_ax_old(example_matrix, ax, state_shape = (4,36,12)):
    if example_matrix.ndim == 1:
        if state_shape is None:
            raise ValueError("state_shape must be provided for flattened input")
        example_matrix = example_matrix.reshape(state_shape)
    density = example_matrix[0]
    velocity_x_matrix = example_matrix[1]
    velocity_y_matrix = example_matrix[2]
    velocity = np.sqrt(velocity_x_matrix**2 + velocity_y_matrix**2)
    heading = np.arctan2(velocity_y_matrix, velocity_x_matrix)

    x, y = np.meshgrid(np.arange(density.shape[1]), np.arange(density.shape[0]))
    u = np.cos(heading)
    v = np.sin(heading)

    mask = np.isnan(density) | (heading == 0)
    u[mask] = np.nan
    v[mask] = np.nan

    im = ax.imshow(density, cmap='Blues', aspect='auto', origin='upper',vmin=0, vmax=1)
    ax.quiver(x, y, u, v, color='black', scale=30, headwidth=3, headlength=4)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation='vertical')

def visualize_step(partial_obs, true_state, estimated_mean, estimated_spread, step_idx):
    """
    Plot partial observation, true state, and EnKF estimated state with uncertainty.
    partial_obs: (F, H, W) partial observation (NaNs where not observed)
    true_state: (F, H, W)
    estimated_mean: (F, H, W)
    estimated_spread: (F, H, W)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].set_title(f"Step {step_idx} - Partial Obs")
    plot_generated_matrix_on_ax(partial_obs, axes[0])

    axes[1].set_title("True State")
    plot_generated_matrix_on_ax(true_state, axes[1])

    axes[2].set_title("Estimated Mean")
    plot_generated_matrix_on_ax(estimated_mean, axes[2])

    axes[3].set_title("Estimated Spread")
    # For spread, just show density spread
    density_spread = estimated_spread.copy()
    density_spread[np.isnan(density_spread)] = 0  # avoid NaNs
    im = axes[3].imshow(density_spread[0], cmap="Reds", origin="upper", vmin=0, vmax=1)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    plt.colorbar(im, ax=axes[3], orientation='vertical')

    plt.tight_layout()
    plt.show()

def create_run_folder(base_dir="runs"):
    """Create a new folder with timestamp to store plots."""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_step_plot_old(partial_obs, true_state, estimated_mean, estimated_spread, step_idx, run_dir):
    """Plot and save step to a folder."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].set_title(f"Step {step_idx} - Partial Obs")
    plot_generated_matrix_on_ax(partial_obs, axes[0])

    axes[1].set_title("True State")
    plot_generated_matrix_on_ax(true_state, axes[1])

    axes[2].set_title("Estimated Mean")
    plot_generated_matrix_on_ax(estimated_mean, axes[2])

    axes[3].set_title("Estimated Spread")
    density_spread = estimated_spread.copy()
    density_spread[np.isnan(density_spread)] = 0
    im = axes[3].imshow(density_spread[0], cmap="Reds", origin="upper")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    plt.colorbar(im, ax=axes[3], orientation='vertical')

    plt.tight_layout()
    filename = os.path.join(run_dir, f"step_{step_idx:03d}.png")
    plt.savefig(filename)
    plt.close(fig)  # close to free memory


def load_model(model_path, DEVICE):
    model = PedPred3().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def density_mae(pred_density, true_density):
    pred_density = torch.tensor(pred_density, dtype=torch.float32)
    true_density = torch.tensor(true_density, dtype=torch.float32)
    return torch.abs(pred_density - true_density).mean().item()


def velocity_mae(target_density, pred_vel, true_vel):
    # Convert to tensors
    target_density = torch.tensor(target_density, dtype=torch.float32)  # (36,12)
    pred_vel = torch.tensor(pred_vel, dtype=torch.float32)  # (2,36,12)
    true_vel = torch.tensor(true_vel, dtype=torch.float32)  # (2,36,12)
    return torch.norm(target_density * (pred_vel - true_vel), dim=-1).mean()


def directional_consistency(target_density, pred_var, true_var):
    target_density = torch.tensor(target_density, dtype=torch.float32)
    pred_var = torch.tensor(pred_var, dtype=torch.float32)
    true_var = torch.tensor(true_var, dtype=torch.float32)
    return torch.abs(target_density * (pred_var - true_var)).mean().item()



def get_data(*mode, n_in, n_out, local_leak: dict = {}, num_workers = 0, pin_memory =False, drop_last = True, prefetch_factor=2):
    mode = mode or ('train', 'valid')
    dataset, _, subset = cfg.dataset.partition(':')
    resolution = cfg.resolution
    period = cfg.period
    kernel = cfg.kernel
    # nin = cfg.nin
    # nout = cfg.nout
    nin = n_in
    nout = n_out
    batch = 1
    # batch = cfg.batch

    r = 1 / resolution
    r = int(r) if r.is_integer() else None  # don't raise exception until usage

    if dataset == 'alex':
        assert not subset
        grid = Grid((-6, -6), 0, (12 * r, 12 * r), resolution)

        data = {mode:
            DataLoader(
                SeqDataset(
                    GridFromH5Dataset(
                        f'data/alex1_{mode}.h5',
                        grid, period,
                        kernel=kernel,
                        random_rotation=(mode == 'train'),
                    ),
                    nin, nout,
                ),
                batch,
                shuffle=(mode == 'train'),
                generator=torch.default_generator,

            )
            for mode in mode
        }

    elif dataset == 'atc':
        # grids must have shape divisible by 4
        grids = {
            'corridor': Grid(origin=(38.2789, -15.8076), theta=2.5647, shape=(36 * r, 12 * r), resolution=resolution),
            'no_walls': Grid(origin=(-3.0431, 4.3197), theta=2.4128, shape=(16 * r, 12 * r), resolution=resolution),
            'all': Grid(origin=(55.3890, -8.2735), theta=2.6970, shape=(88 * r, 36 * r), resolution=resolution),
        }
        grid = grids[subset]

        data = {mode:
            DataLoader(
                FileListDataset(
                    f'data/sunday_atc_{mode}.lst',
                    lambda file:
                    SeqDataset(
                        GridFromH5Dataset(
                            (file.parent /'ATC' / file.stem).with_suffix('.h5'),
                            grid, period,
                            kernel=kernel,
                            random_rotation=False,
                            attach_point_data=(mode == 'test'),
                            attach_point_time=(mode == 'test'),
                        ),
                        nin, nout,
                    )
                    ,
                ),
                batch,
                shuffle=(mode == 'train'),
                generator=torch.default_generator,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                prefetch_factor=prefetch_factor if num_workers > 0 else None
            )
            for mode in mode
        }

    else:
        raise ValueError(f"Invalid {dataset=}.")

    for key in local_leak:
        local_leak[key] = locals()[key]

    data = namedtuple('DataLoaderSet', data)(**data)
    if len(data) == 1: data = data[0]
    return data

