import argparse
import csv
import json
import math
from pathlib import Path
import sys
from datetime import datetime
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fealpy.backend import bm
from fealpy.model.helmholtz.exp0008 import Exp0008


bm.set_backend('pytorch')
torch.set_default_dtype(torch.float64)


class ParametricComplexPINN(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 64, depth: int = 8):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parametric PINN for the axisymmetric expansion chamber Helmholtz example.")
    parser.add_argument('--nx', type=int, default=140, help='Triangle mesh subdivisions in z.')
    parser.add_argument('--ny', type=int, default=40, help='Triangle mesh subdivisions in r.')
    parser.add_argument('--epochs', type=int, default=3000, help='Training epochs.')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden width.')
    parser.add_argument('--depth', type=int, default=4, help='Number of hidden layers.')
    parser.add_argument('--lr', type=float, default=1.0e-3, help='Learning rate.')
    parser.add_argument('--step-size', type=int, default=200, help='StepLR step size.')
    parser.add_argument('--gamma', type=float, default=0.9, help='StepLR gamma.')
    parser.add_argument('--npde', type=int, default=4096, help='Interior sample count per epoch.')
    parser.add_argument('--nbc-inlet', type=int, default=512, help='Inlet boundary sample count per epoch.')
    parser.add_argument('--nbc-outlet', type=int, default=512, help='Outlet boundary sample count per epoch.')
    parser.add_argument('--nbc-wall', type=int, default=2048, help='Wall boundary sample count per epoch.')
    parser.add_argument('--ntl', type=int, default=16, help='TL physics frequency samples per epoch.')
    parser.add_argument('--freq-start', type=float, default=20.0, help='Sweep start frequency in Hz.')
    parser.add_argument('--freq-stop', type=float, default=1000.0, help='Sweep stop frequency in Hz.')
    parser.add_argument('--freq-step', type=float, default=10.0, help='Sweep step in Hz.')
    parser.add_argument('--max-frequencies', type=int, default=0, help='Optional cap on the number of postprocessed frequencies; 0 means no cap.')
    parser.add_argument('--snapshot-count', type=int, default=3, help='Number of field snapshots to save.')
    parser.add_argument('--line-points', type=int, default=401, help='Cross-section points used for TL integration.')
    parser.add_argument('--device', type=str, default='cpu', help='Torch device, e.g. cpu or cuda.')
    parser.add_argument('--loss-pde-weight', type=float, default=2.0, help='PDE loss weight.')
    parser.add_argument('--loss-bc-weight', type=float, default=1.0, help='Boundary-condition loss weight.')
    parser.add_argument('--loss-tl-weight', type=float, default=0, help='TL physics loss weight.')
    parser.add_argument('--output-dir', type=str, default='', help='Optional output directory. Default: example/ml/expansion_chamber_helmholtz_results')
    return parser


def freq_to_k(freq: torch.Tensor, sound_speed: float) -> torch.Tensor:
    return 2.0 * math.pi * freq / sound_speed


def normalize_wave_number(k: torch.Tensor, kmin: float, kmax: float) -> torch.Tensor:
    if abs(kmax - kmin) < 1.0e-14:
        return torch.zeros_like(k)
    return 2.0 * (k - kmin) / (kmax - kmin) - 1.0


def grad_scalar(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """y 关于 x 的梯度, y 是标量函数, x 是输入张量"""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]


def tensor_from_array(arr, *, dtype=torch.float64, device='cpu') -> torch.Tensor:
    return torch.as_tensor(np.asarray(arr), dtype=dtype, device=device)


def build_frequency_candidates(freq_start: float, freq_stop: float, freq_step: float,
                               device: str) -> torch.Tensor:
    """构造训练使用的离散频率集合。

    例如 freq_start=200, freq_stop=220, freq_step=10 时，返回
    [[200], [210], [220]]。训练时每个空间采样点的频率都从这个
    离散集合中随机抽取，而不是从连续区间中随机抽取。
    """
    if freq_step <= 0:
        raise ValueError('freq_step must be positive.')
    if freq_stop < freq_start:
        raise ValueError('freq_stop must be greater than or equal to freq_start.')

    freqs = torch.arange(
        float(freq_start),
        float(freq_stop) + 0.5 * float(freq_step),
        float(freq_step),
        dtype=torch.float64,
        device=device,
    )
    freqs = freqs[freqs <= float(freq_stop) + 1.0e-10]
    if freqs.numel() == 0 or torch.abs(freqs[-1] - float(freq_stop)) > 1.0e-10:
        freqs = torch.cat([freqs, torch.tensor([float(freq_stop)], dtype=torch.float64, device=device)])
    return freqs.reshape(-1, 1)


def sample_discrete_frequencies(freq_candidates: torch.Tensor, count: int, device: str) -> torch.Tensor:
    """从离散频率集合中为每个采样点随机抽取一个频率。"""
    if freq_candidates.ndim != 2 or freq_candidates.shape[1] != 1:
        raise ValueError('freq_candidates must have shape (num_freqs, 1).')
    idx = torch.randint(0, freq_candidates.shape[0], (count,), device=device)
    return freq_candidates[idx]


def sample_points_in_mesh(node: torch.Tensor, cell: torch.Tensor, cell_prob: torch.Tensor,
                          count: int, freq_candidates: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """在三角形网格内部随机采样训练点，并从离散频率集合中随机分配频率。

    返回的是“内部空间点 + 频率”。频率不是连续随机数，而是来自
    freq_start:freq_step:freq_stop 构造出的离散频率集合。
    """
    tri_idx = torch.multinomial(cell_prob, count, replacement=True)
    tri = node[cell[tri_idx]]
    u = torch.rand((count, 1), dtype=torch.float64, device=device)
    v = torch.rand((count, 1), dtype=torch.float64, device=device)
    su = torch.sqrt(u)
    lam0 = 1.0 - su
    lam1 = su * (1.0 - v)
    lam2 = su * v
    pts = lam0 * tri[:, 0, :] + lam1 * tri[:, 1, :] + lam2 * tri[:, 2, :]
    freq = sample_discrete_frequencies(freq_candidates, count, device)
    return pts, freq


def sample_points_on_edges(node: torch.Tensor, edge: torch.Tensor, edge_prob: torch.Tensor,
                           count: int, freq_candidates: torch.Tensor, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """在边界线段上随机采样训练点，并从离散频率集合中随机分配频率。

    入口、出口、壁面边界都用它。频率来自 freq_start:freq_step:freq_stop，
    不是连续均匀随机频率。
    """
    edge_idx = torch.multinomial(edge_prob, count, replacement=True)
    seg = node[edge[edge_idx]]
    t = torch.rand((count, 1), dtype=torch.float64, device=device)
    pts = (1.0 - t) * seg[:, 0, :] + t * seg[:, 1, :]
    freq = sample_discrete_frequencies(freq_candidates, count, device)
    return pts, freq


def boundary_normals(points: torch.Tensor, pde: Exp0008) -> torch.Tensor:
    """根据点所在的边界位置，构造外法向量。"""
    z = points[:, 0:1]
    y = points[:, 1:2]
    ay = torch.abs(y)
    tol = max(pde.atol, 1.0e-10)

    n = torch.zeros_like(points)
    inlet = torch.abs(z - pde.z0) < tol
    outlet = torch.abs(z - pde.z5) < tol
    n[inlet[:, 0], 0] = -1.0
    n[outlet[:, 0], 0] = 1.0

    wall = ~(inlet | outlet)
    sign_y = torch.sign(y)

    inlet_band = z <= pde.z2 + tol
    chamber_band = (z >= pde.z2 - tol) & (z <= pde.z3 + tol)
    outlet_band = z >= pde.z3 - tol

    wall_inlet = wall & inlet_band & (torch.abs(ay - pde.r_in) < tol)
    wall_chamber = wall & chamber_band & (torch.abs(ay - pde.r_e) < tol)
    wall_outlet = wall & outlet_band & (torch.abs(ay - pde.r_out) < tol)
    horiz = wall_inlet | wall_chamber | wall_outlet
    n[horiz[:, 0], 1] = sign_y[horiz[:, 0], 0]

    left_step = wall & (torch.abs(z - pde.z2) < tol) & (ay >= pde.r_in - tol) & (ay <= pde.r_e + tol)
    right_step = wall & (torch.abs(z - pde.z3) < tol) & (ay >= pde.r_out - tol) & (ay <= pde.r_e + tol)
    n[left_step[:, 0], 0] = -1.0
    n[right_step[:, 0], 0] = 1.0
    return n


def complex_prediction(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """把网络的 2 个输出通道组装成复数声压 u = u_real + i u_imag"""
    out = model(x)
    return out[:, 0:1] + 1j * out[:, 1:2]


def axisymmetric_pde_residual(model: nn.Module, xy: torch.Tensor, freq: torch.Tensor,
                              fmin: float, fmax: float, sound_speed: float) -> torch.Tensor:
    """计算轴对称 Helmholtz 方程的 PDE 残差。"""
    k = freq_to_k(freq, sound_speed)
    kmin = 2.0 * math.pi * fmin / sound_speed
    kmax = 2.0 * math.pi * fmax / sound_speed
    x = torch.cat([xy, normalize_wave_number(k, kmin, kmax)], dim=1).requires_grad_(True)
    out = model(x)
    ur = out[:, 0:1]
    ui = out[:, 1:2]

    grad_r = grad_scalar(ur, x)
    grad_i = grad_scalar(ui, x)

    ur_z = grad_r[:, 0:1]
    ur_y = grad_r[:, 1:2]
    ui_z = grad_i[:, 0:1]
    ui_y = grad_i[:, 1:2]

    ur_zz = grad_scalar(ur_z, x)[:, 0:1]
    ur_yy = grad_scalar(ur_y, x)[:, 1:2]
    ui_zz = grad_scalar(ui_z, x)[:, 0:1]
    ui_yy = grad_scalar(ui_y, x)[:, 1:2]

    rho = torch.abs(xy[:, 1:2])
    sign_y = torch.sign(xy[:, 1:2])
    res_r = rho * (ur_zz + ur_yy + (k ** 2) * ur) + sign_y * ur_y
    res_i = rho * (ui_zz + ui_yy + (k ** 2) * ui) + sign_y * ui_y
    return res_r + 1j * res_i


def robin_bc_residual(model: nn.Module, xy: torch.Tensor, freq: torch.Tensor, normals: torch.Tensor,
                      fmin: float, fmax: float, sound_speed: float,
                      g: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """计算入口/出口 Robin 边界残差"""
    k = freq_to_k(freq, sound_speed)
    kmin = 2.0 * math.pi * fmin / sound_speed
    kmax = 2.0 * math.pi * fmax / sound_speed
    x = torch.cat([xy, normalize_wave_number(k, kmin, kmax)], dim=1).requires_grad_(True)
    out = model(x)
    ur = out[:, 0:1]
    ui = out[:, 1:2]

    grad_r = grad_scalar(ur, x)
    grad_i = grad_scalar(ui, x)

    dn_r = grad_r[:, 0:1] * normals[:, 0:1] + grad_r[:, 1:2] * normals[:, 1:2]
    dn_i = grad_i[:, 0:1] * normals[:, 0:1] + grad_i[:, 1:2] * normals[:, 1:2]

    u = ur + 1j * ui
    dn = dn_r + 1j * dn_i
    return dn + alpha * u - g


def neumann_bc_residual(model: nn.Module, xy: torch.Tensor, freq: torch.Tensor, normals: torch.Tensor,
                        fmin: float, fmax: float, sound_speed: float) -> torch.Tensor:
    """计算刚性壁面的 Neumann 边界残差，也就是法向导数应接近 0"""
    k = freq_to_k(freq, sound_speed)
    kmin = 2.0 * math.pi * fmin / sound_speed
    kmax = 2.0 * math.pi * fmax / sound_speed
    x = torch.cat([xy, normalize_wave_number(k, kmin, kmax)], dim=1).requires_grad_(True)
    out = model(x)
    ur = out[:, 0:1]
    ui = out[:, 1:2]

    grad_r = grad_scalar(ur, x)
    grad_i = grad_scalar(ui, x)

    dn_r = grad_r[:, 0:1] * normals[:, 0:1] + grad_r[:, 1:2] * normals[:, 1:2]
    dn_i = grad_i[:, 0:1] * normals[:, 0:1] + grad_i[:, 1:2] * normals[:, 1:2]
    return dn_r + 1j * dn_i


def cross_section_average(model: nn.Module, z_value: float, radius: float, freq_hz: float,
                          fmin: float, fmax: float, line_points: int, device: str,
                          sound_speed: float) -> complex:
    """在某个截面上计算轴对称加权平均声压，用于后处理 TL。这里是不可微版本，主要给结果输出用。"""

    y = torch.linspace(-radius, radius, line_points, dtype=torch.float64, device=device).reshape(-1, 1)
    z = torch.full_like(y, z_value)
    freq = torch.full_like(y, freq_hz)
    k = freq_to_k(freq, sound_speed)
    kmin = 2.0 * math.pi * fmin / sound_speed
    kmax = 2.0 * math.pi * fmax / sound_speed
    x = torch.cat([z, y, normalize_wave_number(k, kmin, kmax)], dim=1)
    with torch.no_grad():
        p = complex_prediction(model, x).squeeze(-1)
    w = torch.abs(y.squeeze(-1))
    num_r = torch.trapz(p.real * w, y.squeeze(-1))
    num_i = torch.trapz(p.imag * w, y.squeeze(-1))
    den = torch.trapz(w, y.squeeze(-1)).clamp_min(1.0e-14)
    return complex((num_r / den).item(), (num_i / den).item())


def cross_section_average_tensor(model: nn.Module, z_value: float, radius: float, freq: torch.Tensor,
                                 fmin: float, fmax: float, line_points: int, device: str,
                                 sound_speed: float) -> torch.Tensor:
    """和上一个功能类似，但保持可微，用于训练时构造 TL 的辅助物理损失。"""
    y = torch.linspace(-radius, radius, line_points, dtype=torch.float64, device=device).reshape(1, -1, 1)
    batch = freq.shape[0]
    y = y.repeat(batch, 1, 1)
    z = torch.full_like(y, z_value)
    freq_grid = freq.reshape(batch, 1, 1).repeat(1, line_points, 1)
    k_grid = freq_to_k(freq_grid, sound_speed)
    kmin = 2.0 * math.pi * fmin / sound_speed
    kmax = 2.0 * math.pi * fmax / sound_speed
    x = torch.cat([z, y, normalize_wave_number(k_grid, kmin, kmax)], dim=-1).reshape(-1, 3)
    p = complex_prediction(model, x).reshape(batch, line_points)
    yy = y.squeeze(-1)
    w = torch.abs(yy)
    num_r = torch.trapz(p.real * w, yy, dim=1)
    num_i = torch.trapz(p.imag * w, yy, dim=1)
    den = torch.trapz(w, yy, dim=1).clamp_min(1.0e-14)
    return (num_r / den) + 1j * (num_i / den)


def theoretical_tl_db(pde: Exp0008, freq_hz: float) -> float:
    """计算经典扩张腔消声器的理论 TL 曲线，用来和 PINN 数值结果做对比。"""
    s1 = math.pi * (pde.r_in ** 2)
    s2 = math.pi * (pde.r_e ** 2)
    area_ratio = s2 / s1
    k = 2.0 * math.pi * freq_hz / pde.sound_speed
    chamber_length = pde.l_e
    val = 1.0 + 0.25 * ((area_ratio - 1.0 / area_ratio) ** 2) * (math.sin(k * chamber_length) ** 2)
    return 10.0 * math.log10(max(val, 1.0e-14))


def tl_physics_loss(model: nn.Module, pde: Exp0008, freq: torch.Tensor,
                    fmin: float, fmax: float, line_points: int, device: str) -> torch.Tensor:
    """把 TL 的理论规律做成训练损失。"""
    area_in = math.pi * (pde.r_in ** 2)
    area_out = math.pi * (pde.r_out ** 2)
    pout = cross_section_average_tensor(
        model, pde.z5, pde.r_out, freq, fmin, fmax, line_points, device, pde.sound_speed
    )
    pout_power = (pout.real ** 2 + pout.imag ** 2).clamp_min(1.0e-14)

    area_ratio = (pde.r_e / pde.r_in) ** 2
    k = 2.0 * math.pi * freq / pde.sound_speed
    theory_factor = 1.0 + 0.25 * ((area_ratio - 1.0 / area_ratio) ** 2) * torch.sin(k * pde.l_e) ** 2
    theory_tau = (1.0 / theory_factor).clamp_min(1.0e-14)

    pred_tau = (area_out * pout_power) / (area_in * abs(complex(pde.p_inc.item())) ** 2)
    pred_tau = pred_tau.clamp_min(1.0e-14)

    return torch.mean((torch.log(pred_tau) - torch.log(theory_tau)) ** 2)


def compute_tl_curve(model: nn.Module, pde: Exp0008, freqs: np.ndarray,
                     args, output_dir: Path) -> list[dict]:
    """扫频计算 TL 曲线，并把结果保存为 CSV 文件和 PNG 图像"""
    area_in = math.pi * (pde.r_in ** 2)
    area_out = math.pi * (pde.r_out ** 2)
    rows = []

    for freq in freqs:
        pout = cross_section_average(
            model, pde.z5, pde.r_out, float(freq),
            args.freq_start, args.freq_stop, args.line_points, args.device, pde.sound_speed
        )
        pin = cross_section_average(
            model, pde.z0, pde.r_in, float(freq),
            args.freq_start, args.freq_stop, args.line_points, args.device, pde.sound_speed
        )

        pout_abs = abs(pout)
        pin_abs = abs(pin)
        pref_abs = abs(pin - complex(pde.p_inc.item()))
        tl_numeric = 10.0 * math.log10(
            (area_in * abs(pde.p_inc.item()) ** 2) / max(area_out * pout_abs ** 2, 1.0e-14)
        )
        tl_theory = theoretical_tl_db(pde, float(freq))
        rows.append({
            'frequency_hz': float(freq),
            'wave_number': float(2.0 * math.pi * freq / pde.sound_speed),
            'tl_db': float(tl_numeric),
            'tl_theory_db': float(tl_theory),
            'tl_abs_error_db': float(abs(tl_numeric - tl_theory)),
            'inlet_avg_abs': float(pin_abs),
            'outlet_avg_abs': float(pout_abs),
            'reflection_abs': float(pref_abs),
        })

    csv_path = output_dir / 'tl_curve.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        [row['frequency_hz'] for row in rows],
        [row['tl_db'] for row in rows],
        lw=1.8,
        label='PINN',
    )
    ax.plot(
        [row['frequency_hz'] for row in rows],
        [row['tl_theory_db'] for row in rows],
        lw=1.6,
        ls='--',
        label='Theory',
    )
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Transmission Loss (dB)')
    ax.set_title('Expansion Chamber Muffler TL')
    ax.grid(True, ls='--', alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'tl_curve.png', dpi=200)
    plt.close(fig)
    return rows


def save_field_snapshots(model: nn.Module, mesh, freqs: np.ndarray, pde: Exp0008,
                         args, output_dir: Path) -> None:
    """选若干个频率，把声压场的：实部\虚部幅值"""
    if len(freqs) == 0:
        return

    cell_bc = tensor_from_array(mesh.entity_barycenter('cell'), device=args.device)
    snapshot_count = max(1, min(args.snapshot_count, len(freqs)))
    snapshot_ids = np.linspace(0, len(freqs) - 1, snapshot_count, dtype=int)

    for idx in np.unique(snapshot_ids):
        freq = float(freqs[idx])
        freq_col = torch.full((cell_bc.shape[0], 1), freq, dtype=torch.float64, device=args.device)
        k_col = freq_to_k(freq_col, pde.sound_speed)
        kmin = 2.0 * math.pi * args.freq_start / pde.sound_speed
        kmax = 2.0 * math.pi * args.freq_stop / pde.sound_speed
        x = torch.cat([cell_bc, normalize_wave_number(k_col, kmin, kmax)], dim=1)
        with torch.no_grad():
            u = complex_prediction(model, x).squeeze(-1)
        real = u.real.detach().cpu().numpy()
        imag = u.imag.detach().cpu().numpy()
        ampl = torch.abs(u).detach().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].set_title(f'Re(p), {freq:.0f} Hz')
        axes[1].set_title(f'Im(p), {freq:.0f} Hz')
        axes[2].set_title(f'|p|, {freq:.0f} Hz')
        mesh.add_plot(axes[0], cellcolor=real, linewidths=0, aspect=1)
        mesh.add_plot(axes[1], cellcolor=imag, linewidths=0, aspect=1)
        mesh.add_plot(axes[2], cellcolor=ampl, linewidths=0, aspect=1)
        for ax in axes:
            ax.set_xlabel('z')
            ax.set_ylabel('r')
        fig.tight_layout()
        fig.savefig(output_dir / f'field_{int(round(freq))}Hz.png', dpi=200)
        plt.close(fig)


def save_training_history(history: list[dict], output_dir: Path) -> None:
    """把训练历史保存成 CSV 文件和 PNG 图像"""
    if not history:
        return

    csv_path = output_dir / 'training_history.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([row['epoch'] for row in history], [row['loss'] for row in history], label='total')
    ax.plot([row['epoch'] for row in history], [row['loss_pde'] for row in history], label='pde')
    ax.plot([row['epoch'] for row in history], [row['loss_bc'] for row in history], label='boundary')
    if 'loss_tl' in history[0]:
        ax.plot([row['epoch'] for row in history], [row['loss_tl'] for row in history], label='tl')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.grid(True, ls='--', alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'training_history.png', dpi=200)
    plt.close(fig)


def main():
    run_started_at = datetime.now()
    run_started_clock = perf_counter()

    args = build_parser().parse_args()
    device = torch.device(args.device)

    pde = Exp0008()
    mesh = pde.init_mesh['triangle'](nx=args.nx, ny=args.ny, **pde.mesh_options)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        result_root = Path(__file__).with_name('expansion_chamber_helmholtz_results')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = result_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    node = tensor_from_array(mesh.entity('node'), device=device)
    cell = tensor_from_array(mesh.entity('cell'), dtype=torch.long, device=device)
    cell_area = tensor_from_array(mesh.entity_measure('cell'), device=device)
    cell_prob = cell_area / cell_area.sum()

    bd_idx = np.asarray(mesh.boundary_face_index())
    bd_bc = np.asarray(mesh.entity_barycenter('face', index=bd_idx))
    inlet_mask = np.asarray(pde.is_inlet_boundary(tensor_from_array(bd_bc, device='cpu'))).astype(bool)
    outlet_mask = np.asarray(pde.is_outlet_boundary(tensor_from_array(bd_bc, device='cpu'))).astype(bool)
    wall_mask = ~(inlet_mask | outlet_mask)

    edge_all = np.asarray(mesh.entity('face', index=bd_idx))
    edge_len = np.asarray(mesh.entity_measure('face', index=bd_idx), dtype=np.float64)

    edge_groups = {
        'inlet': edge_all[inlet_mask],
        'outlet': edge_all[outlet_mask],
        'wall': edge_all[wall_mask],
    }
    edge_prob = {
        name: tensor_from_array(edge_len[mask] / edge_len[mask].sum(), device=device)
        for name, mask in [('inlet', inlet_mask), ('outlet', outlet_mask), ('wall', wall_mask)]}
    edge_groups = {
        name: tensor_from_array(edges, dtype=torch.long, device=device)
        for name, edges in edge_groups.items()}

    freq_candidates = build_frequency_candidates(
        args.freq_start, args.freq_stop, args.freq_step, args.device
    )
    print(
        f"Training frequencies: {freq_candidates.numel()} values, "
        f"from {float(freq_candidates[0]):.6g} Hz to {float(freq_candidates[-1]):.6g} Hz, "
        f"step={args.freq_step:g} Hz"
    )

    # mesh_fig, mesh_ax = plt.subplots(figsize=(10, 3.2))
    # mesh.add_plot(mesh_ax)
    # mesh_ax.set_title('Expansion Chamber Triangle Mesh')
    # mesh_fig.tight_layout()
    # mesh_fig.savefig(output_dir / 'mesh.png', dpi=200)
    # plt.close(mesh_fig)

    model = ParametricComplexPINN(hidden=args.hidden, depth=args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    history = []
    report_stride = max(args.epochs // 100, 1)
    training_started_at = datetime.now()
    training_started_clock = perf_counter()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        pde_xy, pde_freq = sample_points_in_mesh(
            node, cell, cell_prob, args.npde, freq_candidates, args.device)
        inlet_xy, inlet_freq = sample_points_on_edges(
            node, edge_groups['inlet'], edge_prob['inlet'], args.nbc_inlet,
            freq_candidates, args.device)
        outlet_xy, outlet_freq = sample_points_on_edges(
            node, edge_groups['outlet'], edge_prob['outlet'], args.nbc_outlet,
            freq_candidates, args.device)
        wall_xy, wall_freq = sample_points_on_edges(
            node, edge_groups['wall'], edge_prob['wall'], args.nbc_wall,
            freq_candidates, args.device)
        tl_freq = sample_discrete_frequencies(freq_candidates, args.ntl, args.device)

        rpde = axisymmetric_pde_residual(
            model, pde_xy, pde_freq, args.freq_start, args.freq_stop, pde.sound_speed)

        inlet_n = boundary_normals(inlet_xy, pde)
        outlet_n = boundary_normals(outlet_xy, pde)
        wall_n = boundary_normals(wall_xy, pde)
        inlet_k = 2.0 * math.pi * inlet_freq / pde.sound_speed
        outlet_k = 2.0 * math.pi * outlet_freq / pde.sound_speed

        rin = robin_bc_residual(
            model, inlet_xy, inlet_freq, inlet_n,
            args.freq_start, args.freq_stop, pde.sound_speed,
            g=-2j * inlet_k * complex(pde.p_inc.item()),
            alpha=-1j * inlet_k,)
        rout = robin_bc_residual(
            model, outlet_xy, outlet_freq, outlet_n,
            args.freq_start, args.freq_stop, pde.sound_speed,
            g=torch.zeros_like(outlet_k, dtype=torch.complex128),
            alpha=-1j * outlet_k,)
        rwall = neumann_bc_residual(
            model, wall_xy, wall_freq, wall_n, args.freq_start, args.freq_stop, pde.sound_speed)

        loss_pde = torch.mean(rpde.real ** 2 + rpde.imag ** 2)
        loss_bc = (
            torch.mean(rin.real ** 2 + rin.imag ** 2) +
            torch.mean(rout.real ** 2 + rout.imag ** 2) +
            torch.mean(rwall.real ** 2 + rwall.imag ** 2))
        loss_tl = tl_physics_loss(
            model, pde, tl_freq, args.freq_start, args.freq_stop, args.line_points, args.device
        )
        loss = args.loss_pde_weight * loss_pde + args.loss_bc_weight * loss_bc + args.loss_tl_weight * loss_tl
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch == 1 or epoch % report_stride == 0 or epoch == args.epochs:
            row = {
                'epoch': epoch,
                'loss': float(loss.detach().cpu()),
                'loss_pde': float(loss_pde.detach().cpu()),
                'loss_bc': float(loss_bc.detach().cpu()),
                'loss_tl': float(loss_tl.detach().cpu()),
            }
            history.append(row)
            print(
                f"epoch={epoch:5d} loss={row['loss']:.6e} "
                f"pde={row['loss_pde']:.6e} bc={row['loss_bc']:.6e} tl={row['loss_tl']:.6e}"
            )
    training_finished_at = datetime.now()
    training_elapsed_seconds = perf_counter() - training_started_clock

    save_training_history(history, output_dir)

    freqs = freq_candidates.detach().cpu().numpy().reshape(-1).astype(np.float64)
    if args.max_frequencies > 0:
        freqs = freqs[:args.max_frequencies]

    tl_rows = compute_tl_curve(model, pde, freqs, args, output_dir)
    save_field_snapshots(model, mesh, freqs, pde, args, output_dir)
    run_finished_at = datetime.now()
    run_elapsed_seconds = perf_counter() - run_started_clock

    summary = {
        'runtime': {
            'run_started_at': run_started_at.isoformat(timespec='seconds'),
            'run_finished_at': run_finished_at.isoformat(timespec='seconds'),
            'run_elapsed_seconds': round(run_elapsed_seconds, 6),
            'training_started_at': training_started_at.isoformat(timespec='seconds'),
            'training_finished_at': training_finished_at.isoformat(timespec='seconds'),
            'training_elapsed_seconds': round(training_elapsed_seconds, 6),
        },
        'mesh': {'nx': args.nx, 'ny': args.ny, 'cells': int(mesh.number_of_cells())},
        'geometry': {
            'l_in': pde.l_in, 'l_e': pde.l_e, 'l_out': pde.l_out,
            'd_in': pde.d_in, 'd_e': pde.d_e, 'd_out': pde.d_out,
            'l_buf_in': pde.l_buf_in, 'l_buf_out': pde.l_buf_out,
        },
        'training': {
            'epochs': args.epochs, 'npde': args.npde, 'nbc_inlet': args.nbc_inlet,
            'nbc_outlet': args.nbc_outlet, 'nbc_wall': args.nbc_wall,
            'ntl': args.ntl,
            'hidden': args.hidden, 'depth': args.depth,
            'model_input': '[x, y, k]',
            'loss_formula': 'loss = loss_pde + loss_bc_weight * loss_bc + loss_tl_weight * loss_tl',
            'loss_bc_weight': args.loss_bc_weight,
            'loss_tl_weight': args.loss_tl_weight,
        },
        'frequency_sweep': {
            'start_hz': args.freq_start,
            'stop_hz': args.freq_stop,
            'step_hz': args.freq_step,
            'count': int(len(freqs)),
            'training_frequency_count': int(freq_candidates.numel()),
            'training_frequencies_hz': [float(v) for v in freq_candidates.detach().cpu().reshape(-1).tolist()],
        },
        'tl': {
            'min_db': float(min(row['tl_db'] for row in tl_rows)),
            'max_db': float(max(row['tl_db'] for row in tl_rows)),
            'theory_min_db': float(min(row['tl_theory_db'] for row in tl_rows)),
            'theory_max_db': float(max(row['tl_theory_db'] for row in tl_rows)),
            'mean_abs_error_db': float(
                sum(row['tl_abs_error_db'] for row in tl_rows) / max(len(tl_rows), 1)
            ),
            'max_abs_error_db': float(max(row['tl_abs_error_db'] for row in tl_rows)),
        },
    }
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
