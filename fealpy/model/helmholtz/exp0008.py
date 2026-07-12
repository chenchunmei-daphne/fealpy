from typing import Sequence

from ...backend import TensorLike
from ...backend import backend_manager as bm
from ...decorator import cartesian
from ...mesher import ExpansionChamberMesher


class Exp0008(ExpansionChamberMesher):
    r"""Axisymmetric Helmholtz scattering model for an expansion chamber muffler.

    The PDE is

        p_zz + p_rr + (1 / r) p_r + k^2 p = 0,  in Omega,

    with

        p_z + i k p = 2 i k p_inc,   on the inlet,
        p_z - i k p = 0,             on the outlet,
        \partial_n p = 0,            on rigid walls.

    This is a scattering example without a manufactured exact solution. The
    model therefore focuses on geometry, wave-number data, boundary partitions,
    Robin data, and axisymmetric weights needed by downstream solvers.
    """

    def __init__(self, options: dict = {}):
        self.l_in = float(options.get('l_in', 0.05))
        self.l_e = float(options.get('l_e', 0.2))
        self.l_out = float(options.get('l_out', 0.05))
        self.d_in = float(options.get('d_in', 0.025))
        self.d_e = float(options.get('d_e', 0.1))
        self.d_out = float(options.get('d_out', 0.025))
        self.l_buf_in = float(options.get('l_buf_in', 0.20))
        self.l_buf_out = float(options.get('l_buf_out', 0.20))
        self.sound_speed = float(options.get('sound_speed', 340.0))
        self.frequency = float(options.get('frequency', 20.0))
        self.p_inc = bm.asarray(options.get('p_inc', 1.0 + 0.0j), dtype=bm.complex128)
        self.atol = float(options.get('atol', 1.0e-12))

        self.k = bm.asarray(
            options.get('k', 2.0 * bm.pi * self.frequency / self.sound_speed),
            dtype=bm.float64)

        self.r_in = 0.5 * self.d_in
        self.r_e = 0.5 * self.d_e
        self.r_out = 0.5 * self.d_out

        self.z0 = 0.0
        self.z1 = self.l_buf_in
        self.z2 = self.z1 + self.l_in
        self.z3 = self.z2 + self.l_e
        self.z4 = self.z3 + self.l_out
        self.z5 = self.z4 + self.l_buf_out

        self.box = [self.z0, self.z5, -self.r_e, self.r_e]
        self.mesh_options = {
            'l_in': self.l_in,
            'l_e': self.l_e,
            'l_out': self.l_out,
            'd_in': self.d_in,
            'd_e': self.d_e,
            'd_out': self.d_out,
            'l_buf_in': self.l_buf_in,
            'l_buf_out': self.l_buf_out,
            'z_phys_start': self.z1,}

    def geo_dimension(self) -> int:
        return 2

    def domain(self) -> Sequence[float]:
        return self.box

    def wave_number(self) -> float:
        return float(self.k)

    def total_length(self) -> float:
        return self.z5

    def chamber_start(self) -> float:
        return self.z2

    def chamber_end(self) -> float:
        return self.z3

    def inlet_impedance(self) -> complex:
        return complex(-1j * float(self.k))

    def outlet_impedance(self) -> complex:
        return complex(-1j * float(self.k))

    def frequency_samples(self, start: float = 20.0, stop: float = 3200.0, step: float = 10.0):
        count = int(round((stop - start) / step)) + 1
        return [start + i * step for i in range(count)]

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        raise NotImplementedError("Exp0008 is a scattering problem without a closed-form exact solution.")

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        raise NotImplementedError("Exp0008 does not provide an exact gradient.")

    @cartesian
    def laplacian(self, p: TensorLike) -> TensorLike:
        raise NotImplementedError("Exp0008 does not provide an exact Laplacian.")

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        return bm.zeros_like(p[..., 0], dtype=bm.complex128)

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        raise NotImplementedError("Exp0008 does not use Dirichlet boundary conditions.")

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        return bm.zeros_like(p[..., 0], dtype=bm.bool)

    @cartesian
    def neumann(self, p: TensorLike, n: TensorLike) -> TensorLike:
        return bm.zeros_like(p[..., 0], dtype=bm.complex128)

    @cartesian
    def is_neumann_boundary(self, p: TensorLike) -> TensorLike:
        return self.is_wall_boundary(p)

    @cartesian
    def robin(self, p: TensorLike, n: TensorLike) -> TensorLike:
        val = bm.zeros_like(p[..., 0], dtype=bm.complex128)
        inlet = self.is_inlet_boundary(p)
        val[inlet] = -2j * self.k * self.p_inc
        return val

    @cartesian
    def robin_coefficient(self, p: TensorLike) -> TensorLike:
        val = bm.zeros_like(p[..., 0], dtype=bm.complex128)
        robin = self.is_robin_boundary(p)
        val[robin] = -1j * self.k
        return val

    @cartesian
    def incident_pressure(self, p: TensorLike) -> TensorLike:
        val = bm.zeros_like(p[..., 0], dtype=bm.complex128)
        val[self.is_inlet_boundary(p)] = self.p_inc
        return val

    @cartesian
    def radial_coordinate(self, p: TensorLike) -> TensorLike:
        return bm.abs(p[..., 1])

    @cartesian
    def axisymmetric_weight(self, p: TensorLike) -> TensorLike:
        return self.radial_coordinate(p)

    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        z = p[..., 0]
        return bm.abs(z - self.z0) < self.atol

    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        z = p[..., 0]
        return bm.abs(z - self.z5) < self.atol

    @cartesian
    def is_robin_boundary(self, p: TensorLike) -> TensorLike:
        return self.is_inlet_boundary(p) | self.is_outlet_boundary(p)

    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        return ~self.is_robin_boundary(p)