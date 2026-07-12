from typing import Optional
from ..decorator import variantmethod
from ..mesh import TriangleMesh, QuadrangleMesh


class ExpansionChamberMesher:
    """Full 2D expansion chamber muffler domain mesh generator."""

    def geo_dimension(self) -> int:
        return 2

    @variantmethod('triangle')
    def init_mesh(self,
                  l_in: float = 0.05,
                  l_e: float = 0.2,
                  l_out: float = 0.05,
                  d_in: float = 0.025,
                  d_e: float = 0.1,
                  d_out: float = 0.025,
                  l_buf_in: float = 0.0,
                  l_buf_out: float = 0.0,
                  z_phys_start: Optional[float] = None,
                  nx: int = 140,
                  ny: int = 40) -> TriangleMesh:
        return self._create_mesh(
            "triangle", l_in, l_e, l_out, d_in, d_e, d_out,
            l_buf_in, l_buf_out, z_phys_start, nx, ny)

    @init_mesh.register('quadrangle')
    def init_mesh(self,
                  l_in: float = 0.05,
                  l_e: float = 0.2,
                  l_out: float = 0.05,
                  d_in: float = 0.025,
                  d_e: float = 0.1,
                  d_out: float = 0.025,
                  l_buf_in: float = 0.0,
                  l_buf_out: float = 0.0,
                  z_phys_start: Optional[float] = None,
                  nx: int = 140,
                  ny: int = 40) -> QuadrangleMesh:
        return self._create_mesh(
            "quadrangle", l_in, l_e, l_out, d_in, d_e, d_out,
            l_buf_in, l_buf_out, z_phys_start, nx, ny)

    def _create_mesh(self,
                     mesh_type: str,
                     l_in: float,
                     l_e: float,
                     l_out: float,
                     d_in: float,
                     d_e: float,
                     d_out: float,
                     l_buf_in: float,
                     l_buf_out: float,
                     z_phys_start: Optional[float],
                     nx: int,
                     ny: int):

        if z_phys_start is None:
            z_phys_start = l_buf_in

        z0 = 0.0
        z1 = z_phys_start
        z2 = z1 + l_in
        z3 = z2 + l_e
        z4 = z3 + l_out
        z5 = z4 + l_buf_out

        r_in = d_in / 2.0
        r_e = d_e / 2.0
        r_out = d_out / 2.0

        box = [z0, z5, -r_e, r_e]

        eps = 1.0e-12

        def threshold(p):
            z = p[..., 0]
            y = p[..., 1]

            in_inlet_buffer = (
                (z >= z0 - eps) & (z <= z1 + eps) &
                (y >= -r_in - eps) & (y <= r_in + eps))

            in_inlet = (
                (z >= z1 - eps) & (z <= z2 + eps) &
                (y >= -r_in - eps) & (y <= r_in + eps))

            in_chamber = (
                (z >= z2 - eps) & (z <= z3 + eps) &
                (y >= -r_e - eps) & (y <= r_e + eps))

            in_outlet = (
                (z >= z3 - eps) & (z <= z4 + eps) &
                (y >= -r_out - eps) & (y <= r_out + eps))

            in_outlet_buffer = (
                (z >= z4 - eps) & (z <= z5 + eps) &
                (y >= -r_out - eps) & (y <= r_out + eps))

            inside_domain = (
                in_inlet_buffer |
                in_inlet |
                in_chamber |
                in_outlet |
                in_outlet_buffer)

            return ~inside_domain

        if mesh_type == "triangle":
            return TriangleMesh.from_box(box, nx=nx, ny=ny, threshold=threshold)

        if mesh_type == "quadrangle":
            return QuadrangleMesh.from_box(box, nx=nx, ny=ny, threshold=threshold)

        raise ValueError(f"Unsupported mesh_type: {mesh_type}")