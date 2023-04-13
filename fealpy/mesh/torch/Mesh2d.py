
import numpy as np
import torch
from typing import Union
from torch import Tensor

from .Mesh import Mesh, MeshDataStructure


class Mesh2dDataStructure(MeshDataStructure):
    """
    @brief The topology data structure of mesh 2d.\
           This is an abstract class and can not be used directly.
    """
    def total_edge(self) -> Tensor:
        """
        @brief Return total edges in mesh.

        @return: Tensor with shape (NC*NEC, 2) where NN is number of cells,\
                 NEC is number of edges in each cell.

        @note: There are 2 nodes constructing an edge.
        """
        cell = self.cell
        localEdge = self.localEdge
        totalEdge = cell[:, localEdge].reshape(-1, 2)
        return totalEdge

    def construct(self):
        NC = self.NC
        NEC = self.NEC

        totalEdge = self.total_edge()
        _, i0, j = np.unique(np.sort(totalEdge, axis=-1),
                return_index=True,
                return_inverse=True,
                axis=0)
        NE = i0.shape[0]
        self.NE = NE

        self.edge2cell = torch.zeros((NE, 4), dtype=self.itype, device=self.device)

        i0 = torch.from_numpy(i0)
        i1 = torch.zeros(NE, dtype=self.itype)
        i1[j] = range(NEC*NC)

        self.edge2cell[:, 0] = i0//NEC
        self.edge2cell[:, 1] = i1//NEC
        self.edge2cell[:, 2] = i0%NEC
        self.edge2cell[:, 3] = i1%NEC

        self.edge = totalEdge[i0, :]

    def clear(self):
        self.edge = None
        self.edge2cell = None


    ### Cell ###

    def cell_to_node(self):
        """
        @brief Neighber info from cell to node.

        @return: A tensor with shape (NC, NVC), containing indexes of nodes in\
                 every cells.
        """
        return self.cell

    def cell_to_edge(self):
        """
        @brief Neighber info from cell to edge.

        @return: A tensor with shape (NC, NEC), containing indexes of edges in\
                 every cells.
        """
        NC = self.NC
        NE = self.NE
        NEC = self.NEC
        edge2cell = self.edge2cell

        cell2edge = torch.zeros((NC, NEC), dtype=self.itype, device=self.device)
        cell2edge[edge2cell[:, 0], edge2cell[:, 2]] = torch.arange(NE)
        cell2edge[edge2cell[:, 1], edge2cell[:, 3]] = torch.arange(NE)
        return cell2edge

    def cell_to_edge_sign(self):
        """
        @brief
        """
        NC = self.NC
        NEC = self.NEC

        edge2cell = self.edge2cell

        cell2edgeSign = torch.zeros((NC, NEC), dtype=torch.bool, device=self.device)
        cell2edgeSign[edge2cell[:, 0], edge2cell[:, 2]] = True

        return cell2edgeSign

    cell_to_face = cell_to_edge
    cell_to_face_sign = cell_to_edge_sign

    def cell_to_cell(self):
        pass


    ### Edge ###

    def edge_to_node(self):
        """
        @brief Neighber info from edge to node.

        @return: A tensor with shape (NE, NEC), containing indexes of nodes in\
                 every edges.
        """
        return self.edge

    def edge_to_edge(self):
        pass

    def edge_to_cell(self):
        """
        @brief Neighber info from edge to cell.

        @return: A tensor with shape (NE, 4), providing 4 features for each edge:
        - (0) Index of cell in the left of edge;
        - (1) Index of cell in the right of edge;
        - (2) Local index of the edge in the left cell;
        - (3) Locel index of the edge in the right cell.
        """
        return self.edge2cell


    ### Face ###

    face_to_cell = edge_to_cell


    ### Node ###

    def node_to_node(self):
        pass

    def node_to_edge(self):
        pass

    def node_to_cell(self):
        pass


    ### Boundary ###

    def boundary_node_flag(self):
        """
        @brief Boundary node indicator.

        @return: A bool tensor with shape (NN, ) to indicate if a node is\
                 on the boundary or not.
        """
        NN = self.NN
        edge = self.edge
        isBdEdge = self.boundary_edge_flag()
        isBdNode = torch.zeros((NN,), dtype=torch.bool, device=self.device)
        isBdNode[edge[isBdEdge, :]] = True
        return isBdNode

    def boundary_edge_flag(self):
        """
        @brief Boundary edge indicator.

        @return: A bool tensor with shape (NE, ) to indicate if an edge is\
                 part of boundary or not.
        """
        edge2cell = self.edge2cell
        return edge2cell[:, 0] == edge2cell[:, 1]

    boundary_face_flag = boundary_edge_flag

    def boundary_cell_flag(self):
        """
        @brief Boundary cell indicator.

        @return: A bool tensor with shape (NC, ) to indicate if a cell locats\
                 next to the boundary.
        """
        NC = self.NC
        edge2cell = self.edge2cell
        isBdCell = torch.zeros((NC,), dtype=torch.bool, device=self.device)
        isBdEdge = self.boundary_edge_flag()
        isBdCell[edge2cell[isBdEdge, 0]] = True
        return isBdCell


class Mesh2d(Mesh):
    ds: Mesh2dDataStructure

    def top_dimension(self):
        return 2

    def entity(self, etype: Union[int, str]=2):
        """
        @brief Get entities in mesh.
        """
        if etype in {'cell', 2}:
            return self.ds.cell
        elif etype in {'edge', 'face', 1}:
            return self.ds.edge
        elif etype in {'node', 0}:
            return self.node
        raise ValueError(f"Invalid etype '{etype}'.")

    def entity_barycenter(self, etype: Union[int, str], index=np.s_[:]):
        """
        @brief Get barycenters of entities.
        """
        node = self.entity('node')
        if etype in {'cell', 2}:
            cell = self.ds.cell
            return torch.sum(node[cell[index], :], dim=1) / cell.shape[1]
        elif etype in {'edge', 'face', 1}:
            edge = self.ds.edge
            return torch.sum(node[edge[index], :], dim=1) / edge.shape[1]
        elif etype in {'node', 0}:
            return node[index]
        raise ValueError('the entity `{}` is not correct!'.format(etype))

    def entity_measure(self, etype: Union[int, str]=2, index=np.s_[:]):
        """
        @brief Get measurements for entities.
        """
        if etype in {'cell', 2}:
            return self.cell_area(index=index)
        elif etype in {'edge', 'face', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return 0
        raise ValueError(f"Invalid etity type {etype}.")

    def cell_area(self):
        """
        @brief
        """
        NC = self.number_of_cells()
        node = self.node
        edge = self.ds.edge
        edge2cell = self.ds.edge2cell
        is_inner_edge = ~self.ds.boundary_edge_flag()

        v = (node[edge[:, 1], :] - node[edge[:, 0], :])
        val = torch.sum(v*node[edge[:, 0], :], dim=1)
        a = torch.bincount(edge2cell[:, 0], weights=val, minlength=NC)
        a += torch.bincount(edge2cell[is_inner_edge, 1], weights=-val[is_inner_edge], minlength=NC)
        a /= 2
        return a