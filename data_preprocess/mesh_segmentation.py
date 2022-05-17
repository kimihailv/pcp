import numpy as np
import scipy
import trimesh as tm
import pymeshlab as pml
from scipy.sparse import csr_matrix
from faiss import Kmeans
from igl import signed_distance

"""
Implementation of method which is described in https://ieeexplore.ieee.org/document/1348360
Some code fragments were inspired by https://github.com/kugelrund/mesh_segmentation
"""


class MeshSegmentation:
    max_faces = 10000

    def __init__(self, nu, delta, n_patches, fast_fit=False):
        """
        :param nu: multiplier for angle distance
        :param delta: importance of geodesic distance
        :param n_patches: the number of patches
        :param fast_fit: if True then decimated mesh will be segmented
                         and generated labels will be transferred on src mesh
        """
        self.nu = nu
        self.delta = delta
        self.n_patches = n_patches
        self.fast_fit = fast_fit
        self.mesh = None
        self.adj_faces = None
        self.shared_edges = None
        self.src_faces_centers = None

    def calc_geodesic_dist(self):
        adj_faces_v = self.mesh.faces[self.adj_faces]  # n x 2 (adjacent faces) x 3 (numbers' of vertices)
        faces_centers = self.mesh.vertices[adj_faces_v].mean(axis=2) # n x 2 (adjacent faces) x 3 (coord of centers)
        edges_centers = self.mesh.vertices[self.shared_edges].mean(axis=1)  # n x 3 (coords of centers)
        diff = (faces_centers - edges_centers[:, np.newaxis, :])
        dists = np.linalg.norm(diff, axis=2).sum(axis=1)
        return dists / dists.mean()

    def calc_angle_dist(self):
        cos_angles = np.cos(self.mesh.face_adjacency_angles)
        dists = self.nu * (1 - cos_angles)

        return dists / dists.mean()

    def calc_adj_matrix(self):
        geodesic_dist = self.calc_geodesic_dist()
        angle_dist = self.calc_angle_dist()

        dists = self.delta * geodesic_dist + (1 - self.delta) * angle_dist
        faces_num = self.mesh.faces.shape[0]

        adj_matrix = csr_matrix((dists, (self.adj_faces[:, 0], self.adj_faces[:, 1])),
                                shape=(faces_num, faces_num))

        adj_matrix += adj_matrix.T
        return adj_matrix

    def calc_laplacian(self):
        adj_matrix = self.calc_adj_matrix()
        w = scipy.sparse.csgraph.dijkstra(adj_matrix)
        non_reachable = np.where(np.isinf(w))
        w[non_reachable] = 0
        sigma = w.mean()
        w = np.exp(-w / (2 * sigma ** 2))
        w[non_reachable] = 0

        d = 1 / w.sum(axis=1) ** 0.5

        return (w * d).T * d

    def calc_eighvectors(self):
        laplacian = self.calc_laplacian()
        _, v = scipy.linalg.eigh(laplacian,
                                 subset_by_index=(laplacian.shape[0] - self.n_patches,
                                                  laplacian.shape[0] - 1))

        norms = np.linalg.norm(v, axis=1)
        zero_v = np.where(norms == 0)
        norms[zero_v] = 1
        return np.ascontiguousarray(v / norms[:, np.newaxis])

    def fit(self, vertices, faces):
        if self.fast_fit:
            self.src_faces_centers = vertices[faces].mean(axis=1)
            vertices, faces = self._decimate(vertices, faces)

        self.mesh = tm.base.Trimesh(vertices, faces)
        self.adj_faces, self.shared_edges = tm.graph.face_adjacency(mesh=self.mesh, return_edges=True)

    def predict(self):
        v = self.calc_eighvectors().astype(np.float32)
        q = v.dot(v.T)
        centroids_idxs = list(np.unravel_index(q.argmin(), q.shape))

        for _ in range(2, self.n_patches):
            new_idx = np.argmin(np.max(q[centroids_idxs, :], axis=0))
            centroids_idxs.append(new_idx)

        clt = Kmeans(self.n_patches, self.n_patches, niter=10)
        clt.train(v, init_centroids=v[centroids_idxs, :])
        labels = clt.index.search(v, 1)[1].flatten()

        if self.fast_fit:
            closest_faces = signed_distance(self.src_faces_centers, self.mesh.vertices,
                                            self.mesh.faces)[1]

            labels = labels[closest_faces]

        return labels

    def fit_predict(self, vertices, faces):
        self.fit(vertices, faces)
        return self.predict()

    @staticmethod
    def _decimate(vertices, faces):
        mesh_set = pml.MeshSet()
        mesh_set.add_mesh(pml.Mesh(vertices, faces))
        mesh_set.simplification_quadric_edge_collapse_decimation(targetfacenum=MeshSegmentation.max_faces)
        mesh_set.remove_unreferenced_vertices()
        mesh_set.remove_zero_area_faces()
        mesh_set.remove_duplicate_vertices()
        mesh_set.remove_duplicate_faces()
        mesh_set.repair_non_manifold_edges()
        mesh = mesh_set.current_mesh()
        return mesh.vertex_matrix(), mesh.face_matrix()

    @staticmethod
    def gather_patches(labels):
        patches = []

        for idx in range(labels.max() + 1):
            patches.append(np.where(labels == idx)[0])

        return patches
