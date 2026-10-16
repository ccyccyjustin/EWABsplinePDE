import numpy as np
from scipy.interpolate import splev, splrep, BSpline
import scipy.sparse as sp

def create_knots(int_knots, boundary_knots, degree=3):
    assert len(boundary_knots) == 2
    t = np.concatenate(([boundary_knots[0]] * (degree + 1),
                        int_knots,
                        [boundary_knots[1]] * (degree + 1)))
    assert (np.diff(t) >= 0).all()
    return t

def create_spline_basis(all_knots, degree):
    """
    Creates a continuous BSpline object representing the entire basis space.
    """
    n_basis = len(all_knots) - degree - 1

    # TRICK: By passing the Identity matrix as the coefficients,
    # spl(x) will evaluate ALL individual basis functions at once, returning a matrix.
    c = np.eye(n_basis)

    # extrapolate=False ensures points outside the domain return NaN instead of blowing up
    return BSpline(all_knots, c, degree, extrapolate=False)


def bs(x, spl_obj, intercept=True):
    """
    Evaluates the BSpline object at data points x to create the design matrix.
    """
    # Evaluate the continuous object
    # TODO: handle nan for x?
    basis_matrix = spl_obj(x)
    # basis_matrix2 = BSpline.design_matrix(x, all_knots, degree).toarray()

    # Handle Intercept
    if not intercept:
        basis_matrix = basis_matrix[:, 1:]
    return basis_matrix


def build_custom_tensor_basis(X, spline_objs:list, dim, active_dims):
    """
    Generalization of tensor basis construction for subsets of dimensions.
    """

    # Evaluate 1D bases for the active dimensions
    assert X.shape[1] == dim
    B_list = [bs(X[:, d], spline_objs[d], intercept=True)
              for d in active_dims]

    B_tensor = B_list[0]
    for i in range(1, len(B_list)):
        # Row-wise Kronecker product (Khatri-Rao)
        B_tensor = (B_tensor[:, :, None] * B_list[i][:, None, :]).reshape(X.shape[0], -1)

    return B_tensor


def _build_sparse_1d(x, knots, deg):
    """Evaluates 1D basis strictly returning non-zero values and column indices."""
    # knots = self.all_knots[d]
    k = deg + 1
    sp_mat = BSpline.design_matrix(x, knots, deg)

    # Directly view active indices and weights without memory copy
    data = sp_mat.data.reshape(-1, k)
    indices = sp_mat.indices.reshape(-1, k)
    return data, indices


def build_custom_sparse_tensor_basis(X, n_splines:np.ndarray, all_knots:list, deg, dim, active_dims=None):
    """
    Constructs an (N, M) sparse CSR tensor basis in O(N * 16) time and O(N * 16) memory.
    Never allocates dense arrays.
    """
    if active_dims is None:
        active_dims = list(range(dim))

    N = X.shape[0]
    k = deg + 1  # 4 for cubic

    if len(active_dims) == 1:
        d = active_dims[0]
        data, indices = _build_sparse_1d(X[:, d], all_knots[d], deg)
        indptr = np.arange(0, N * k + 1, k, dtype=np.int32)
        return sp.csr_matrix((data.ravel(), indices.ravel(), indptr),
                             shape=(N, n_splines[d]))

    # 2D Optimized Vectorized Kernel
    if len(active_dims) == 2:
        d0, d1 = active_dims[0], active_dims[1]
        data0, idx0 = _build_sparse_1d(X[:, d0], all_knots[d0], deg)
        data1, idx1 = _build_sparse_1d(X[:, d1], all_knots[d1], deg)
        M1 = n_splines[d1]

        # Outer product of non-zero entries: (N, 4, 1) x (N, 1, 4) -> (N, 16)
        # Flattened directly into 1D memory buffers
        vals = (data0[:, :, None] * data1[:, None, :]).reshape(N, -1).ravel()
        cols = (idx0[:, :, None] * M1 + idx1[:, None, :]).reshape(N, -1).ravel()

        # indptr is strictly uniform
        k_total = k * k  # 16
        indptr = np.arange(0, N * k_total + 1, k_total, dtype=np.int32)

        M_total = n_splines[d0] * n_splines[d1]
        return sp.csr_matrix((vals, cols, indptr), shape=(N, M_total))

    # General N-D fallback
    data_acc, idx_acc = _build_sparse_1d(X[:, active_dims[0]], all_knots[active_dims[0]], deg)
    cols_acc = n_splines[active_dims[0]]
    for d in active_dims[1:]:
        data_d, idx_d = _build_sparse_1d(X[:, d], all_knots[d], deg)
        next_cols = n_splines[d]
        idx_acc = (idx_acc[:, :, None] * next_cols + idx_d[:, None, :]).reshape(N, -1)
        data_acc = (data_acc[:, :, None] * data_d[:, None, :]).reshape(N, -1)
        cols_acc *= next_cols

    k_tot = data_acc.shape[1]
    indptr = np.arange(0, N * k_tot + 1, k_tot, dtype=np.int32)
    return sp.csr_matrix((data_acc.ravel(), idx_acc.ravel(), indptr), shape=(N, cols_acc))


def knots_data_count(X: np.ndarray, all_knots: list, deg: int):
    """
    For each spline interval, calculate how many data points
    :param X:
    :param all_knots:
    :param deg:
    :return:
    """
    knot_counts = []
    for dim, knots_1d in enumerate(all_knots):
        knot_counts_1d = np.zeros(len(knots_1d) - 1 - 2 * deg)
        for i, a_i in enumerate(knots_1d[deg:-(1 + deg)]):
            b_i = knots_1d[deg + i + 1]
            knot_counts_1d[i] = ((X[:, dim] >= a_i) & (X[:, dim] < b_i)).sum()
        knot_counts.append(knot_counts_1d)
    return knot_counts

def project_to_bspline(y, x, s=0, k=3):
    tck = splrep(x, y, s=s, k=k)
    y_fit = splev(x, tck)
    return y_fit


def compute_spline_integral_gl(spl, u, v, sparse=False, weight_func=None):
    """
    Optimized vector-constructed calculation of the 1D spline derivative integral:
        K_{ij} = \int w(x) D^u B_i(x) D^v B_j(x) dx
    Handles non-uniform knot sequences and optional spatial weight functions w(x)
    using vectorized Gauss-Legendre quadrature.
    """
    deg = spl.k
    all_knots = spl.t
    n_splines = spl.c.shape[0]
    internal_knots = all_knots[deg:-deg]

    x_L = internal_knots[:-1]
    x_R = internal_knots[1:]

    # Increase quadrature points when integrating a non-constant weight function w(x)
    n_quad = max(1, deg + 1) if weight_func is None else max(4, 2 * deg + 2)
    quad_x, quad_w = np.polynomial.legendre.leggauss(n_quad)

    widths = x_R - x_L
    midpoints = x_R + x_L

    x_pts = 0.5 * widths[:, None] * quad_x[None, :] + 0.5 * midpoints[:, None]
    w_pts = 0.5 * widths[:, None] * quad_w[None, :]

    x_flat = x_pts.ravel()
    w_flat = w_pts.ravel()

    # Apply 1D spatial coefficient weight function if provided (4.2)
    if weight_func is not None:
        w_flat = w_flat * np.asarray(weight_func(x_flat), dtype=float).ravel()

    vals_u = spl(x_flat, nu=u)
    vals_v = vals_u if u == v else spl(x_flat, nu=v)

    if not sparse:
        if sp.issparse(vals_u):
            vals_u = vals_u.toarray()
            vals_v = vals_u if u == v else vals_v.toarray()
        K = vals_u.T @ (w_flat[:, None] * vals_v)
    else:
        if sp.issparse(vals_u):
            W = sp.diags(w_flat, format='csr')
            K = vals_u.T @ W @ vals_v
            if not sp.issparse(K):
                K = sp.csr_matrix(K)
        else:
            K_dense = vals_u.T @ (w_flat[:, None] * vals_v)

            # Enforce exact local support: B_i and B_j overlap only if |i - j| <= deg
            idx_diff = np.abs(np.subtract.outer(np.arange(n_splines), np.arange(n_splines)))
            K_dense[idx_diff > deg] = 0.0
            K_dense[np.abs(K_dense) < 1e-15] = 0.0
            K = sp.csr_matrix(K_dense)
        K.eliminate_zeros()
    return K