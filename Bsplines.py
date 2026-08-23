import numpy as np
from scipy.interpolate import splev, splrep, BSpline

# def create_knots_(x, df=None, knots=None, degree=3, intercept=True, boundary_knots=None):
#     """
#     Python implementation of R's splines::bs()
#     """
#     assert np.ndim(x) == 1, "support 1d now..."
#     x = np.asarray(x)
#
#     # 1. Determine Boundary Knots
#     if boundary_knots is None:
#         boundary_knots = [np.min(x), np.max(x)]
#     else:
#         boundary_knots = np.sort(boundary_knots)
#
#     # 2. Determine Internal Knots
#     if knots is None:
#         if df is None:
#             # Default behavior if neither df nor knots provided
#             int_knots = []
#         else:
#             # Calculate internal knots based on degrees of freedom (quantiles)
#             # R logic: n_internal_knots = df - degree - (1 if intercept else 0)
#             n_int_knots = df - degree - (1 if intercept else 0)
#             if n_int_knots < 0:
#                 raise ValueError(f"df={df} is too small for degree={degree}")
#
#             # Place knots at quantiles within the boundary
#             qs = np.linspace(0, 100, n_int_knots + 2)[1:-1]
#             int_knots = np.percentile(x, qs)
#     else:
#         int_knots = np.sort(knots)
#
#     # 3. Construct Augmented Knot Vector
#     # We repeat the boundary knots (degree + 1) times at both ends
#     all_knots = np.r_[np.repeat(boundary_knots[0], degree + 1), int_knots, np.repeat(boundary_knots[1], degree + 1)]
#     return all_knots

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


def compute_spline_integral_gl(spl, u, v):
    """
    Optimized vector-constructed calculation of the 1D spline derivative integral.
    Handles non-uniform knot sequences exactly using vectorized Gauss-Legendre quadrature.

    Parameters:
    -----------
    all_knots : array-like
        The complete knot vector (including padded boundary knots).
    u, v : int
        Derivative orders.
    deg : int
        Spline polynomial degree.
    x_eval : array-like, optional
        Custom evaluation points or interval boundaries. If None, integrates over
        the entire internal knot domain.
    """
    # 1. Setup full padded knot vector
    deg = spl.k
    all_knots = spl.t
    internal_knots = all_knots[deg:-deg]

    x_L = internal_knots[:-1]
    x_R = internal_knots[1:]

    # 3. Generate baseline Gauss-Legendre quadrature points on [-1, 1]
    n_quad = max(1, deg + 1)
    quad_x, quad_w = np.polynomial.legendre.leggauss(n_quad)

    # 4. Vectorized scaling transformation to local non-uniform intervals
    widths = x_R - x_L
    midpoints = x_R + x_L

    x_pts = 0.5 * widths[:, None] * quad_x[None, :] + 0.5 * midpoints[:, None]
    w_pts = 0.5 * widths[:, None] * quad_w[None, :]

    # Flatten arrays to perform a single unified evaluation pass
    x_flat = x_pts.ravel()
    w_flat = w_pts.ravel()

    vals_u = spl(x_flat, nu=u)
    vals_v = spl(x_flat, nu=v)

    # Clean up any NaN limits produced precisely at the right domain boundary 'b'
    # vals_u = np.nan_to_num(vals_u)
    # vals_v = np.nan_to_num(vals_v)

    # 6. Contract across the flattened quadrature dimensions
    omega = np.einsum('k,ki,kj->ij', w_flat, vals_u, vals_v)
    return omega