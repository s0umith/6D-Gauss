import torch
from typing import Optional

def make_rotation_mat(direction: torch.Tensor, up: torch.Tensor):
    xaxis = torch.cross(up, direction)
    xaxis = torch.divide(xaxis, torch.linalg.norm(xaxis, dim=-1, keepdim=True))

    yaxis = torch.cross(direction, xaxis)
    yaxis = torch.divide(yaxis, torch.linalg.norm(yaxis, dim=-1, keepdim=True))

    rotation_matrix = torch.eye(3, dtype=direction.dtype, device=direction.device)
    # column1
    rotation_matrix[0, 0] = xaxis[0]
    rotation_matrix[1, 0] = yaxis[0]
    rotation_matrix[2, 0] = direction[0]
    # column2
    rotation_matrix[0, 1] = xaxis[1]
    rotation_matrix[1, 1] = yaxis[1]
    rotation_matrix[2, 1] = direction[1]
    # column3
    rotation_matrix[0, 2] = xaxis[2]
    rotation_matrix[1, 2] = yaxis[2]
    rotation_matrix[2, 2] = direction[2]

    return rotation_matrix

def exclude_negatives(camera_optical_center, sample_points, dirs):
    v = camera_optical_center[None] - sample_points
    d = torch.bmm(
        v.view(v.shape[0], 1, v.shape[-1]), dirs.view(dirs.shape[0], dirs.shape[-1], 1)
    )[..., 0, 0]
    return d > 0

def compute_line_intersection(points, directions, weights=None, return_residuals=False):
    # Compute the direction cross-products
    cross_products = torch.cross(directions[:-1], directions[1:], dim=1)

    # Compute the intersection matrix
    A = cross_products
    b = torch.sum(torch.multiply(points[1:], cross_products), dim=1)

    if weights is not None:
        A = torch.multiply(A, weights[1:, None])
        b = torch.multiply(b, weights[1:])

    parallel_vectors = (torch.abs(cross_products) < 1.0e-7).all(dim=-1)

    non_parallel_vectors = torch.logical_not(parallel_vectors)
    A = A[non_parallel_vectors]
    b = b[non_parallel_vectors]

    # Solve the linear system of equations using the pseudo-inverse
    lstsq_results = torch.linalg.lstsq(A, b)

    # Reshape the solution to obtain the intersection points
    intersections = lstsq_results.solution

    if return_residuals:
        return intersections, lstsq_results.residuals

    if torch.count_nonzero(parallel_vectors) != 0 and torch.isnan(intersections).any():
        print("Parallel vectors")
    if torch.isnan(intersections).any():
        print("Wrong intersection")
        print(A.shape[0])
        intersections = torch.ones_like(intersections, requires_grad=True)

    return intersections

def compute_line_intersection_impl2(
    points, directions, weights: Optional[torch.Tensor] = None, return_residuals=False
):
    """Alternative implementation using least squares."""
    projs = (
        torch.eye(directions.shape[-1], dtype=points.dtype, device=points.device)
        - directions[:, :, None] * directions[:, None, :]
    )  # I - n*n.T

    if weights is not None:
        proj_weighted = torch.multiply(projs, weights[:, None, None])
    else:
        proj_weighted = projs

    R = torch.sum(
        proj_weighted,
        dim=0,
    )

    q = projs @ points[:, :, None]
    if weights is not None:
        q_weighted = torch.multiply(q, weights[:, None, None])
    else:
        q_weighted = q
    q = torch.sum(
        q_weighted,
        dim=0,
    )

    # Solve the least squares problem: Rp = q
    if torch.linalg.det(R) < 1.0e-7:
        return torch.tensor(
            [float("nan"), float("nan"), float("nan")], dtype=R.dtype, device=R.device
        )
    lstsq_results = torch.linalg.solve(R, q)

    intersections = lstsq_results[:, 0]

    if return_residuals:
        return intersections, lstsq_results.residuals

    return intersections

def compute_line_intersection_impl3(
    points, directions, weights=None, return_residuals=False
):
    """
    Another implementation using weighted least squares.
    :param points: (N, 3) array of points on the lines
    :param directions: (N, 3) array of unit direction vectors
    :returns: (3,) array of intersection point
    """
    dirs_mat = directions[:, :, None] @ directions[:, None, :]
    points_mat = points[:, :, None]
    I = torch.eye(3, dtype=points.dtype, device=points.device)

    R_matrix = I - dirs_mat
    if weights is not None:
        R_matrix = torch.multiply(R_matrix, weights[:, None, None])
    b_matrix = (I - dirs_mat) @ points_mat
    if weights is not None:
        b_matrix = torch.multiply(b_matrix, weights[:, None, None])

    # Solve the linear system of equations using the pseudo-inverse
    lstsq_results = torch.linalg.lstsq(R_matrix.sum(dim=0), b_matrix.sum(dim=0))

    # Reshape the solution to obtain the intersection points
    intersections = lstsq_results.solution[:, 0]

    if return_residuals:
        return intersections, lstsq_results.residuals

    return intersections

def IRLS(y, X, maxiter, w_init=1, d=0.0001, tolerance=0.001):
    n, p = X.shape
    delta = torch.full((n,), d, dtype=X.dtype, device=X.device).view(1, n)
    w = torch.full((n,), w_init, dtype=X.dtype, device=X.device)
    W = torch.diag(w)
    B = torch.inverse((X.T @ W) @ X) @ ((X.T @ W) @ y)
    for _ in range(maxiter):
        _B = B
        _w = torch.abs(y[:, None] - (X @ B[:, None])).T
        w = 1.0 / torch.maximum(delta, _w)
        W = torch.diag(w[0])
        B = torch.inverse((X.T @ W) @ X) @ ((X.T @ W) @ y)
        tol = torch.sum(torch.abs(B - _B))
        print("Tolerance = %s" % tol)
        if tol < tolerance:
            return B
    return B

def compute_line_intersection_impl4(
    points, directions, weights=None, return_residuals=False
):  # IRLS-based implementation
    # Compute the direction cross-products
    cross_products = torch.cross(directions[:-1], directions[1:], dim=1)

    # Compute the intersection matrix
    A = cross_products
    b = torch.sum(torch.multiply(points[1:], cross_products), dim=1)

    if weights is not None:
        A = torch.multiply(A, weights[1:, None])
        b = torch.multiply(b, weights[1:])

    parallel_vectors = (torch.abs(cross_products) < 1.0e-7).all(dim=-1)

    non_parallel_vectors = torch.logical_not(parallel_vectors)
    A = A[non_parallel_vectors]
    b = b[non_parallel_vectors]

    # Solve the linear system of equations using IRLS
    intersection = IRLS(b, A, 100)

    return intersection

# RANSAC Implementation
def ransac_line_intersection(
    points, directions, weights=None, max_iterations=100, error_threshold=1e-3, minimal_sample_size=3
):
    """
    Estimate the intersection point using RANSAC to improve robustness to outliers.
    """
    num_points = points.shape[0]
    best_inliers = None
    best_pose = None
    best_inlier_count = 0

    for iteration in range(max_iterations):
        # Randomly sample minimal subset
        if num_points < minimal_sample_size:
            print(f"Iteration {iteration}: Not enough points for minimal sample size.")
            break
        sample_indices = torch.randperm(num_points)[:minimal_sample_size]
        sample_points = points[sample_indices]
        sample_directions = directions[sample_indices]
        sample_weights = weights[sample_indices] if weights is not None else None

        # Estimate pose from minimal subset
        try:
            estimated_pose = compute_line_intersection(sample_points, sample_directions, sample_weights)
        except Exception as e:
            # If estimation fails, skip this iteration
            print(f"Iteration {iteration}: Estimation failed with error {e}")
            continue

        # Compute errors for all correspondences
        vectors_to_estimated = estimated_pose - points
        cross_prod = torch.cross(vectors_to_estimated, directions, dim=1)  # Specify dim
        distances = torch.linalg.norm(cross_prod, dim=1) / torch.linalg.norm(directions, dim=1)

        # Optional: Print distances for debugging
        # print(f"Iteration {iteration}: Distances: {distances}")

        # Determine inliers
        inliers = distances < error_threshold
        inlier_count = inliers.sum().item()

        # Optional: Print inlier count
        # print(f"Iteration {iteration}: Inlier count: {inlier_count}")

        if inlier_count > best_inlier_count:
            best_inliers = inliers
            best_pose = estimated_pose
            best_inlier_count = inlier_count

        # Early exit if a good model is found
        if best_inlier_count > num_points * 0.8:
            print(f"Iteration {iteration}: Early exit with inlier count: {best_inlier_count}")
            break

    if best_inliers is None or best_inlier_count < minimal_sample_size:
        # Fallback to using all data if no valid model is found
        print("RANSAC failed to find a valid model; using all data points.")
        final_pose = compute_line_intersection(points, directions, weights)
    else:
        # Re-estimate pose using all inliers
        inlier_points = points[best_inliers]
        inlier_directions = directions[best_inliers]
        inlier_weights = weights[best_inliers] if weights is not None else None
        final_pose = compute_line_intersection(inlier_points, inlier_directions, inlier_weights)

    return final_pose

# Example usage of RANSAC in pose estimation
def estimate_camera_pose_ransac(points, directions, weights=None):
    # Assume points and directions are tensors of shape (N, 3)
    # weights is an optional tensor of shape (N,)
    camera_optical_center = ransac_line_intersection(points, directions, weights)

    # Compute rotation matrix if necessary
    # For example, using the estimated camera optical center and some up vector
    # up_vector = torch.tensor([0, 1, 0], dtype=points.dtype, device=points.device)
    # direction_vector = some_direction_vector_obtained_elsewhere
    # rotation_matrix = make_rotation_mat(direction_vector, up_vector)

    return camera_optical_center

# The rest of your code remains unchanged

def make_rotation_mat(direction: torch.Tensor, up: torch.Tensor):
    xaxis = torch.cross(up, direction)
    xaxis = torch.divide(xaxis, torch.linalg.norm(xaxis, dim=-1, keepdim=True))

    yaxis = torch.cross(direction, xaxis)
    yaxis = torch.divide(yaxis, torch.linalg.norm(yaxis, dim=-1, keepdim=True))

    rotation_matrix = torch.eye(3)
    # column1
    rotation_matrix[0, 0] = xaxis[0]
    rotation_matrix[1, 0] = yaxis[0]
    rotation_matrix[2, 0] = direction[0]
    # column2
    rotation_matrix[0, 1] = xaxis[1]
    rotation_matrix[1, 1] = yaxis[1]
    rotation_matrix[2, 1] = direction[1]
    # column3
    rotation_matrix[0, 2] = xaxis[2]
    rotation_matrix[1, 2] = yaxis[2]
    rotation_matrix[2, 2] = direction[2]

    return rotation_matrix