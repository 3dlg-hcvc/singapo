import numpy as np


def _sample_points_in_box3d(bbox_vertices, num_samples):
    """
    Sample points in a axis-aligned 3D bounding box\n
    - bbox_vertices: the vertices of the bounding box in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n
    - num_samples: the number of samples to use\n

    Return:\n
    - points: the sampled points in the form: [[x0, y0, z0], [x1, y1, z1], ...]
    """

    # Compute the bounding box size
    bbox_size = np.max(bbox_vertices, axis=0) - np.min(bbox_vertices, axis=0)

    # Sample points in the bounding box
    points = np.random.rand(num_samples, 3) * bbox_size + np.min(bbox_vertices, axis=0)

    return points


def _apply_forward_transformations(points, transformations):
    """
    Apply forward transformations to the points\n
    - points: the points in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n
    - transformations: list of transformations to apply\n

    Return:\n
    - points_transformed: the transformed points in the form: [[x0, y0, z0], [x1, y1, z1], ...]
    """
    if len(transformations) == 0:
        return points

    # To homogeneous coordinates
    points_transformed = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

    # Apply the transformations one by one in order
    for transformation in transformations:
        if transformation["type"] == "translation":
            points_transformed = np.matmul(
                transformation["matrix"], points_transformed.T
            ).T

        elif transformation["type"] == "rotation":
            axis_origin = np.append(transformation["rotation_axis_origin"], 0)
            points_recentered = points_transformed - axis_origin

            points_rotated = np.matmul(transformation["matrix"], points_recentered.T).T
            points_transformed = points_rotated + axis_origin

        elif transformation["type"] == "plucker":
            points_transformed = np.matmul(
                transformation["matrix"], points_transformed.T
            ).T

        else:
            raise ValueError(f"Unknown transformation type: {transformation['type']}")

    return points_transformed[..., :3]


def _apply_backward_transformations(points, transformations):
    """
    Apply backward transformations to the points\n
    - points: the points in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n
    - transformations: list of transformations to apply\n
        - The inverse of the transformations are applied in reverse order\n

    Return:\n
    - points_transformed: the transformed points in the form: [[x0, y0, z0], [x1, y1, z1], ...]

    Reference: https://mathematica.stackexchange.com/questions/106257/how-do-i-get-the-inverse-of-a-homogeneous-transformation-matrix
    """
    if len(transformations) == 0:
        return points

    # To homogeneous coordinates
    points_transformed = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

    # Apply the transformations one by one in reverse order
    for transformation in transformations[::-1]:
        inv_transformation = np.eye(4)
        inv_transformation[:3, :3] = transformation["matrix"][:3, :3].T
        inv_transformation[:3, 3] = -np.matmul(
            transformation["matrix"][:3, :3].T, transformation["matrix"][:3, 3]
        )

        if transformation["type"] == "translation":
            points_transformed = np.matmul(inv_transformation, points_transformed.T).T

        elif transformation["type"] == "rotation":
            axis_origin = np.append(transformation["rotation_axis_origin"], 0)
            points_recentered = points_transformed - axis_origin

            points_rotated = np.matmul(inv_transformation, points_recentered.T).T
            points_transformed = points_rotated + axis_origin

        elif transformation["type"] == "plucker":
            points_transformed = np.matmul(inv_transformation, points_transformed.T).T

        else:
            raise ValueError(f"Unknown transformation type: {transformation['type']}")

    return points_transformed[..., :3]


def _count_points_in_box3d(points, bbox_vertices):
    """
    Count the number of points in a 3D bounding box\n
    - points: the points in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n
    - bbox_vertices: the vertices of the bounding box in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n
        - The bbox is assumed to be axis-aligned\n

    Return:\n
    - num_points_in_bbox: the number of points in the bounding box
    """

    # Count the number of points in the bounding box
    num_points_in_bbox = np.sum(
        np.all(points >= np.min(bbox_vertices, axis=0), axis=1)
        & np.all(points <= np.max(bbox_vertices, axis=0), axis=1)
    )

    return num_points_in_bbox


def iou_aabb(bbox1_vertices, bbox2_verices):
    """
    Compute the IoU between two axis-aligned bounding boxes\n
    - bbox1_vertices: the vertices of the first bounding box in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n
    - bbox2_vertices: the vertices of the second bounding box in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n

    Return:\n
    - iou: the IoU between the two bounding boxes
    """

    # Compute the intersection and union of the two bounding boxes
    min_bbox = np.maximum(np.min(bbox1_vertices, axis=0), np.min(bbox2_verices, axis=0))
    max_bbox = np.minimum(np.max(bbox1_vertices, axis=0), np.max(bbox2_verices, axis=0))
    intersection = np.prod(np.clip(max_bbox - min_bbox, a_min=0, a_max=None))
    union = (
        np.prod(np.max(bbox1_vertices, axis=0) - np.min(bbox1_vertices, axis=0))
        + np.prod(np.max(bbox2_verices, axis=0) - np.min(bbox2_verices, axis=0))
        - intersection
    )

    # Compute the IoU
    iou = intersection / union if union > 0 else 0

    return iou


def sampling_iou(
    bbox1_vertices,
    bbox2_vertices,
    bbox1_transformations,
    bbox2_transformations,
    num_samples=10000,
):
    """
    Compute the IoU between two bounding boxes\n
    - bbox1_vertices: the vertices of the first bounding box\n
    - bbox2_vertices: the vertices of the second bounding box\n
    - bbox1_transformations: list of transformations applied to the first bounding box\n
    - bbox2_transformations: list of transformations applied to the second bounding box\n
    - num_samples (optional): the number of samples to use per bounding box\n

    Return:\n
    - iou: the IoU between the two bounding boxes after applying the transformations
    """
    # if no transformations are applied, use the axis-aligned bounding box IoU
    if len(bbox1_transformations) == 0 and len(bbox2_transformations) == 0:
        return iou_aabb(bbox1_vertices, bbox2_vertices)

    # Volume of the two bounding boxes
    bbox1_volume = np.prod(
        np.max(bbox1_vertices, axis=0) - np.min(bbox1_vertices, axis=0)
    )
    bbox2_volume = np.prod(
        np.max(bbox2_vertices, axis=0) - np.min(bbox2_vertices, axis=0)
    )

    # Sample points in the two bounding boxes
    bbox1_points = _sample_points_in_box3d(bbox1_vertices, num_samples)
    bbox2_points = _sample_points_in_box3d(bbox2_vertices, num_samples)

    # Transform the points
    forward_bbox1_points = _apply_forward_transformations(
        bbox1_points, bbox1_transformations
    )
    forward_bbox2_points = _apply_forward_transformations(
        bbox2_points, bbox2_transformations
    )

    # Transform the forward points to the other box's rest pose frame
    forward_bbox1_points_in_rest_bbox2_frame = _apply_backward_transformations(
        forward_bbox1_points, bbox2_transformations
    )
    forward_bbox2_points_in_rest_bbox1_frame = _apply_backward_transformations(
        forward_bbox2_points, bbox1_transformations
    )

    # Count the number of points in the other bounding box
    num_bbox1_points_in_bbox2 = _count_points_in_box3d(
        forward_bbox1_points_in_rest_bbox2_frame, bbox2_vertices
    )
    num_bbox2_points_in_bbox1 = _count_points_in_box3d(
        forward_bbox2_points_in_rest_bbox1_frame, bbox1_vertices
    )

    # Compute the IoU
    intersect = (
        bbox1_volume * num_bbox1_points_in_bbox2
        + bbox2_volume * num_bbox2_points_in_bbox1
    ) / 2
    union = bbox1_volume * num_samples + bbox2_volume * num_samples - intersect
    iou = intersect / union

    return iou
