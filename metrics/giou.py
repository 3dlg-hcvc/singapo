import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
from metrics.iou import (
    _sample_points_in_box3d,
    _apply_backward_transformations,
    _apply_forward_transformations,
    _count_points_in_box3d,
)


def giou_aabb(bbox1_vertices, bbox2_verices):
    """
    Compute the generalized IoU between two axis-aligned bounding boxes\n
    - bbox1_vertices: the vertices of the first bounding box in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n
    - bbox2_vertices: the vertices of the second bounding box in the form: [[x0, y0, z0], [x1, y1, z1], ...]\n

    Return:\n
    - giou: the gIoU between the two bounding boxes
    """
    volume1 = np.prod(np.max(bbox1_vertices, axis=0) - np.min(bbox1_vertices, axis=0))
    volume2 = np.prod(np.max(bbox2_verices, axis=0) - np.min(bbox2_verices, axis=0))

    # Compute the intersection and union of the two bounding boxes
    min_bbox = np.maximum(np.min(bbox1_vertices, axis=0), np.min(bbox2_verices, axis=0))
    max_bbox = np.minimum(np.max(bbox1_vertices, axis=0), np.max(bbox2_verices, axis=0))
    intersection = np.prod(np.clip(max_bbox - min_bbox, a_min=0, a_max=None))
    union = volume1 + volume2 - intersection
    # Compute IoU
    iou = intersection / union if union > 0 else 0

    # Compute the smallest enclosing box
    min_enclosing_bbox = np.minimum(np.min(bbox1_vertices, axis=0), np.min(bbox2_verices, axis=0))
    max_enclosing_bbox = np.maximum(np.max(bbox1_vertices, axis=0), np.max(bbox2_verices, axis=0))
    volume3 = np.prod(max_enclosing_bbox - min_enclosing_bbox)
    
    # Compute gIoU
    giou = iou - (volume3 - union) / volume3 if volume3 > 0 else iou

    return giou


def sampling_giou(
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
        return giou_aabb(bbox1_vertices, bbox2_vertices)

    # Volume of the two bounding boxes
    bbox1_volume = np.prod(
        np.max(bbox1_vertices, axis=0) - np.min(bbox1_vertices, axis=0)
    )
    bbox2_volume = np.prod(
        np.max(bbox2_vertices, axis=0) - np.min(bbox2_vertices, axis=0)
    )
    # Volume of the smallest enclosing box
    min_enclosing_bbox = np.minimum(np.min(bbox1_vertices, axis=0), np.min(bbox2_vertices, axis=0))
    max_enclosing_bbox = np.maximum(np.max(bbox1_vertices, axis=0), np.max(bbox2_vertices, axis=0))
    cbbox_volume = np.prod(max_enclosing_bbox - min_enclosing_bbox)

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

    giou = iou - (cbbox_volume * num_samples - union) / (cbbox_volume * num_samples) if cbbox_volume > 0 else iou

    return giou


def sampling_cDist(
    part1,
    part2,
    bbox1_transformations,
    bbox2_transformations,
):
    '''
    Compute the centroid distance between two bounding boxes\n
    - bbox1_vertices: the vertices of the first bounding box\n
    - bbox2_vertices: the vertices of the second bounding box\n
    - bbox1_transformations: list of transformations applied to the first bounding box\n
    - bbox2_transformations: list of transformations applied to the second bounding box\n
    '''
    
    bbox1_centroid = np.array(part1['aabb']['center'], dtype=np.float32).reshape(1, 3)
    bbox2_centroid = np.array(part2['aabb']['center'], dtype=np.float32).reshape(1, 3)

    # Transform the centroids
    bbox1_transformed_centroids = _apply_forward_transformations(bbox1_centroid, bbox1_transformations)
    bbox2_transformed_centroids = _apply_forward_transformations(bbox2_centroid, bbox2_transformations)

    # Compute the centroid distance
    cDist = np.linalg.norm(bbox1_transformed_centroids - bbox2_transformed_centroids)

    return cDist