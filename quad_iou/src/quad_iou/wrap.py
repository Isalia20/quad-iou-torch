import torch
from torch import Tensor
from quad_iou_kernel import calculateIoU

def calculate_iou(quad_0: Tensor, quad_1: Tensor, sort_input_quads: bool = True) -> Tensor:
    """
    Calculates the Intersection over Union (IoU) between two sets of quadrilaterals.

    Parameters:
    - quad_0 (Tensor): A tensor of shape (M, 4, 2) representing the coordinates of M quadrilaterals. Each quadrilateral
      is defined by 4 vertices, and each vertex has 2 coordinates (x, y).
    - quad_1 (Tensor): A tensor of shape (N, 4, 2) representing the coordinates of N quadrilaterals. Each quadrilateral
      is defined similarly to those in quad_0.
    - sort_input_quads (bool, optional): Whether to sort the vertices of the quadrilaterals before calculating IoU.
      Sorting can be necessary depending on the specific calculation requirements. Default is True.

    Returns:
    - Tensor: An (N, M) tensor where each element [i, j] is the IoU between the i-th quadrilateral from quad_1 and the
      j-th quadrilateral from quad_0.

    This function computes the IoU for each pair of quadrilaterals from the two input tensors and returns a matrix
    of these IoU values.
    """
    return calculateIoU(quad_0, quad_1, sort_input_quads)
