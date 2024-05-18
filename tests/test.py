import torch
import quad_iou
import pytest

def create_quadrilateral_tensor(points, dtype):
    """Create a tensor for quadrilateral points, change its data type, and move it to the GPU."""
    return torch.tensor(points, dtype=dtype).reshape(-1, 4, 2).cuda()

DTYPE_IDX_MAP = {
    torch.float16: 0,
    torch.float32: 1,
    torch.float64: 2,
}


test_data = [
    # Test case 1: Identical quadrilaterals
    (
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
        [1.0000, 1.0000, 1.0000]
    ),
    # Test case 2: 0 overlap
    (
        [0, 0, 1, 0, 1, 1, 0, 1],
        [2, 0, 3, 0, 3, 1, 2, 1],
        [0.0000, 0.0000, 0.0000]
    ),
    # Test case 3: Half overlap(from right)
    (
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0.5, 0, 1.5, 0, 1.5, 1, 0.5, 1],
        [0.3333, 0.3333, 0.3333]
    ),
    # Test case 4: Half overlap(from bottom)
    (
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0.5, 1, 0.5, 1, 1.5, 0, 1.5],
        [0.3333, 0.3333, 0.3333]
    ),
    # Test case 5: Half overlap(from left)
    (
        [0, 0, 1, 0, 1, 1, 0, 1],
        [-0.5, 0, 0.5, 0, 0.5, 1, -0.5, 1],
        [0.3333, 0.3333, 0.3333]
    ),
    # Test case 6: Half overlap(from top)
    (
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, -0.5, 1, -0.5, 1, 0.5, 0, 0.5],
        [0.3333, 0.3333, 0.3333]
    ),
    # Test case 7: Half overlap(from right) non clockwise points
    (
        [1, 0, 0, 0, 1, 1, 0, 1],
        [0.5, 0, 1.5, 1, 0.5, 1, 1.5, 0],
        [0.3333, 0.3333, 0.3333]
    ),
    # Test case 8: Half overlap(from bottom)
    (
        [0, 1, 0, 0, 1, 1, 1, 0],
        [0, 1.5, 0, 0.5, 1, 1.5, 1, 0.5],
        [0.3333, 0.3333, 0.3333]
    ),
    # Test case 9: Half overlap(from left)
    (
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0.5, 0, -0.5, 0, 0.5, 1, -0.5, 1],
        [0.3333, 0.3333, 0.3333]
    ),
    # Test case 10: Half overlap(from top)
    (
        [0, 1, 0, 0, 1, 0, 1, 1],
        [0, -0.5, 1, -0.5, 1, 0.5, 0, 0.5],
        [0.3333, 0.3333, 0.3333]
    ),
    # Test case 11: Intersection with a single point but no overlap
    (
        [0, 0, 1, 0, 1, 1, 0, 1],
        [1, 1, 2, 1, 2, 2, 1, 2],
        [0.0000, 0.0000, 0.0000]
    ),
    # Test case 12: No intersection,vertices touching
    (
        [0, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 2, 0, 2, 1, 1, 1],
        [0.0000, 0.0000, 0.0000]
    ),
    # Test case 13: Parallelograms but no intersection
    (
        [0, 0, 1, 0, 1.5, 1, 0.5, 1],
        [1.5, 0, 2.5, 0, 3, 1, 2, 1],
        [0.0000, 0.0000, 0.0000]
    ),
    # Test case 14: Two quads, one quad's vertex being the diagonal of another quad
    (
        [0, 0, 1, 0, 1.5, 1, 0.5, 1],
        [0.5, 1, 1, 0, 2.5, 1, 1.5, 2],
        [0.2000, 0.2000, 0.2000]
    ),
    # Test case 15: Two quadrilaterals with 8 intersection points
    (
        [0, 0, 0.5, 2.5, 2.5, 3, 2, 0],
        [0, 1, 1.3, 3.2, 3, 1.5, 1, -0.4],
        [0.6870, 0.6891, 0.6891]
    ),
    # Test case 16: Two quadrilaterals with 2 intersection points
    (
        [0, 0, 0.5, 2.5, 2.5, 3, 2, 0],
        [0, 1, 1.5, 3.2, -2, 3, -1, 5],
        [0.0268, 0.0267, 0.0267]
    ),
    # Test case 17: Two quadrilaterals with 4 intersection points
    (
        [0, 0, 0.5, 2.5, 2.5, 3, 2, 0],
        [0, 1, 1, 1.7, 1.5, -1, 1, -1],
        [0.2177, 0.2177, 0.2177]
    ),
    # Test case 18: Two quadrilaterals with 5 intersection points
    (
        [0, 0, 0.5, 2.5, 2.5, 3, 2, 0],
        [0, 1, 1, 4, 3, 0, 1, -1],
        [0.5483, 0.5479, 0.5479]
    ),
    # Test case 19: Two quadrilaterals with 3 intersection points
    (
        [0, 0, 2, 0, 2.5, 3, 0.5, 2.5],
        [0.2, 1, 0.8, 0.5, 1.7, 2.4, 1.4, 3.3],
        [0.2769, 0.2769, 0.2769]
    ),
    # Test case 20: Two quadrilaterals with 1 intersection points
    (
        [0, 0, 0.5, 2.5, 2.5, 3, 2, 0],
        [0.2, 1, 1.4, 1.9, 1.7, 1.3, 0.8, 0.5],
        [0.1787, 0.1786, 0.1786]
    ),
    # Test case 21: No intersection inside of the quad
    (
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0.5, 0.5, 0.7, 0.7, 0.8, 0.4, 0.6, 0.1],
        [0.0950, 0.0950, 0.0950]
    ),
    # Test case 22: Two quadrilaterals with 7 intersection points
    (
        [0, 0, 1, 0, 1, 1, 0, 1],
        [-0.2, 0.5, 0.4, 1.1, 1.4, 0.4, 0.5, 0],
        [0.6470, 0.6476, 0.6476]
    ),
    # Test case 23: Two quadrilaterals with 6 intersection points
    (
        [0, 0, 1, 0, 1, 1, 0, 1],
        [-0.2, 0.5, 0.4, 1.1, 1.4, 0.4, 0.5, 0.1],
        [0.5933, 0.5938, 0.5938]
    ),
    # Test case 24: All corners on the vertices
    (
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0.5, 0.5, 1, 1, 0.5, 0.5, 0],
        [0.5000, 0.5000, 0.5000]
    ),
    # Test case 25: 2 corners on the vertex of another quadrilateral
    (
        [0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 0.5, 0.5, 2, 2, 2],
        [0.2115, 0.2115, 0.2115]
    ),
]

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("points1, points2, expected_iou", test_data)
def test_calculate_iou(dtype, points1, points2, expected_iou):
    """Test the Intersection over Union (IoU) calculation for quadrilaterals with different tensor data types."""
    tensor_1 = create_quadrilateral_tensor(points1, dtype)
    tensor_2 = create_quadrilateral_tensor(points2, dtype)
    expected_iou_tensor = torch.tensor([expected_iou[DTYPE_IDX_MAP[dtype]]], dtype=dtype).cuda()

    calculated_iou = quad_iou.calculate_iou(tensor_1, tensor_2, True).to(dtype).round(decimals=4)
    assert torch.allclose(calculated_iou, expected_iou_tensor, atol=1e-6), f"Calculated IoU: {calculated_iou}, Expected IoU: {expected_iou_tensor}"

if __name__ == "__main__":
    pytest.main()