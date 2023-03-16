def check_tensor_shape(x, /, target_shape=None, target_dim=None):
    if target_dim is not None:
        assert x.dim() == target_dim, f"expected {target_dim}-dimensional data, but instead got {x.dim()}"
    if target_shape is not None:
        assert (
            x.size() == target_shape
        ), f"expected shape {target_shape}, but got {x.size()} instead"

def check_tensor_shapes_match(x, y):
    try:
        check_tensor_shape(x, target_shape=y.size())
    except AssertionError as e:
        raise AssertionError(
            f"expected tensors to have the same shape, but got {x.size()} and {y.size()} instead"
        ) from e