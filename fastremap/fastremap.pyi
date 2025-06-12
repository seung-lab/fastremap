from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

def unique(
    labels: ArrayLike,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: Union[int, None] = None,
) -> Union[
    NDArray[Any],
    tuple[NDArray[Any], NDArray[Any]],
    tuple[NDArray[Any], NDArray[Any], NDArray[Any]],
    tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]],
]:
    """Compute the sorted set of unique labels in the input array.

    Args:
        labels: The input array containing labels.
        return_index: If True, also return the index of the first detected
            occurance of each label.
        return_inverse: If True, also return the indices of the unique array
            (for the specified axis, if provided) that can be used to
            reconstruct the input array.
        return_counts: If True, also return the unique label frequency as an
            array.
        axis: If specified and not None, compute the unique values along this
            axis.

    Returns:
        Either an array of the sorted values or a tuple containing the following
        elements depending on the flags set.

        unique:
            The sorted unique values.
            Always provided.

        unique_indices, optional:
            The indices of the first occurrences of the unique values in the
            original array.
            Only provided if return_index is True.

        unique_inverse, optional:
            The indices to reconstruct the original array from the unique array.
            Only provided if return_inverse is True.

        unique_counts, optional:
            The number of times each of the unique values comes up in the
            original array.
            Only provided if return_counts is True.
    """
    ...

def renumber(
    arr: NDArray[Any],
    start: Union[int, float] = 1,
    preserve_zero: bool = True,
    in_place: bool = False,
) -> tuple[NDArray[Any], dict[int, int] | dict[float, float]]:
    """Renumber an array.

    Given an array of integers, renumber all the unique values starting
    from 1. This can allow us to reduce the size of the data width required
    to represent it.

    Args:
        arr: A numpy array.
        start (default: 1): Start renumbering from this value.
        preserve_zero (default: True): Don't renumber zero.
        in_place (default: False): Perform the renumbering in-place to avoid
            an extra copy. This option depends on a fortran or C contiguous
            array. A copy will be made if the array is not contiguous.

    Returns:
        A renumbered array, dict with remapping of oldval => newval.
    """
    ...

def indices(
    arr: NDArray[Any],
    value: Union[int, float],
) -> NDArray[Any]:
    """Search an array for indices where value matches the array value."""
    ...

def remap(
    arr: ArrayLike,
    table: dict[int, int] | dict[float, float],
    preserve_missing_labels: bool = False,
    in_place: bool = False,
) -> NDArray[Any]:
    """Remap an input numpy array in-place according to a dictionary "table".

    Args:
        arr: An N-dimensional numpy array.
        table: A dictionary resembling: { label: new_label_value, ... }.
        preserve_missing_labels: If an array value is not present in "table"...
            True: Leave it alone.
            False: Throw a KeyError.
        in_place: if True, modify the input array to reduce memory consumption.

    Returns:
        The remapped array.
    """
    ...

def refit(
    arr: NDArray[Any],
    value: Union[int, float, None] = None,
    increase_only: bool = False,
    exotics: bool = False,
) -> NDArray[Any]:
    """Resize the array to the smallest dtype of the same kind that will fit.

    For example, if the input array is uint8 and the value is 2^20 return the
    array as a uint32.

    Works for standard floating, integer, unsigned integer, and complex types.

    Args:
        arr: A numpy array.
        value: Value to fit array to. If None, it is set to the value of the
            absolutely larger of the min and max value in the array.
        increase_only: if true, only resize the array if it can't contain value.
            If false, always resize to the smallest size that fits.
        exotics: If true, allow e.g. half precision floats (16-bit) or double
            complex (128-bit)

    Returns:
        The refitted array.
    """
    ...

def narrow_dtype(
    dtype: DTypeLike,
    exotics: bool = False,
) -> DTypeLike:
    """Widen the given dtype to the next size of the same type.

    For example, int16 -> int8 or uint64 -> uint32.

    8-bit types will map to themselves.

    Args:
        exotics: Whether to include exotics like half precision floats (16-bit)
            or double complex (128-bit).

    Returns:
        The downgraded dtype.
    """
    ...

def widen_dtype(
    dtype: DTypeLike,
    exotics: bool = False,
) -> DTypeLike:
    """Widen the given dtype to the next size of the same type.

    For example, int8 -> int16 or uint32 -> uint64.

    64-bit types will map to themselves.

    Args:
        exotics: Whether to include exotics like half precision floats (16-bit)
            or double complex (128-bit).

    Returns:
        The upgraded dtype.
    """
    ...

def mask(
    arr: ArrayLike,
    labels: ArrayLike,
    in_place: bool = False,
    value: Union[int, float] = 0,
) -> NDArray[Any]:
    """Mask out designated labels in an array with the given value.

    Alternative implementation of:

        arr[np.isin(labels)] = value

    Args:
        arr: An N-dimensional numpy array.
        labels: An iterable list of integers.
        in_place: If True, modify the input array to reduce memory consumption.
        value: A mask value.

    Returns:
        The array with `labels` masked out.
    """
    ...

def mask_except(
    arr: NDArray[Any],
    labels: ArrayLike,
    in_place: bool = False,
    value: Union[int, float] = 0,
) -> NDArray[Any]:
    """Mask out all labels except the provided list.

    Alternative implementation of:

        arr[~np.isin(labels)] = value

    Args:
        arr: An N-dimensional numpy array.
        labels: An iterable list of integers.
        in_place: If True, modify the input array to reduce memory consumption.
        value: A mask value.

    Returns:
        The array with all labels except `labels` masked out.
    """
    ...

def component_map(
    component_labels: ArrayLike,
    parent_labels: ArrayLike,
) -> dict[int, int] | dict[float, float]:
    """Generate a mapping from connected components to their parent labels.

    Given two sets of images that have a surjective mapping between their
    labels, generate a dictionary for that mapping. For example, generate a
    mapping from connected components of labels to their parent labels.

    Returns:
        { $COMPONENT_LABEL: $PARENT_LABEL }

    Examples:
        >>> fastremap.component_map([1, 2, 3, 4], [5, 5, 6, 7])
        {1: 5, 2: 5, 3: 6, 4: 7}
    """
    ...

def inverse_component_map(
    parent_labels: ArrayLike,
    component_labels: ArrayLike,
) -> dict[int, int] | dict[float, float]:
    """Generate a mapping from parent labels to connected components.

    Given two sets of images that have a mapping between their labels, generate
    a dictionary for that mapping. For example, generate a mapping from
    connected components of labels to their parent labels.

    Returns:
        A dictionary resembling: { $PARENT_LABEL: [ $COMPONENT_LABELS, ... ] }.

    Examples:
        >>> fastremap.inverse_component_map([1, 2, 1, 3], [4, 4, 5, 6])
        {1: [4, 5], 2: [4], 3: [6]}
    """
    ...

def remap_from_array(
    arr: NDArray[np.uint],
    vals: NDArray[np.uint],
    in_place: bool = True,
) -> NDArray[Any]:
    """Remap an input numpy array according to the given values array.

    Args:
        arr: An N-dimensional numpy array.
        vals: An array of values to remap to, where the index of the value in
            the array corresponds to the label in the input array.
        in_place: If True, modify the input array to reduce memory consumption.

    Returns:
        The remapped array.
    """
    ...

def remap_from_array_kv(
    arr: NDArray[np.uint],
    keys: NDArray[np.uint],
    vals: NDArray[np.uint],
    preserve_missing_labels: bool = True,
    in_place: bool = True,
) -> NDArray[Any]:
    """Remap an input numpy array according to the keys and values arrays.

    Args:
        arr: An N-dimensional numpy array.
        keys: An array of keys to remap from. Must be the same length as `vals`.
        vals: An array of values to remap to. Must be the same length as `vals`.
        preserve_missing_labels: If an array value is not present in `keys`...
            True: Leave it alone.
            False: Throw a KeyError.
        in_place: If True, modify the input array to reduce memory consumption.

    Returns:
        The remapped array.
    """
    ...

def transpose(arr: NDArray[Any]) -> NDArray[Any]:
    """For up to four dimensional matrices, perform in-place transposition.

    Square matrices up to three dimensions are faster than numpy's out-of-place
    algorithm. Default to the out-of-place implementation numpy uses for cases
    that aren't specially handled.

    Args:
        arr: The input numpy array to transpose.

    Returns:
        The transposed numpy array
    """
    ...

def asfortranarray(arr: NDArray[Any]) -> NDArray[Any]:
    """For up to four dimensional matrices, perform in-place transposition.

    Square matrices up to three dimensions are faster than numpy's out-of-place
    algorithm. Default to the out-of-place implementation numpy uses for cases
    that aren't specially handled.

    Args:
        arr: The input numpy array to transpose.

    Returns:
        The transposed numpy array.
    """
    ...

def ascontiguousarray(arr: NDArray[Any]) -> NDArray[Any]:
    """For up to four dimensional matrices, perform in-place transposition.

    Square matrices up to three dimensions are faster than numpy's out-of-place
    algorithm. Default to the out-of-place implementation numpy uses for cases
    that aren't specially handled.

    Args:
        arr: The input numpy array to transpose.

    Returns:
        The transposed numpy array.
    """
    ...

def minmax(
    arr: NDArray[Any],
) -> tuple[Union[int, float, None], Union[int, float, None]]:
    """Returns (min(arr), max(arr)) computed in a single pass.

    Returns (None, None) if array is size zero.
    """
    ...

def pixel_pairs(labels: NDArray[Any]) -> int:
    """Computes the number of matching adjacent memory locations.

    This is useful for rapidly evaluating whether an image is
    more binary or more connectomics like.
    """
    ...

def foreground(arr: NDArray[Any]) -> int:
    """Returns the number of non-zero voxels in an array."""
    ...

def point_cloud(arr: NDArray[Any]) -> dict[int, NDArray[Any]]:
    """Generate a mapping from labels to their (x, y, z) position in the image.

    Zero is considered a background label.

    Args:
        arr: A 2D or 3D numpy array.

    Returns:
        A dictionary mapping label values to their (x, y, z) coordinates in the
        image. The coordinates are stored as a numpy array of shape (N, 3),
        where N is the number of points for that label.
    """
    ...

def tobytes(
    image: NDArray[Any],
    chunk_size: tuple[int, int, int],
    order: str = "C",
) -> list[bytes]:
    """Compute the bytes with the image divided into a grid of cutouts.

    Return the resultant binaries indexed by their cutout's gridpoint in
    fortran order.

    This is faster than calling tobytes on each cutout individually if the input
    and output orders match.
    """
    ...
