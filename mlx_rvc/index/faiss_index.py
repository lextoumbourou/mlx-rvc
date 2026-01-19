"""FAISS index loading and feature blending for RVC."""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

# Set this before importing faiss to avoid OpenMP conflicts with MLX
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import faiss
    # Use single-threaded mode to avoid OpenMP conflicts with MLX
    # MLX and faiss both use OpenMP, and multi-threaded faiss search
    # causes segfaults when MLX has already initialized OpenMP
    faiss.omp_set_num_threads(1)
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class FaissIndex:
    """
    FAISS index for feature retrieval and blending.

    RVC uses FAISS to store ContentVec features from the training data.
    During inference, input features are blended with similar features
    from the training set, improving voice similarity.

    Example:
        index = FaissIndex.load("voice.index")
        blended = index.blend(features, index_rate=0.5)
    """

    def __init__(self, index, features: np.ndarray):
        """
        Args:
            index: FAISS index object
            features: All feature vectors from the index, shape (n_vectors, dim)
        """
        self.index = index
        self.features = features
        self.dimension = index.d
        self.n_vectors = index.ntotal

    @classmethod
    def load(cls, index_path: Union[str, Path]) -> "FaissIndex":
        """
        Load FAISS index from file.

        Args:
            index_path: Path to .index file

        Returns:
            FaissIndex instance
        """
        if not HAS_FAISS:
            raise ImportError(
                "FAISS is required for index blending. "
                "Install with: uv pip install faiss-cpu"
            )

        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        # Load the FAISS index
        index = faiss.read_index(str(index_path))

        # Reconstruct all vectors from the index
        # This is how RVC stores the training features
        features = index.reconstruct_n(0, index.ntotal)

        return cls(index, features)

    def search(
        self,
        query: np.ndarray,
        k: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Args:
            query: Query vectors, shape (n_queries, dim)
            k: Number of neighbors to retrieve

        Returns:
            Tuple of (distances, indices), each shape (n_queries, k)
        """
        # Ensure float32 for FAISS
        if query.dtype != np.float32:
            query = query.astype(np.float32)

        distances, indices = self.index.search(query, k)
        return distances, indices

    def blend(
        self,
        features: np.ndarray,
        index_rate: float = 0.5,
        k: int = 8,
    ) -> np.ndarray:
        """
        Blend input features with retrieved features from the index.

        The blending algorithm:
        1. For each input feature, find k nearest neighbors in the index
        2. Compute weights as inverse square of distances
        3. Weighted average of retrieved features
        4. Blend with original: output = retrieved * index_rate + original * (1 - index_rate)

        Args:
            features: Input features, shape (n_frames, dim) or (batch, n_frames, dim)
            index_rate: Blend ratio (0 = original only, 1 = retrieved only)
            k: Number of neighbors for retrieval

        Returns:
            Blended features, same shape as input
        """
        if index_rate == 0:
            return features

        # Handle batch dimension
        original_shape = features.shape
        if features.ndim == 3:
            batch_size = features.shape[0]
            features = features.reshape(-1, features.shape[-1])
        else:
            batch_size = None

        # Ensure float32 for FAISS search
        features_f32 = features.astype(np.float32) if features.dtype != np.float32 else features

        # Search for nearest neighbors
        distances, indices = self.search(features_f32, k=k)

        # Compute weights as inverse square distance
        # Add small epsilon to avoid division by zero
        weights = np.square(1.0 / (distances + 1e-9))
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Retrieve and weight the features
        # indices shape: (n_frames, k)
        # self.features[indices] shape: (n_frames, k, dim)
        retrieved = self.features[indices]  # (n_frames, k, dim)

        # Weighted sum: (n_frames, k, 1) * (n_frames, k, dim) -> sum -> (n_frames, dim)
        blended = np.sum(retrieved * weights[:, :, np.newaxis], axis=1)

        # Blend with original features
        output = blended * index_rate + features_f32 * (1 - index_rate)

        # Restore original dtype and shape
        if features.dtype != np.float32:
            output = output.astype(features.dtype)

        if batch_size is not None:
            output = output.reshape(original_shape)

        return output


def load_index(index_path: Union[str, Path]) -> Optional[FaissIndex]:
    """
    Load FAISS index from file.

    Returns None if FAISS is not installed or file doesn't exist.

    Args:
        index_path: Path to .index file

    Returns:
        FaissIndex instance or None
    """
    if not HAS_FAISS:
        return None

    index_path = Path(index_path)
    if not index_path.exists():
        return None

    return FaissIndex.load(index_path)
