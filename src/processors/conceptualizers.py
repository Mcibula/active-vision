"""
Conceptual feature extraction
"""

from __future__ import annotations

from typing import Any

import numpy as np

from processors import TrajVectorizer
from structures import ConceptDomain, ConceptPoint, ConceptSpace, QualityDimension, Trajectory


class MotionConceptExtractor:
    """
    Extract motion-domain conceptual points from trajectories
    """

    def __init__(
            self,
            vectorizer: TrajVectorizer | None = None,
            domain_name: str = 'motion'
    ) -> None:
        """
        Initialize the motion concept extractor

        :param vectorizer: Trajectory vectorizer used to produce motion signatures
        :param domain_name: Name of the produced conceptual domain
        """

        self.vectorizer: TrajVectorizer = vectorizer or TrajVectorizer()
        self.domain_name: str = domain_name

    @property
    def dim(self) -> int:
        """
        Dimensionality of the produced motion vector
        """

        return (self.vectorizer.n_pos + self.vectorizer.n_rot) * 3

    @property
    def domain(self) -> ConceptDomain:
        """
        Motion conceptual domain corresponding to the produced vector
        """

        dimensions = [
            QualityDimension(name=f'motion_{idx}')
            for idx in range(self.dim)
        ]

        return ConceptDomain(self.domain_name, dimensions)

    @property
    def space(self) -> ConceptSpace:
        """
        Single-domain motion conceptual space
        """

        return ConceptSpace([self.domain])

    def __call__(
            self,
            traj: Trajectory,
            label: str | None = None,
            metadata: dict[str, Any] | None = None
    ) -> ConceptPoint:
        """
        Extract a conceptual point from a trajectory

        :param traj: Trajectory to vectorize
        :param label: Optional class/category label of the extracted point
        :param metadata: Optional non-conceptual provenance and quality data
        :return: Motion-domain conceptual point
        """

        vector = self.vectorizer(traj)
        return ConceptPoint(
            vector=vector,
            label=label,
            metadata=metadata
        )

    def from_vector(
            self,
            vector: np.ndarray,
            label: str | None = None,
            metadata: dict[str, Any] | None = None
    ) -> ConceptPoint:
        """
        Construct a motion-domain conceptual point from an existing vector

        :param vector: Existing motion-domain vector
        :param label: Optional class/category label of the constructed point
        :param metadata: Optional non-conceptual provenance and quality data
        :return: Motion-domain conceptual point
        """

        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.dim,):
            raise ValueError

        return ConceptPoint(
            vector=vector,
            label=label,
            metadata=metadata
        )
