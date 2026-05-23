"""
Conceptual feature extraction
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from structures import ConceptDomain, ConceptPoint, ConceptSpace, QualityDimension, Snapshot, Trajectory
from .traj_vectorizer import TrajVectorizer


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


class VisualConceptExtractor:
    """
    Extract visual-domain conceptual points from object snapshots
    """

    def __init__(self, domain_name: str = 'visual') -> None:
        """
        Initialize the visual concept extractor

        :param domain_name: Name of the produced conceptual domain
        """

        self.domain_name: str = domain_name

    @property
    def dim(self) -> int:
        """
        Dimensionality of the produced visual vector
        """

        return len(self.domain)

    @property
    def domain(self) -> ConceptDomain:
        """
        Visual conceptual domain corresponding to the produced vector
        """

        dims = [
            QualityDimension('mask_area_ratio', bounds=(0.0, 1.0)),
            QualityDimension('mask_aspect_ratio'),
            QualityDimension('mask_extent', bounds=(0.0, 1.0)),
            QualityDimension('mask_solidity', bounds=(0.0, 1.0)),
            QualityDimension('mask_compactness', bounds=(0.0, 1.0)),
            QualityDimension('mask_eccentricity', bounds=(0.0, 1.0)),
            *[
                QualityDimension(f'hu_{idx}')
                for idx in range(7)
            ],
            *[
                QualityDimension(f'hsv_hist_{idx}', bounds=(0.0, 1.0))
                for idx in range(32)
            ],
            QualityDimension('edge_density', bounds=(0.0, 1.0)),
            *[
                QualityDimension(f'grad_hist_{idx}', bounds=(0.0, 1.0))
                for idx in range(8)
            ],
        ]

        return ConceptDomain(
            name=self.domain_name,
            dimensions=dims
        )

    @property
    def space(self) -> ConceptSpace:
        """
        Single-domain visual conceptual space
        """

        return ConceptSpace([self.domain])

    def __call__(
            self,
            snapshots: Snapshot | list[Snapshot],
            label: str | None = None,
            metadata: dict[str, Any] | None = None
    ) -> ConceptPoint:
        """
        Extract a conceptual point from one or more object snapshots

        :param snapshots: Snapshots to summarize visually
        :param label: Optional class/category label of the extracted point
        :param metadata: Optional non-conceptual provenance and quality data
        :return: Visual-domain conceptual point
        """

        if isinstance(snapshots, Snapshot):
            snapshots = [snapshots]

        if not snapshots:
            raise ValueError

        vectors = np.array(
            [
                self._snapshot_vector(snapshot)
                for snapshot in snapshots
            ],
            dtype=np.float32
        )

        return ConceptPoint(
            vector=vectors.mean(axis=0),
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
        Construct a visual-domain conceptual point from an existing vector

        :param vector: Existing visual-domain vector
        :param label: Optional class/category label of the constructed point
        :param metadata: Optional non-conceptual provenance and quality data
        :return: Visual-domain conceptual point
        """

        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.dim,):
            raise ValueError

        return ConceptPoint(
            vector=vector,
            label=label,
            metadata=metadata
        )

    @staticmethod
    def _snapshot_vector(snapshot: Snapshot) -> np.ndarray:
        """
        Compute a visual descriptor vector from a single object snapshot

        :param snapshot: Object snapshot to describe
        :return: Visual-domain descriptor vector
        """

        mask = snapshot.mask > 0

        if mask.shape != snapshot.rgb.shape[:2]:
            raise ValueError

        if not mask.any():
            raise ValueError

        mask_u8 = mask.astype(np.uint8)

        # Basic mask geometry
        area = np.count_nonzero(mask)
        area_ratio = area / mask.size

        ys, xs = np.nonzero(mask)
        mask_w = xs.max() - xs.min() + 1
        mask_h = ys.max() - ys.min() + 1

        aspect_ratio = mask_w / mask_h
        extent = area / (mask_w * mask_h)

        # Contour-based shape descriptors
        contours, _ = cv2.findContours(
            mask_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        contour_area = sum(cv2.contourArea(contour) for contour in contours)
        perimeter = sum(cv2.arcLength(contour, closed=True) for contour in contours)
        compactness = (
            4.0 * np.pi * contour_area / (perimeter ** 2)
            if perimeter > 0 else 0.0
        )
        compactness = float(np.clip(compactness, 0.0, 1.0))

        points = np.column_stack((xs, ys)).astype(np.int32).reshape(-1, 1, 2)
        if len(points) >= 3:
            hull = cv2.convexHull(points)
            hull_area = cv2.contourArea(hull)
            solidity = contour_area / hull_area if hull_area > 0 else 0.0
        else:
            solidity = 0.0

        solidity = float(np.clip(solidity, 0.0, 1.0))

        # Moment-based shape descriptors
        moments = cv2.moments(mask_u8)
        if moments['m00'] > 0:
            cov = np.array([
                [moments['mu20'], moments['mu11']],
                [moments['mu11'], moments['mu02']],
            ], dtype=np.float32) / moments['m00']

            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.maximum(eigvals, 0.0)
            eccentricity = np.sqrt(1.0 - eigvals[0] / eigvals[-1]) if eigvals[-1] > 0 else 0.0
        else:
            eccentricity = 0.0

        eccentricity = float(np.clip(eccentricity, 0.0, 1.0))

        hu = cv2.HuMoments(moments).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

        # Color distribution
        hsv = cv2.cvtColor(snapshot.rgb, cv2.COLOR_RGB2HSV)
        hsv_hist = cv2.calcHist(
            images=[hsv],
            channels=[0, 1],
            mask=(mask_u8 * 255),
            histSize=[8, 4],
            ranges=[0, 180, 0, 256]
        ).astype(np.float32)

        hist_sum = hsv_hist.sum()
        if hist_sum > 0:
            hsv_hist /= hist_sum

        hsv_hist = hsv_hist.flatten()

        # Edge and gradient texture
        gray = cv2.cvtColor(snapshot.rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        edge_density = np.count_nonzero((edges > 0) & mask) / area

        gray_f = gray.astype(np.float32) / 255.0
        grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        angle = np.mod(angle, 180.0)
        grad_mask = mask & (magnitude > 0)

        if grad_mask.any():
            grad_hist, _ = np.histogram(
                angle[grad_mask],
                bins=8,
                range=(0.0, 180.0),
                weights=magnitude[grad_mask]
            )
            grad_sum = grad_hist.sum()
            if grad_sum > 0:
                grad_hist = grad_hist / grad_sum
        else:
            grad_hist = np.zeros(8, dtype=np.float32)

        return np.array(
            [
                # Mask features
                area_ratio,
                aspect_ratio,
                extent,

                # Contour features
                solidity,
                compactness,

                # Moment-based shape features
                eccentricity,
                *hu_log,

                # Color features
                *hsv_hist,

                # Edge and texture features
                edge_density,
                *grad_hist,
            ],
            dtype=np.float32
        )
