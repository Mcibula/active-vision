"""
Data structures operating with conceptual spaces
"""

from __future__ import annotations

from typing import Any, Callable, Iterator

import numpy as np


class QualityDimension:
    """
    Single quality dimension of a conceptual domain
    """

    def __init__(
            self,
            name: str,
            unit: str | None = None,
            weight: float = 1.0,
            bounds: tuple[float, float] | None = None
    ) -> None:
        """
        Initialize a quality dimension

        :param name: Dimension name
        :param unit: Optional physical or conventional unit
        :param weight: Dimension weight used by distance computation
        :param bounds: Optional normalization bounds as `(min, max)`
        """

        if not name:
            raise ValueError

        if weight <= 0:
            raise ValueError

        if bounds is not None and bounds[0] >= bounds[1]:
            raise ValueError

        self.name: str = name
        self.unit: str | None = unit
        self.weight: float = float(weight)
        self.bounds: tuple[float, float] | None = bounds

    def __repr__(self) -> str:
        return f'<QualityDimension "{self.name}">'


class ConceptDomain:
    """
    Domain grouping integral quality dimensions
    """

    def __init__(self, name: str, dimensions: list[QualityDimension]) -> None:
        """
        Initialize a conceptual domain

        :param name: Domain name
        :param dimensions: Integral quality dimensions of the domain
        """

        if not name or not dimensions:
            raise ValueError

        if len({dim.name for dim in dimensions}) != len(dimensions):
            raise ValueError

        self.name: str = name
        self._dimensions: list[QualityDimension] = dimensions

    def __repr__(self) -> str:
        return f'<ConceptDomain "{self.name}" with {self.dim} dimensions>'

    def __len__(self) -> int:
        return len(self._dimensions)

    def __iter__(self) -> Iterator[QualityDimension]:
        yield from self._dimensions

    @property
    def dimensions(self) -> list[QualityDimension]:
        """
        List of integral quality dimensions in this domain
        """

        return self._dimensions

    @property
    def dim(self) -> int:
        """
        Number of quality dimensions in this domain
        """

        return len(self)

    @property
    def names(self) -> list[str]:
        """
        Names of all quality dimensions in this domain
        """

        return [
            dim.name
            for dim in self
        ]

    @property
    def weights(self) -> np.ndarray:
        """
        Dimension weights as a vector
        """

        return np.array(
            [
                dim.weight
                for dim in self
            ],
            dtype=np.float32
        )

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """
        Normalize values according to dimension bounds where bounds are available

        :param values: Domain values to normalize
        :return: Normalized domain values
        """

        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1] != self.dim:
            raise ValueError

        norm = values.copy()
        for idx, dim in enumerate(self):
            if dim.bounds is None:
                continue

            low, high = dim.bounds
            norm[..., idx] = (norm[..., idx] - low) / (high - low)

        return norm

    def l2_dist(self, a: np.ndarray, b: np.ndarray, normalize: bool = True) -> float:
        """
        Calculate weighted Euclidean (L2) distance within this domain

        :param a: First domain vector
        :param b: Second domain vector
        :param normalize: Whether to normalize values before distance computation
        :return: Weighted Euclidean distance
        """

        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        if a.shape != (self.dim,) or b.shape != (self.dim,):
            raise ValueError

        if normalize:
            a = self.normalize(a)
            b = self.normalize(b)

        diff = (a - b) * self.weights
        return float(np.linalg.norm(diff, ord=2))


class ConceptPoint:
    """
    Point representing one exemplar in a conceptual space
    """

    def __init__(
            self,
            vector: np.ndarray,
            label: str | None = None,
            metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize a conceptual point

        :param vector: Numeric conceptual-space vector
        :param label: Optional class/category label
        :param metadata: Optional non-conceptual provenance and quality data
        """

        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or len(vector) == 0:
            raise ValueError

        if not np.isfinite(vector).all():
            raise ValueError

        self._vector: np.ndarray = vector
        self.label: str | None = label
        self.metadata: dict[str, Any] = (
            metadata.copy()
            if metadata is not None
            else {}
        )

    def __repr__(self) -> str:
        label = (
            f' "{self.label}"'
            if self.label is not None
            else ''
        )

        return f'<ConceptPoint{label} in R^{self.dim}>'

    @property
    def vector(self) -> np.ndarray:
        """
        Numeric conceptual vector representing this exemplar
        """

        return self._vector

    @property
    def dim(self) -> int:
        """
        Dimensionality of the conceptual point
        """

        return len(self._vector)


class ConceptPrototype(ConceptPoint):
    """
    Prototype represented by the centroid of multiple conceptual points
    """

    def __init__(
            self,
            label: str,
            vector: np.ndarray,
            support: int,
            metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize a concept prototype

        :param label: Prototype label
        :param vector: Prototype vector
        :param support: Number of exemplars used to compute this prototype
        :param metadata: Optional metadata
        """

        if support <= 0:
            raise ValueError

        super().__init__(vector=vector, label=label, metadata=metadata)
        self.support: int = int(support)

    def __repr__(self) -> str:
        return f'<ConceptPrototype "{self.label}" supported by {self.support} exemplars in R^{self.dim}>'


class ConceptSpace:
    """
    Metric conceptual space composed of one or more domains
    """

    def __init__(self, domains: list[ConceptDomain]) -> None:
        """
        Initialize a conceptual space

        :param domains: Conceptual domains composing the space
        """

        if not domains:
            raise ValueError

        if len({domain.name for domain in domains}) != len(domains):
            raise ValueError

        self._domains: list[ConceptDomain] = domains
        self._slices: dict[str, slice] = {}

        start = 0
        for domain in domains:
            stop = start + domain.dim
            self._slices[domain.name] = slice(start, stop)
            start = stop

    def __repr__(self) -> str:
        return f'<ConceptSpace with {len(self)} domains in R^{self.dim}>'

    def __len__(self) -> int:
        return len(self._domains)

    def __iter__(self) -> Iterator[ConceptDomain]:
        yield from self._domains

    @property
    def domains(self) -> list[ConceptDomain]:
        return self._domains

    @property
    def dim(self) -> int:
        """
        Total dimensionality of this conceptual space
        """

        return sum(domain.dim for domain in self)

    @property
    def domain_names(self) -> list[str]:
        """
        Names of domains composing the conceptual space
        """

        return [domain.name for domain in self]

    def domain_slice(self, name: str) -> slice:
        """
        Vector slice corresponding to a domain

        :param name: Domain name
        :return: Slice of the conceptual vector corresponding to the domain
        """

        if name not in self._slices:
            raise KeyError

        return self._slices[name]

    def validate(self, point: ConceptPoint) -> None:
        """
        Validate whether a point belongs to this conceptual space

        :param point: Conceptual point to validate
        """

        if point.dim != self.dim:
            raise ValueError

    def distance(
            self,
            a: ConceptPoint,
            b: ConceptPoint,
            domain_weights: dict[str, float] | None = None,
            normalize: bool = True
    ) -> float:
        r"""
        Calculate distance between two conceptual points.
        Domain distances are computed independently and then weight-summed, i.e.,
        .. math::
          \mathrm{dist}(x,y) = \sum_i w_i \lVert x_i - y_i \rVert_2

        :param a: Conceptual point
        :param b: Conceptual point
        :param domain_weights: Optional weights to use for weighting the domain distances
        :param normalize: Whether to normalize each domain distance
        :return: Domain-weighted conceptual distance
        """

        self.validate(a)
        self.validate(b)

        total = 0.0
        domain_weights = domain_weights or {}

        for domain in self:
            dom_slice = self.domain_slice(domain.name)
            weight = domain_weights.get(domain.name, 1.0)
            if weight < 0:
                raise ValueError

            total += weight * domain.l2_dist(
                a.vector[dom_slice],
                b.vector[dom_slice],
                normalize=normalize
            )

        return float(total)


class ConceptMemory:
    """
    Memory storing conceptual exemplars and computing prototypes
    """

    def __init__(self, space: ConceptSpace) -> None:
        """
        Initialize an empty conceptual memory

        :param space: Conceptual space in which all points are represented
        """

        self.space: ConceptSpace = space
        self._points: list[ConceptPoint] = []

    def __repr__(self) -> str:
        return f'<ConceptMemory with {len(self)} points in R^{self.space.dim}>'

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self) -> Iterator[ConceptPoint]:
        yield from self._points

    @property
    def points(self) -> list[ConceptPoint]:
        """
        All the conceptual points stored in this memory
        """

        return self._points

    @property
    def vectors(self) -> np.ndarray:
        r"""
        Stored conceptual points as an :math:`N \times D` matrix
        """

        if not self._points:
            return np.empty((0, self.space.dim), dtype=np.float32)

        return np.array([point.vector for point in self._points], dtype=np.float32)

    @property
    def labels(self) -> list[str | None]:
        """
        Labels of stored conceptual points
        """

        return [point.label for point in self._points]

    def add(self, point: ConceptPoint) -> None:
        """
        Store a conceptual point

        :param point: Conceptual point to store
        """

        self.space.validate(point)
        self._points.append(point)

    def extend(self, points: list[ConceptPoint]) -> None:
        """
        Store multiple conceptual points

        :param points: Conceptual points to store
        """

        for point in points:
            self.add(point)

    def nearest(
            self,
            query: ConceptPoint,
            k: int = 1,
            domain_weights: dict[str, float] | None = None
    ) -> list[tuple[ConceptPoint, float]]:
        """
        Find the nearest stored conceptual points

        :param query: A conceptual point to find the neighbors for
        :param k: Number of nearest neighbors to return
        :param domain_weights: Optional weights to use for weighting the domain distances
        :return: Nearest conceptual points together with their distances
        """

        if k <= 0:
            raise ValueError

        self.space.validate(query)
        distances = [
            (
                point,
                self.space.distance(
                    a=query, b=point,
                    domain_weights=domain_weights
                )
            )
            for point in self._points
        ]
        distances.sort(key=lambda item: item[1])

        return distances[:k]

    def prototypes(self, label_key: str | None = None) -> dict[str, ConceptPrototype]:
        """
        Compute prototype centroids grouped by labels

        :param label_key: Optional metadata key to use for grouping;
                          if `None`, `ConceptPoint.label` is used
        :return: Prototype centroids indexed by label
        """

        groups: dict[str, list[ConceptPoint]] = {}
        for point in self._points:
            label = (
                point.metadata.get(label_key)
                if label_key is not None
                else point.label
            )

            if label is None:
                continue

            groups.setdefault(str(label), []).append(point)

        prototypes = {}
        for label, points in groups.items():
            vectors = np.array([point.vector for point in points], dtype=np.float32)
            prototypes[label] = ConceptPrototype(
                label=label,
                vector=vectors.mean(axis=0),
                support=len(points)
            )

        return prototypes

    @classmethod
    def from_points(cls, space: ConceptSpace, points: list[ConceptPoint]) -> ConceptMemory:
        """
        Initialize conceptual memory from points

        :param space: Conceptual space in which all points are represented
        :param points: Conceptual points to store
        :return: Conceptual memory initialized from the points
        """

        memory = cls(space)
        memory.extend(points)

        return memory


class ConceptEntity:
    """
    Entity represented by optional projections into multiple conceptual spaces
    """

    def __init__(
            self,
            entity_id: str,
            projections: dict[str, ConceptPoint] | None = None,
            label: str | None = None,
            metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Initialize a conceptual entity with projections into named spaces

        :param entity_id: Stable entity identifier
        :param projections: Optional conceptual projections indexed by space key
        :param label: Optional class/category label
        :param metadata: Optional non-conceptual provenance and quality data
        """

        if not entity_id:
            raise ValueError

        self.entity_id: str = entity_id
        self.label: str | None = label
        self.metadata: dict[str, Any] = (
            metadata.copy()
            if metadata is not None
            else {}
        )
        self._projections: dict[str, ConceptPoint] = {}

        if projections is not None:
            for space, point in projections.items():
                self.add_projection(space, point)

    def __repr__(self) -> str:
        label = (
            f' "{self.label}"'
            if self.label is not None
            else ''
        )

        return f'<ConceptEntity{label} "{self.entity_id}" with {len(self)} projections>'

    def __len__(self) -> int:
        return len(self._projections)

    @property
    def projections(self) -> dict[str, ConceptPoint]:
        """
        Conceptual projections indexed by space key
        """

        return self._projections

    @property
    def available_spaces(self) -> list[str]:
        """
        Names of conceptual spaces into which this entity is projected
        """

        return list(self._projections)

    def add_projection(self, space: str, point: ConceptPoint) -> None:
        """
        Add or replace a conceptual-space projection

        :param space: Conceptual space key
        :param point: Conceptual point representing the entity in that space
        """

        if not space:
            raise ValueError

        self._projections[space] = point

    def has_projection(self, space: str) -> bool:
        """
        Check whether this entity has a projection into a conceptual space

        :param space: Conceptual space key
        :return: Whether a projection for the requested space is available
        """

        return space in self._projections

    def projection(self, space: str) -> ConceptPoint:
        """
        Return a conceptual-space projection

        :param space: Conceptual space key
        :return: Conceptual point representing the entity in the requested space
        """

        if space not in self._projections:
            raise KeyError

        return self._projections[space]

    def shared_spaces(
            self,
            other: ConceptEntity,
            required_spaces: list[str] | None = None
    ) -> list[str]:
        """
        Find conceptual spaces shared with another entity

        :param other: Entity to compare with
        :param required_spaces: Optional spaces that must be available in both entities
        :return: Space keys available in both entities
        """

        shared = [
            space
            for space in self.available_spaces
            if other.has_projection(space)
        ]

        if required_spaces is None:
            return shared

        missing = [
            space
            for space in required_spaces
            if space not in shared
        ]
        if missing:
            raise ValueError

        return shared

    def distance(
            self,
            other: ConceptEntity,
            spaces: dict[str, ConceptSpace],
            space_weights: dict[str, float] | None = None,
            required_spaces: list[str] | None = None,
            space_domain_weights: dict[str, dict[str, float]] | None = None,
            normalize: bool = True
    ) -> float:
        """
        Calculate distance to another conceptual entity

        :param other: Entity to compare with
        :param spaces: Conceptual spaces indexed by space key
        :param space_weights: Optional weights for conceptual-space distances
        :param required_spaces: Optional spaces that must be available in both entities
        :param space_domain_weights: Optional per-space weights for internal domain distances
        :param normalize: Whether to normalize distances inside each conceptual space
        :return: Weighted average distance over comparable spaces
        """

        space_weights = space_weights or {}
        space_domain_weights = space_domain_weights or {}
        shared = self.shared_spaces(other, required_spaces)

        total = 0.0
        used_weight = 0.0

        for space in shared:
            if space not in spaces:
                raise KeyError

            weight = space_weights.get(space, 1.0)

            if weight < 0:
                raise ValueError

            if weight == 0:
                continue

            total += weight * spaces[space].distance(
                a=self.projection(space),
                b=other.projection(space),
                domain_weights=space_domain_weights.get(space),
                normalize=normalize
            )
            used_weight += weight

        if used_weight == 0:
            raise ValueError

        return float(total / used_weight)


class ConceptEntityMemory:
    """
    Memory storing conceptual entities with projections into named spaces
    """

    def __init__(self, spaces: dict[str, ConceptSpace]) -> None:
        """
        Initialize an empty conceptual entity memory

        :param spaces: Conceptual spaces indexed by space key
        """

        if not spaces:
            raise ValueError

        self.spaces: dict[str, ConceptSpace] = spaces
        self._entities: list[ConceptEntity] = []

    def __repr__(self) -> str:
        return f'<ConceptEntityMemory with {len(self)} entities and {len(self.spaces)} spaces>'

    def __len__(self) -> int:
        return len(self._entities)

    def __iter__(self) -> Iterator[ConceptEntity]:
        yield from self._entities

    @property
    def entities(self) -> list[ConceptEntity]:
        """
        All conceptual entities stored in this memory
        """

        return self._entities

    def validate(self, entity: ConceptEntity) -> None:
        """
        Validate whether an entity belongs to this memory

        :param entity: Conceptual entity to validate
        """

        for space, point in entity.projections.items():
            if space not in self.spaces:
                raise KeyError

            self.spaces[space].validate(point)

    def add(self, entity: ConceptEntity) -> None:
        """
        Store a conceptual entity

        :param entity: Conceptual entity to store
        """

        self.validate(entity)
        self._entities.append(entity)

    def extend(self, entities: list[ConceptEntity]) -> None:
        """
        Store multiple conceptual entities

        :param entities: Conceptual entities to store
        """

        for entity in entities:
            self.add(entity)

    def distance(
            self,
            a: ConceptEntity,
            b: ConceptEntity,
            space_weights: dict[str, float] | None = None,
            required_spaces: list[str] | None = None,
            space_domain_weights: dict[str, dict[str, float]] | None = None,
            normalize: bool = True
    ) -> float:
        """
        Calculate distance between two stored-compatible entities

        :param a: First conceptual entity
        :param b: Second conceptual entity
        :param space_weights: Optional weights for conceptual-space distances
        :param required_spaces: Optional spaces that must be available in both entities
        :param space_domain_weights: Optional per-space weights for internal domain distances
        :param normalize: Whether to normalize distances inside each conceptual space
        :return: Weighted average distance over comparable spaces
        """

        self.validate(a)
        self.validate(b)

        return a.distance(
            other=b,
            spaces=self.spaces,
            space_weights=space_weights,
            required_spaces=required_spaces,
            space_domain_weights=space_domain_weights,
            normalize=normalize
        )

    def nearest(
            self,
            query: ConceptEntity,
            k: int = 1,
            space_weights: dict[str, float] | None = None,
            required_spaces: list[str] | None = None,
            space_domain_weights: dict[str, dict[str, float]] | None = None
    ) -> list[tuple[ConceptEntity, float]]:
        """
        Find nearest stored entities over comparable spaces

        :param query: Conceptual entity to find the neighbors for
        :param k: Number of nearest neighbors to return
        :param space_weights: Optional weights for conceptual-space distances
        :param required_spaces: Optional spaces that must be available in both entities
        :param space_domain_weights: Optional per-space weights for internal domain distances
        :return: Nearest conceptual entities together with their distances
        """

        if k <= 0:
            raise ValueError

        self.validate(query)
        distances = []

        for entity in self._entities:
            try:
                distance = self.distance(
                    a=query, b=entity,
                    space_weights=space_weights,
                    required_spaces=required_spaces,
                    space_domain_weights=space_domain_weights
                )
            except ValueError:
                continue

            distances.append((entity, distance))

        distances.sort(key=lambda item: item[1])
        return distances[:k]

    @classmethod
    def from_entities(
            cls,
            spaces: dict[str, ConceptSpace],
            entities: list[ConceptEntity]
    ) -> ConceptEntityMemory:
        """
        Initialize conceptual entity memory from entities

        :param spaces: Conceptual spaces indexed by space key
        :param entities: Conceptual entities to store
        :return: Conceptual entity memory initialized from the entities
        """

        memory = cls(spaces)
        memory.extend(entities)

        return memory

    @classmethod
    def from_space_memories(
            cls,
            memories: dict[str, ConceptMemory],
            key_fn: Callable[[str, ConceptPoint], str],
            label_fn: Callable[[str, ConceptPoint], str | None] | None = None,
            metadata_fn: Callable[[str, ConceptPoint], dict[str, Any]] | None = None
    ) -> ConceptEntityMemory:
        """
        Initialize entity memory by aligning space-specific memories

        :param memories: Conceptual memories indexed by space key
        :param key_fn: Function mapping `(space_key, point)` to a stable entity id
        :param label_fn: Optional function mapping `(space_key, point)` to an entity label
        :param metadata_fn: Optional function mapping `(space_key, point)` to entity metadata
        :return: Conceptual entity memory initialized from space memories
        """

        if not memories:
            raise ValueError

        spaces: dict[str, ConceptSpace] = {
            space: memory.space
            for space, memory in memories.items()
        }
        projections_by_id: dict[str, dict[str, ConceptPoint]] = {}
        labels_by_id: dict[str, str | None] = {}
        metadata_by_id: dict[str, dict[str, Any]] = {}

        for space, memory in memories.items():
            if not space:
                raise ValueError

            for point in memory:
                entity_id = key_fn(space, point)
                if not entity_id:
                    raise ValueError

                projections = projections_by_id.setdefault(entity_id, {})
                if space in projections:
                    raise ValueError

                projections[space] = point

                label = (
                    label_fn(space, point)
                    if label_fn is not None
                    else point.label
                )
                if entity_id not in labels_by_id or labels_by_id[entity_id] is None:
                    labels_by_id[entity_id] = label

                if metadata_fn is not None:
                    metadata = metadata_fn(space, point)
                    metadata_by_id.setdefault(entity_id, {}).update(metadata)

        entities = [
            ConceptEntity(
                entity_id=entity_id,
                projections=projections_by_id[entity_id],
                label=labels_by_id.get(entity_id),
                metadata=metadata_by_id.get(entity_id)
            )
            for entity_id in sorted(projections_by_id)
        ]

        return cls.from_entities(spaces, entities)
