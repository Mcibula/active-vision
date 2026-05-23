"""
Utilities supporting the conceptual framework
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from processors.conceptualizers import MotionConceptExtractor

if TYPE_CHECKING:
    from structures import ConceptPoint


def load_traj_export(
        npz_path: str | Path,
        json_path: str | Path | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
        extractor: MotionConceptExtractor | None = None
) -> ConceptPoint:
    """
    Load one exported trajectory as a motion-domain conceptual point

    :param npz_path: Path to an exported `.npz` file containing `traj_vec`
    :param json_path: Optional path to the corresponding metadata `.json` file
    :param label: Optional conceptual label
    :param metadata: Optional metadata overriding or extending the JSON file metadata
    :param extractor: Initialized instance of a motion concept extractor used to
                      construct a conceptual point from the export
    """

    npz_path: Path = Path(npz_path)
    json_path: Path = (
        npz_path.with_suffix('.json')
        if json_path is None
        else Path(json_path)
    )

    with np.load(npz_path) as data:
        if 'traj_vec' not in data:
            raise KeyError

        vector = data['traj_vec']

    point_metadata: dict[str, Any] = {
        'npz_path': str(npz_path)
    }

    if json_path.exists():
        with json_path.open() as f:
            point_metadata.update(json.load(f))

        point_metadata['json_path'] = str(json_path)

    if metadata is not None:
        point_metadata.update(metadata)

    if label is None:
        label = (
            point_metadata.get('motion_label')
            or point_metadata.get('label')
        )

    extractor = extractor or MotionConceptExtractor()
    return extractor.from_vector(
        vector,
        label=label,
        metadata=point_metadata
    )


def load_traj_exports(
        export_dir: str | Path,
        manifest: str | None = None,
        extractor: MotionConceptExtractor | None = None
) -> list[ConceptPoint]:
    """
    Load multiple trajectory exports from a directory

    :param export_dir: Path to the directory to load trajectory exports from
    :param manifest: Path to the manifest file; if not `None`, only files listed in the manifest are loaded;
                     supported manifest formats are `.json` and `.csv`
    :param extractor: Initialized instance of a motion concept extractor used to
                      construct conceptual points from the exports
    """

    export_dir = Path(export_dir)
    extractor = extractor or MotionConceptExtractor()

    if manifest is None:
        return [
            load_traj_export(path, extractor=extractor)
            for path in sorted(export_dir.glob('*.npz'))
        ]

    manifest = Path(manifest)
    records = _load_manifest(manifest)
    points = []

    for record in records:
        fname = record.get('file') or record.get('npz') or record.get('path')
        if fname is None:
            raise KeyError

        npz_path = Path(fname)
        if not npz_path.is_absolute():
            npz_path = export_dir / npz_path

        json_name = record.get('json')
        json_path = None

        if json_name is not None:
            json_path = Path(json_name)

            if not json_path.is_absolute():
                json_path = export_dir / json_path

        label = (
            record.get('motion_label')
            or record.get('label')
        )
        metadata = {
            key: value
            for key, value in record.items()
            if key not in {'file', 'npz', 'path', 'json', 'label'}
        }

        points.append(
            load_traj_export(
                npz_path=npz_path,
                json_path=json_path,
                label=label,
                metadata=metadata,
                extractor=extractor
            )
        )

    return points


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix == '.json':
        with path.open() as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get('records', [])

        if not isinstance(data, list):
            raise ValueError

        return data

    if path.suffix == '.csv':
        with path.open(newline='') as f:
            return list(csv.DictReader(f))

    raise ValueError
