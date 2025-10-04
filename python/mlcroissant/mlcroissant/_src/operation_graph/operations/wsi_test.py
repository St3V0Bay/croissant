"""WSI tests (non-hermetic).

These tests exercise the OpenSlide whole-slide reader and patch extraction.
"""

import pathlib

import pytest
from etils import epath

from mlcroissant._src.core.constants import EncodingFormat
from mlcroissant._src.operation_graph.operations.read import Read
from mlcroissant._src.core.path import Path
from mlcroissant._src.tests.nodes import create_test_file_object, create_test_field
from mlcroissant._src.structure_graph.nodes.source import FileProperty, Source


@pytest.mark.nonhermetic
def test_read_openslide_metadata(tmp_path):
    openslide = pytest.importorskip("openslide")
    import requests

    url = (
        "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/"
        "JP2K-33003-1.svs"
    )
    out = epath.Path(tmp_path) / "JP2K-33003-1.svs"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    file_obj = create_test_file_object(encoding_formats=[EncodingFormat.WHOLESLIDE])
    operation = Read(
        operations=None,  # not used in __repr__-free execution path
        node=file_obj,
        folder=epath.Path(tmp_path),
        fields=(),
    )
    df = operation.call(Path(filepath=out, fullpath=pathlib.PurePath()))
    assert {"wsi_properties", "wsi_dimensions", "wsi_level_count"}.issubset(df.columns)

    # Verify we can open the slide and read a small region using the returned filepath
    slide = openslide.OpenSlide(str(df.iloc[0][FileProperty.filepath]))
    patch = slide.read_region((0, 0), 0, (32, 32))
    assert patch.size == (32, 32)


@pytest.mark.nonhermetic
def test_wsi_patch_extraction(tmp_path):
    pytest.importorskip("openslide")
    import requests

    url = (
        "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/"
        "JP2K-33003-1.svs"
    )
    out = epath.Path(tmp_path) / "JP2K-33003-1.svs"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    # Build a minimal DataFrame-like row and Field to drive the patch extraction path.
    # We simulate a record with filepath + coordinates columns.
    import pandas as pd
    row = pd.Series({
        FileProperty.filepath: out,
        "x": 0,
        "y": 0,
        "w": 64,
        "h": 64,
    })

    # Create a content field with a WSI patch transform that reads coordinates
    # from x,y,w,h columns.
    field = create_test_field(
        source=Source(
            extract=Source.extract.__class__(file_property=FileProperty.content),
            transforms=[
                Source.transforms.__args__[0](  # type: ignore[attr-defined]
                    wsi_patch=True,
                    wsi_level=0,
                    wsi_x_column="x",
                    wsi_y_column="y",
                    wsi_width_column="w",
                    wsi_height_column="h",
                )
            ],
        )
    )

    # Call the internal extraction helper directly to avoid full pipeline overhead.
    from mlcroissant._src.operation_graph.operations.field import _extract_wsi_patch

    out_row = _extract_wsi_patch(row, field)
    img = out_row[FileProperty.content]
    # Should be a PIL.Image.Image and match requested size
    from PIL import Image as PIL_Image

    assert isinstance(img, PIL_Image.Image)
    assert img.size == (64, 64)