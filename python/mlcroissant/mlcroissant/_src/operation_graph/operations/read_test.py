"""Read operation tests."""

import pathlib
import pickle
import tempfile
from unittest import mock

from etils import epath
import numpy as np
import pandas as pd
import pandas.testing as pd_testing
import pytest

from mlcroissant._src.core.constants import EncodingFormat
from mlcroissant._src.core.path import Path
from mlcroissant._src.operation_graph.operations.read import _read_arff_file
from mlcroissant._src.operation_graph.operations.read import _reading_method
from mlcroissant._src.operation_graph.operations.read import Read
from mlcroissant._src.operation_graph.operations.read import ReadingMethod
from mlcroissant._src.structure_graph.nodes.file_object import FileObject
from mlcroissant._src.structure_graph.nodes.source import Extract
from mlcroissant._src.structure_graph.nodes.source import FileProperty
from mlcroissant._src.structure_graph.nodes.source import Source
from mlcroissant._src.tests.nodes import create_test_field
from mlcroissant._src.tests.nodes import create_test_file_object
from mlcroissant._src.tests.nodes import empty_file_object
from mlcroissant._src.tests.operations import operations

# Example from scipy docs (loadarff example):
# https://docs.scipy.org/doc/scipy-1.13.0/reference/generated/
# scipy.io.arff.loadarff.html
ARFF_CONTENT = """@relation foo
@attribute width  numeric
@attribute height numeric
@attribute color  {red,green,blue,yellow,black}
@data
5.0,3.25,blue
4.5,3.75,green
3.0,4.00,red
"""


@pytest.fixture
def file_object_with_missing_encoding_format():
    """FileObject with a missing encoding format."""  # noqa: D403
    return create_test_file_object(encoding_formats=None)


def test_str_representation():
    """String representation contains the file object name."""
    operation = Read(
        operations=operations(),
        node=empty_file_object,
        folder=epath.Path(),
        fields=(),
    )
    assert str(operation) == "Read(file_object_name)"


def test_reading_arff():
    """ARFF files are read into a DataFrame with expected rows/cols."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(ARFF_CONTENT)
        filepath = f.name
    actual_df = _read_arff_file(filepath)
    data = [(5.0, 3.25, b"blue"), (4.5, 3.75, b"green"), (3.0, 4.0, b"red")]
    expected_df = pd.DataFrame(
        data,
        columns=["width", "height", "color"],
    )
    pd_testing.assert_frame_equal(actual_df, expected_df)


def test_explicit_message_when_pyarrow_is_not_installed():
    """Helpful ImportError is raised when pyarrow is missing."""
    with mock.patch.object(pd, "read_parquet", side_effect=ImportError):
        with tempfile.TemporaryDirectory() as folder:
            content_url = "file.parquet"
            folder = epath.Path(folder)
            # Create filepath = `folder/file.parquet`.
            filepath = folder / content_url
            file = Path(
                filepath=filepath,
                fullpath=pathlib.PurePath(),
            )
            filepath.touch()
            read = Read(
                operations=operations(),
                node=create_test_file_object(
                    encoding_formats=["application/x-parquet"],
                    content_url=content_url,
                ),
                folder=folder,
                fields=(),
            )
            with pytest.raises(
                ImportError,
                match=r".*pip install mlcroissant\[parquet\].*",
            ):
                read.call([file])


def test_reading_method():
    """Reading method resolution is correct for each field type."""
    json_field = create_test_field(
        source=Source(extract=Extract(json_path="path"))
    )
    column_field = create_test_field(
        source=Source(extract=Extract(column="column"))
    )
    content_field = create_test_field(
        source=Source(extract=Extract(file_property=FileProperty.content))
    )
    lines_field = create_test_field(
        source=Source(extract=Extract(file_property=FileProperty.lines))
    )
    filename = create_test_field(
        source=Source(extract=Extract(file_property=FileProperty.filename))
    )
    assert _reading_method(
        empty_file_object, (json_field, filename)
    ) == ReadingMethod.JSON
    assert _reading_method(
        empty_file_object, (column_field, filename)
    ) == ReadingMethod.CONTENT
    assert _reading_method(
        empty_file_object, (content_field, filename)
    ) == ReadingMethod.CONTENT
    assert _reading_method(
        empty_file_object, (lines_field, filename)
    ) == ReadingMethod.LINES
    assert _reading_method(
        empty_file_object, (filename,)
    ) == ReadingMethod.NONE
    with pytest.raises(ValueError):
        _reading_method(empty_file_object, (content_field, lines_field))


@pytest.mark.parametrize(("encoding_format"), [(EncodingFormat.DICOM)])
def test_read_dicom_file(
    encoding_format: str,
    tmpdir: epath.Path,
    file_object_with_missing_encoding_format: FileObject,
):
    """DICOM reader returns a non-empty DataFrame for a minimal DICOM file."""
    pydicom = pytest.importorskip("pydicom")
    filepath = tmpdir / "file.dcm"
    # Create a dummy DICOM file
    pixel_array = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = pydicom.dataset.FileDataset(
        filepath,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    ds.PatientName = "Test^Patient"
    ds.add_new(0x00280002, "US", 1)  # Samples per Pixel
    ds.add_new(0x00280004, "CS", "MONOCHROME2")  # Photometric Interpretation
    ds.add_new(0x00280010, "US", 2)  # Rows
    ds.add_new(0x00280011, "US", 2)  # Columns
    ds.add_new(0x00280100, "US", 8)  # Bits Allocated
    ds.add_new(0x00280101, "US", 8)  # Bits Stored
    ds.add_new(0x00280102, "US", 7)  # High Bit
    ds.add_new(0x00280103, "US", 0)  # Pixel Representation
    ds.PixelData = pixel_array.tobytes()

    ds.save_as(filepath)
    file_object_with_missing_encoding_format.encoding_formats = [
        encoding_format
    ]
    operation = Read(
        operations=operations(),
        node=file_object_with_missing_encoding_format,
        folder=tmpdir,
        fields=(),
    )
    content = operation.call(Path(filepath=filepath, fullpath=filepath))
    assert not content.empty


def test_pickable():
    """Read operation is picklable and preserves folder path."""
    operation = Read(
        operations=operations(),
        node=empty_file_object,
        folder=epath.Path("/foo/bar"),
        fields=(),
    )
    operation = pickle.loads(pickle.dumps(operation))
    assert operation.folder == epath.Path("/foo/bar")


def test_read_dicom_file_returns_expected_bytes_and_fileprops(
    tmpdir: epath.Path, file_object_with_missing_encoding_format: FileObject
):
    """DICOM content bytes and file properties are correctly returned."""
    pydicom = pytest.importorskip("pydicom")

    filepath = tmpdir / "file.dcm"

    # Minimal valid DICOM with 2x2 8-bit monochrome image
    pixel_array = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = pydicom.dataset.FileDataset(
        filepath,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    ds.PatientName = "Test^Patient"
    ds.add_new(0x00280002, "US", 1)  # Samples per Pixel
    ds.add_new(0x00280004, "CS", "MONOCHROME2")  # Photometric Interpretation
    ds.add_new(0x00280010, "US", 2)  # Rows
    ds.add_new(0x00280011, "US", 2)  # Columns
    ds.add_new(0x00280100, "US", 8)  # Bits Allocated
    ds.add_new(0x00280101, "US", 8)  # Bits Stored
    ds.add_new(0x00280102, "US", 7)  # High Bit
    ds.add_new(0x00280103, "US", 0)  # Pixel Representation
    ds.PixelData = pixel_array.tobytes()

    ds.save_as(filepath)

    file_object_with_missing_encoding_format.encoding_formats = [
        EncodingFormat.DICOM
    ]
    operation = Read(
        operations=operations(),
        node=file_object_with_missing_encoding_format,
        folder=tmpdir,
        fields=(),
    )

    df = operation.call(
        Path(filepath=filepath, fullpath=filepath)
    )
    assert len(df) == 1
    row = df.iloc[0]

    # Verify content bytes and file properties are set
    assert row[FileProperty.content] == pixel_array.tobytes()
    assert row[FileProperty.filename] == "file.dcm"
    assert row[FileProperty.filepath] == filepath
    expected_fullpath = filepath
    assert row[FileProperty.fullpath] == expected_fullpath


def test_read_dicom_file_raises_when_pydicom_missing(
    tmpdir: epath.Path, monkeypatch
):
    """Helpful ImportError is raised when pydicom is missing."""
    # Import the module to patch its optional deps
    import mlcroissant._src.operation_graph.operations.read as read_mod

    class _DummyDeps:
        @property
        def pydicom(self):  # pragma: no cover
            raise ImportError("simulated missing pydicom")

    monkeypatch.setattr(read_mod, "deps", _DummyDeps())

    # No actual file read should happen since dependency check fails first
    with pytest.raises(
        ImportError,
        match="Missing dependency to read DICOM files",
    ):
        read_mod._read_dicom_file(tmpdir / "does_not_matter.dcm")
