"""Source module."""

from __future__ import annotations

import collections
import dataclasses
import enum
import logging
import re
from typing import Any

from ml_croissant._src.core import constants
from ml_croissant._src.core.issues import Issues
from ml_croissant._src.structure_graph.base_node import ID_REGEX, validate_name


def parse_reference(issues: Issues, source_data: str) -> tuple[str, ...]:
    """Parses a reference from a string called `source_data`."""
    source_regex = re.compile(rf"^\#\{{({ID_REGEX})(?:\/([^\/]+))*\}}$")
    match = source_regex.match(source_data)
    if match is None:
        issues.add_error(
            f"Malformed source data: {source_data}. The source data should be written"
            " as `#{name}` where name is valid ID."
        )
        return ()
    groups = tuple(group for group in match.groups() if group is not None)
    # Only validate the root group, because others can point to external columns
    # (like in a CSV) with fuzzy names.
    validate_name(issues, groups[0])
    return groups


class FileProperty(enum.Enum):
    content = enum.auto()
    filename = enum.auto()
    filepath = enum.auto()
    fullpath = enum.auto()


def is_file_property(file_property: str):
    for possible_file_property in FileProperty:
        if possible_file_property.name == file_property:
            return True
    return False


@dataclasses.dataclass(frozen=True)
class DataExtraction:
    csv_column: str | None = None
    file_property: FileProperty | None = None
    json_path: str | None = None


@dataclasses.dataclass(frozen=True)
class Transform:
    """Container for transformation.

    Args:
        format: The format for a date, e.g. "%Y-%m-%d %H:%M:%S.%f".
        regex: A regex pattern with a capturing group to extract information in a
            string.
        replace: A replace pattern, e.g. "pattern_to_remove/pattern_to_add".
        separator: A separator in a string to yield a list.
    """

    format: str | None = None
    regex: str | None = None
    replace: str | None = None
    separator: str | None = None


@dataclasses.dataclass(frozen=True)
class Source:
    r"""Python representation of sources and references.

    Croissant accepts several manners to declare sources:

    When the origin is a field:

    ```json
    "source": {
        "field": "record_set/name",
    }
    ```

    When the origin is a FileSet or a FileObject:

    ```json
    "source": {
        "distribution": "my-csv",
        "dataExtraction": {
            "csvColumn": "my-csv-column"
        }
    }
    ```

    See the specs for all supported parameters by `dataExtraction`.

    You can also add one or more transformations with `applyTransform`:

    ```json
    "source": {
        "field": "record_set/name",
        "applyTransform": {
            "format": "yyyy-MM-dd HH:mm:ss.S",
            "regex": "([^\\/]*)\\.jpg",
            "separator": "|"
        }
    }
    ```
    """

    extraction: DataExtraction = dataclasses.field(default_factory=DataExtraction)
    transforms: tuple[Transform, ...] = ()
    uid: str | None = None

    @classmethod
    def from_json_ld(cls, issues: Issues, json_ld: Any) -> Source:
        """Creates a new source from a JSON-LD `field` and populates issues."""
        if isinstance(json_ld, list):
            if len(json_ld) != 1:
                raise ValueError(f"Field {json_ld} should have one element.")
            return Source.from_json_ld(issues, json_ld[0])
        elif isinstance(json_ld, (dict, collections.defaultdict)):
            try:
                transforms = json_ld.get(constants.CROISSANT_APPLY_TRANSFORM, [])
                if not isinstance(transforms, list):
                    raise ValueError(
                        'Field "apply_transform" should be parsed as a list.'
                    )
                transforms = tuple(
                    Transform(
                        format=transform.get(constants.CROISSANT_FORMAT),
                        regex=transform.get(constants.CROISSANT_REGEX),
                        replace=transform.get(constants.CROISSANT_REPLACE),
                        separator=transform.get(constants.CROISSANT_SEPARATOR),
                    )
                    for transform in transforms
                )
                # Safely access and check "data_extraction" from JSON-LD.
                data_extraction = json_ld.get(constants.CROISSANT_DATA_EXTRACTION, {})
                if isinstance(data_extraction, list) and data_extraction:
                    data_extraction = data_extraction[0]
                if len(data_extraction) > 1:
                    issues.add_error(
                        f"{constants.ML_COMMONS_DATA_EXTRACTION} should have one of the"
                        f" following properties: {constants.ML_COMMONS_FORMAT},"
                        f" {constants.ML_COMMONS_REGEX},"
                        f" {constants.CROISSANT_REPLACE} or"
                        f" {constants.ML_COMMONS_SEPARATOR}"
                    )
                # Safely access and check "uid" from JSON-LD.
                distribution = json_ld.get(constants.CROISSANT_DISTRIBUTION)
                field = json_ld.get(constants.CROISSANT_FIELD)
                if distribution is not None and field is None:
                    uid = distribution
                elif distribution is None and field is not None:
                    uid = field
                else:
                    uid = None
                    issues.add_error(
                        f"Every {constants.ML_COMMONS_SOURCE} should declare either"
                        f" {constants.ML_COMMONS_FIELD} or"
                        f" {constants.SCHEMA_ORG_DISTRIBUTION}"
                    )
                # Safely access and check "file_property" from JSON-LD.
                file_property = data_extraction.get(constants.CROISSANT_FILE_PROPERTY)
                if is_file_property(file_property):
                    file_property = FileProperty[file_property]
                elif file_property is not None:
                    issues.add_error(
                        f"Property {constants.ML_COMMONS_FILE_PROPERTY} can only have"
                        " values in `fullpath`, `filepath` and `content`. Got:"
                        f" {file_property}"
                    )
                # Build the source.
                extraction = DataExtraction(
                    csv_column=data_extraction.get(constants.CROISSANT_CSV_COLUMN),
                    file_property=file_property,
                    json_path=data_extraction.get(constants.CROISSANT_JSON_PATH),
                )
                return Source(
                    extraction=extraction,
                    transforms=transforms,
                    uid=uid,
                )
            except TypeError as exception:
                issues.add_error(
                    f"Malformed `source`: {json_ld}. Got exception: {exception}"
                )
                return Source()
        else:
            issues.add_error(f"`source` has wrong type: {type(json_ld)} ({json_ld})")
            return Source()

    def __bool__(self):
        """Allows to write `if not node.source` / `if node.source`."""
        return self.uid is not None

    def get_field(self) -> str:
        if self.uid is None:
            # This case already rose an issue and should not happen at run time.
            raise ""
        if self.extraction.csv_column:
            return self.extraction.csv_column
        elif self.extraction.file_property:
            return self.extraction.file_property.name
        elif self.extraction.json_path:
            return self.extraction.json_path
        else:
            return self.uid.split("/")[-1]


def _apply_transform_fn(value: str, transform: Transform) -> str:
    """Applies one transform to `value`."""
    if transform.regex is not None:
        source_regex = re.compile(transform.regex)
        match = source_regex.match(value)
        if match is None:
            logging.debug(f"Could not match {source_regex} in {value}")
            return value
        for group in match.groups():
            if group is not None:
                return group
    return value


def apply_transforms_fn(value: str, source: Source | None = None) -> str:
    """Applies all transforms in `source` to `value`."""
    if source is None:
        return value
    transforms = source.transforms
    for transform in transforms:
        value = _apply_transform_fn(value, transform)
    return value
