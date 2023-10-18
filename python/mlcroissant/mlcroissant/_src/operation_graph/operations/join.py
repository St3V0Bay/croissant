"""Join operation module."""

import dataclasses

import pandas as pd

from mlcroissant._src.operation_graph.base_operation import Operation
from mlcroissant._src.structure_graph.nodes.record_set import RecordSet
from mlcroissant._src.structure_graph.nodes.source import apply_transforms_fn
from mlcroissant._src.structure_graph.nodes.source import Source


@dataclasses.dataclass(frozen=True, repr=False)
class Join(Operation):
    """Joins pd.DataFrames."""

    node: RecordSet

    def __call__(self, *args: pd.Series) -> pd.Series:
        """See class' docstring."""
        if len(args) == 1:
            return args[0]
        predecessors: list[str] = [
            operation.node.uid for operation in self.operations.predecessors(self)
        ]
        if len(predecessors) != len(args):
            raise ValueError(f"Unsupported: Trying to join {len(args)} pd.DataFrames.")
        fields = self.node.fields
        # `joins` is the list of joins to execute (source1, df1) x (source2, df2).
        joins: list[tuple[tuple[Source, pd.DataFrame], tuple[Source, pd.DataFrame]]] = (
            []
        )
        for field in fields:
            left = field.source
            right = field.references
            if left is None or right is None:
                continue
            if left.uid is None or right.uid is None:
                continue
            left_index = predecessors.index(left.uid.split("/")[0])
            right_index = predecessors.index(right.uid.split("/")[0])
            join = ((left, args[left_index]), (right, args[right_index]))
            if join not in joins:
                joins.append(join)
        for (left, df_left), (right, df_right) in joins:
            assert left is not None and left.uid is not None, (
                f'Left reference for "{field.uid}" is None. It should be a valid'
                " reference."
            )
            assert right is not None and right.uid is not None, (
                f'Right reference for "{field.uid}" is None. It should be a valid'
                " reference."
            )
            left_key = left.get_field()
            right_key = right.get_field()
            assert left_key in df_left.columns, (
                f'Column "{left_key}" does not exist in node "{left.uid}".'
                f" Existing columns: {df_left.columns}"
            )
            assert right_key in df_right.columns, (
                f'Column "{right_key}" does not exist in node "{right.uid}".'
                f" Existing columns: {df_right.columns}"
            )
            df_left[left_key] = df_left[left_key].transform(
                apply_transforms_fn, source=left
            )
            return df_left.merge(
                df_right,
                left_on=left_key,
                right_on=right_key,
                how="left",
                suffixes=(None, "_right"),
            )
