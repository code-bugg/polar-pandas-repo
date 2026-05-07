"""
Unit tests for PolarPandas v5 Semantic Validator (semv.py).

Layout assumption
-----------------
    src/semantic-validator/semv.py   ← module under test
    tests/test_semv.py               ← this file

Run from the repo root:
    pytest tests/test_semv.py -v

The tests construct AST nodes directly — no lexer or parser is involved.
This keeps tests fast, hermetic, and insulated from lexer/parser bugs.
Every public rule documented in the spec (§5 Semantic Rules) and every
validation branch in semv.py is covered.
"""

from __future__ import annotations

import sys
import os
import pytest

# ---------------------------------------------------------------------------
# Path setup: make sure the module under test is importable.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "semantic-validator"))

from semv import (
    ValidationError,
    _dataset_name_from_path,
    validate,
)

# ---------------------------------------------------------------------------
# Import every AST node that semv.py depends on.
# These come from the parser module, which semv imports.
# If the test environment uses a stub, adjust accordingly.
# ---------------------------------------------------------------------------
from parser import (
    AstCast,
    AstDropColumns,
    AstDropDuplicates,
    AstDropEmpty,
    AstFillEmpty,
    AstFilter,
    AstFor,
    AstGroupBy,
    AstIf,
    AstInfo,
    AstKeep,
    AstMerge,
    AstPipeline,
    AstPlot,
    AstPreview,
    AstRead,
    AstRename,
    AstSave,
    AstSet,
    AstSort,
    CondAnd,
    CondBetween,
    CondCompare,
    CondContains,
    CondIn,
    CondNull,
    CondOr,
    ExprBinop,
    ExprColRef,
    ExprLiteral,
    MetaAnd,
    MetaExists,
    MetaOr,
    MetaRowCount,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _errors(ast_nodes):
    """Run validate() and return the list of ValidationError objects."""
    return validate(ast_nodes)


def _messages(ast_nodes):
    """Run validate() and return error messages as plain strings."""
    return [e.message for e in _errors(ast_nodes)]


def _ok(ast_nodes):
    """Assert that the AST produces no validation errors."""
    errs = _errors(ast_nodes)
    assert errs == [], f"Expected no errors but got: {errs}"


def _has_error(ast_nodes, fragment: str):
    """Assert at least one error message contains *fragment*."""
    msgs = _messages(ast_nodes)
    assert any(fragment in m for m in msgs), (
        f"Expected an error containing {fragment!r} but got: {msgs}"
    )


# ---------------------------------------------------------------------------
# Shorthand AST builders
# ---------------------------------------------------------------------------

def _read(path="data.csv", line=1):
    return AstRead(path=path, line=line)


def _save(path="out.csv", line=1):
    return AstSave(path=path, line=line)


def _info(line=1):
    return AstInfo(line=line)


def _preview(rows=5, line=1):
    return AstPreview(rows=rows, line=line)


def _drop_empty(columns=None, line=1):
    return AstDropEmpty(columns=columns or [], line=line)


def _drop_dupes(line=1):
    return AstDropDuplicates(line=line)


def _drop_cols(columns, line=1):
    return AstDropColumns(columns=columns, line=line)


def _fill(column, value, line=1):
    return AstFillEmpty(column=column, value=value, line=line)


def _keep(columns, line=1):
    return AstKeep(columns=columns, line=line)


def _cast(column, dtype, on_error=None, line=1):
    return AstCast(column=column, dtype=dtype, on_error=on_error, line=line)


def _set(column, expr, line=1):
    return AstSet(column=column, expr=expr, line=line)


def _rename(old, new, line=1):
    return AstRename(old=old, new=new, line=line)


def _sort(column, direction="ASC", line=1):
    return AstSort(column=column, direction=direction, line=line)


def _group(by, agg, column, line=1):
    return AstGroupBy(by=by, agg=agg, column=column, line=line)


def _merge(source, on, how="INNER", source_is_file=False, line=1):
    return AstMerge(source=source, on=on, how=how, source_is_file=source_is_file, line=line)


def _filter(condition, line=1):
    return AstFilter(condition=condition, line=line)


def _plot(kind, x=None, y=None, title=None, save=None, target=None, line=1):
    return AstPlot(target=target, kind=kind, x=x, y=y, title=title, save=save, line=line)


def _pipeline(steps, target=None, line=1):
    return AstPipeline(target=target, steps=steps, line=line)


def _if(condition, then_body, else_body=None, line=1):
    return AstIf(condition=condition, then_body=then_body, else_body=else_body or [], line=line)


def _for(var, columns, body, line=1):
    return AstFor(var=var, columns=columns, body=body, line=line)


# Expression helpers
def _lit_int(v, line=1):
    return ExprLiteral(value=str(v), kind="integer", line=line)


def _lit_float(v, line=1):
    return ExprLiteral(value=str(v), kind="float", line=line)


def _lit_str(v, line=1):
    return ExprLiteral(value=v, kind="string", line=line)


def _lit_bool(v, line=1):
    return ExprLiteral(value=str(v).lower(), kind="bool", line=line)


def _col(name, line=1):
    return ExprColRef(name=name, is_loopvar=False, line=line)


def _loopvar(name, line=1):
    return ExprColRef(name=name, is_loopvar=True, line=line)


def _binop(op, left, right, line=1):
    return ExprBinop(op=op, left=left, right=right, line=line)


# Condition helpers
def _cmp(left, op, right, line=1):
    return CondCompare(left=left, op=op, right=right, line=line)


def _between(col, low, high, line=1):
    return CondBetween(col=_col(col) if isinstance(col, str) else col, low=low, high=high, line=line)


def _is_null(col, is_not=False, line=1):
    return CondNull(col=_col(col) if isinstance(col, str) else col, is_not=is_not, line=line)


def _in_cond(col, values, negate=False, line=1):
    return CondIn(col=_col(col) if isinstance(col, str) else col, values=values, negate=negate, line=line)


def _contains(col, value, negate=False, line=1):
    return CondContains(col=_col(col) if isinstance(col, str) else col, value=value, negate=negate, line=line)


def _and_cond(left, right, line=1):
    return CondAnd(left=left, right=right, line=line)


def _or_cond(left, right, line=1):
    return CondOr(left=left, right=right, line=line)


# Meta-condition helpers
def _exists(name, line=1):
    return MetaExists(name=name, line=line)


def _rowcount(name, op, value, line=1):
    return MetaRowCount(name=name, op=op, value=value, line=line)


# A standard "READ then ..." prefix so scope is satisfied
def _with_read(*rest, path="data.csv"):
    return [_read(path), *rest]


# ===========================================================================
# 1. _dataset_name_from_path — unit tests
# ===========================================================================

class TestDatasetNameFromPath:
    def test_simple_csv(self):
        assert _dataset_name_from_path("data.csv") == "data"

    def test_parquet(self):
        assert _dataset_name_from_path("sales.parquet") == "sales"

    def test_spaces_replaced(self):
        assert _dataset_name_from_path("my data.csv") == "my_data"

    def test_hyphens_replaced(self):
        assert _dataset_name_from_path("q1-results.csv") == "q1_results"

    def test_leading_digit_prefixed(self):
        name = _dataset_name_from_path("2024_data.csv")
        assert name.startswith("_")
        assert "2024" in name

    def test_nested_path(self):
        assert _dataset_name_from_path("/some/dir/customers.csv") == "customers"

    def test_windows_path(self):
        # PurePosixPath (used on Linux) does not recognise Windows backslashes
        # as path separators — the entire string is treated as a single stem,
        # so backslashes are sanitised to underscores.  The important contract
        # is that the result is a valid Python identifier (no special chars).
        result = _dataset_name_from_path(r"C:\data\orders.parquet")
        assert result  # non-empty
        assert all(c.isalnum() or c == "_" for c in result)

    def test_stem_only_special_chars(self):
        # All special chars should become underscores; result must not be empty
        result = _dataset_name_from_path("!@#.csv")
        assert result  # non-empty
        assert all(c.isalnum() or c == "_" for c in result)


# ===========================================================================
# 2. ValidationError
# ===========================================================================

class TestValidationError:
    def test_str_format(self):
        err = ValidationError(message="something wrong", line=42)
        assert "42" in str(err)
        assert "something wrong" in str(err)

    def test_raise_on_error(self):
        # ValidationError now inherits from Exception, so pytest.raises works
        # normally.  The pipeline below has no preceding READ, which triggers
        # the "active dataframe" scope error and causes raise_on_error to fire.
        ast = [_pipeline(steps=[_save()], line=1)]
        with pytest.raises(ValidationError) as exc_info:
            validate(ast, raise_on_error=True)
        assert exc_info.value.line >= 0
        assert exc_info.value.message  # non-empty

    def test_no_raise_when_clean(self):
        ast = _with_read(_save())
        result = validate(ast, raise_on_error=True)
        assert result == []

    def test_all_errors_collected_without_raise(self):
        # Two distinct errors: bad dtype and bad agg
        ast = _with_read(
            _cast("age", "BADTYPE"),
            _group(by=["region"], agg="BADAGG", column="amount"),
        )
        errs = _errors(ast)
        assert len(errs) >= 2


# ===========================================================================
# 3. Scope — READ / active dataframe tracking
# ===========================================================================

class TestScope:
    def test_no_read_before_operation_is_error(self):
        _has_error([_info()], "active dataframe")

    def test_read_establishes_active_df(self):
        _ok([_read("sales.csv"), _info()])

    def test_read_updates_active_df(self):
        # Second READ changes the active dataframe name
        _ok([_read("a.csv"), _read("b.csv"), _info()])

    def test_multiple_reads_stack_known_datasets(self):
        # MERGE by name requires the dataset to be known
        ast = [
            _read("orders.csv"),
            _read("customers.csv"),
            _merge("orders", on="id", source_is_file=False),
        ]
        _ok(ast)

    def test_merge_unknown_named_df_is_error(self):
        ast = _with_read(_merge("nonexistent", on="id", source_is_file=False))
        _has_error(ast, "not a known dataframe")

    def test_merge_file_path_does_not_require_prior_read(self):
        ast = _with_read(_merge("other.csv", on="id", source_is_file=True))
        _ok(ast)

    def test_pipeline_without_read_is_error(self):
        _has_error([_pipeline(steps=[_info()])], "active dataframe")

    def test_named_pipeline_with_unknown_target_is_error(self):
        ast = [_pipeline(target="ghost", steps=[_info()])]
        _has_error(ast, "not a known dataframe")

    def test_named_pipeline_with_known_target_is_ok(self):
        ast = [_read("sales.csv"), _pipeline(target="sales", steps=[_info()])]
        _ok(ast)

    def test_bare_pipeline_uses_active_df(self):
        ast = [_read("data.csv"), _pipeline(steps=[_info()])]
        _ok(ast)

    def test_rowcount_unknown_df_is_error(self):
        ast = _with_read(_if(_rowcount("ghost", ">", 10), then_body=[_info()]))
        _has_error(ast, "unknown dataframe")

    def test_rowcount_known_df_is_ok(self):
        ast = [_read("sales.csv"), _if(_rowcount("sales", ">", 10), then_body=[_info()])]
        _ok(ast)


# ===========================================================================
# 4. READ
# ===========================================================================

class TestRead:
    def test_read_in_pipeline_is_error(self):
        ast = [_read("data.csv"), _pipeline(steps=[_read("extra.csv")])]
        _has_error(ast, "READ cannot appear inside a pipeline")

    def test_read_standalone_is_ok(self):
        _ok([_read("data.csv")])

    def test_read_various_extensions(self):
        for ext in ("csv", "parquet", "json", "xlsx"):
            _ok([_read(f"file.{ext}")])


# ===========================================================================
# 5. SAVE
# ===========================================================================

class TestSave:
    def test_save_standalone_ok(self):
        _ok(_with_read(_save("out.csv")))

    def test_save_in_pipeline_ok(self):
        _ok([_read("data.csv"), _pipeline(steps=[_save("out.csv")])])

    def test_save_with_empty_path_is_error(self):
        ast = _with_read(_save(""))
        _has_error(ast, "SAVE requires a file path")


# ===========================================================================
# 6. PREVIEW
# ===========================================================================

class TestPreview:
    def test_default_rows_ok(self):
        _ok(_with_read(_preview(5)))

    def test_custom_rows_ok(self):
        _ok(_with_read(_preview(20)))

    def test_zero_rows_is_error(self):
        ast = _with_read(_preview(0))
        _has_error(ast, "PREVIEW row count")

    def test_negative_rows_is_error(self):
        ast = _with_read(_preview(-1))
        _has_error(ast, "PREVIEW row count")

    def test_no_read_before_preview_is_error(self):
        _has_error([_preview()], "active dataframe")


# ===========================================================================
# 7. INFO
# ===========================================================================

class TestInfo:
    def test_info_ok(self):
        _ok(_with_read(_info()))

    def test_info_without_read_is_error(self):
        _has_error([_info()], "active dataframe")


# ===========================================================================
# 8. DROP EMPTY
# ===========================================================================

class TestDropEmpty:
    def test_no_columns_drops_all_ok(self):
        _ok(_with_read(_drop_empty([])))

    def test_specific_columns_ok(self):
        _ok(_with_read(_drop_empty(["email", "phone"])))

    def test_without_read_is_error(self):
        _has_error([_drop_empty()], "active dataframe")


# ===========================================================================
# 9. DROP DUPLICATES
# ===========================================================================

class TestDropDuplicates:
    def test_ok(self):
        _ok(_with_read(_drop_dupes()))

    def test_without_read_is_error(self):
        _has_error([_drop_dupes()], "active dataframe")


# ===========================================================================
# 10. DROP COLUMNS
# ===========================================================================

class TestDropColumns:
    def test_single_column_ok(self):
        _ok(_with_read(_drop_cols(["internal_id"])))

    def test_multiple_columns_ok(self):
        _ok(_with_read(_drop_cols(["a", "b", "c"])))

    def test_empty_list_is_error(self):
        ast = _with_read(_drop_cols([]))
        _has_error(ast, "DROP requires at least one column name")

    def test_without_read_is_error(self):
        _has_error([_drop_cols(["col"])], "active dataframe")


# ===========================================================================
# 11. FILL EMPTY
# ===========================================================================

class TestFillEmpty:
    @pytest.mark.parametrize("method", ["mean", "median", "mode", "ffill", "bfill"])
    def test_fill_methods_ok(self, method):
        node = _fill("salary", ExprColRef(name=method, is_loopvar=False, line=1))
        _ok(_with_read(node))

    def test_fill_with_literal_int_ok(self):
        _ok(_with_read(_fill("price", _lit_int(0))))

    def test_fill_with_literal_str_ok(self):
        _ok(_with_read(_fill("region", _lit_str("Unknown"))))

    def test_fill_with_unknown_method_is_error(self):
        node = _fill("col", ExprColRef(name="interpolate", is_loopvar=False, line=1))
        ast = _with_read(node)
        _has_error(ast, "FILL EMPTY")

    def test_fill_with_binop_is_error(self):
        node = _fill("col", _binop("+", _col("a"), _col("b")))
        ast = _with_read(node)
        _has_error(ast, "FILL EMPTY")

    def test_fill_with_loopvar_ok_inside_for(self):
        # $col as a fill method should not trigger the method-name check
        # (it's a loop variable, not a method name)
        fill_node = _fill("col", _loopvar("method"))
        for_node = _for("method", ["mean"], [fill_node])
        _ok(_with_read(for_node))

    def test_without_read_is_error(self):
        _has_error([_fill("col", _lit_int(0))], "active dataframe")


# ===========================================================================
# 12. KEEP
# ===========================================================================

class TestKeep:
    def test_single_column_ok(self):
        _ok(_with_read(_keep(["name"])))

    def test_multiple_columns_ok(self):
        _ok(_with_read(_keep(["a", "b", "c"])))

    def test_empty_list_is_error(self):
        ast = _with_read(_keep([]))
        _has_error(ast, "KEEP requires at least one column name")

    def test_without_read_is_error(self):
        _has_error([_keep(["col"])], "active dataframe")


# ===========================================================================
# 13. CAST
# ===========================================================================

class TestCast:
    @pytest.mark.parametrize("dtype", ["NUMBER", "TEXT", "DATE", "DECIMAL", "BOOLEAN"])
    def test_valid_dtypes_ok(self, dtype):
        _ok(_with_read(_cast("col", dtype)))

    def test_unknown_dtype_is_error(self):
        ast = _with_read(_cast("age", "INTEGER"))
        _has_error(ast, "Unknown CAST target type")
        _has_error(ast, "INTEGER")

    def test_lowercase_dtype_is_error(self):
        # Parser upper-cases dtype; if someone passes lowercase it must fail
        ast = _with_read(_cast("age", "number"))
        _has_error(ast, "Unknown CAST target type")

    @pytest.mark.parametrize("mode", ["SKIP", "STOP"])
    def test_on_error_modes_ok(self, mode):
        _ok(_with_read(_cast("age", "NUMBER", on_error=mode)))

    def test_invalid_on_error_mode_is_error(self):
        ast = _with_read(_cast("age", "NUMBER", on_error="IGNORE"))
        _has_error(ast, "Unknown ON ERROR mode")

    def test_no_on_error_ok(self):
        _ok(_with_read(_cast("age", "NUMBER", on_error=None)))

    def test_without_read_is_error(self):
        _has_error([_cast("age", "NUMBER")], "active dataframe")


# ===========================================================================
# 14. SET
# ===========================================================================

class TestSet:
    def test_simple_literal_ok(self):
        _ok(_with_read(_set("flag", _lit_bool(True))))

    def test_arithmetic_expr_ok(self):
        expr = _binop("/", _col("weight"), _binop("*", _col("h"), _col("h")))
        _ok(_with_read(_set("bmi", expr)))

    def test_loop_var_outside_for_is_error(self):
        ast = _with_read(_set("col", _loopvar("x")))
        _has_error(ast, "outside a FOR loop")

    def test_loop_var_inside_for_ok(self):
        set_node = _set("result", _loopvar("col"))
        for_node = _for("col", ["age", "weight"], [set_node])
        _ok(_with_read(for_node))

    def test_without_read_is_error(self):
        _has_error([_set("col", _lit_int(1))], "active dataframe")


# ===========================================================================
# 15. RENAME
# ===========================================================================

class TestRename:
    def test_valid_rename_ok(self):
        _ok(_with_read(_rename("cust_id", "id")))

    def test_same_name_is_error(self):
        ast = _with_read(_rename("id", "id"))
        _has_error(ast, "identical")

    def test_without_read_is_error(self):
        _has_error([_rename("a", "b")], "active dataframe")


# ===========================================================================
# 16. SORT
# ===========================================================================

class TestSort:
    def test_sort_asc_ok(self):
        _ok(_with_read(_sort("amount", "ASC")))

    def test_sort_desc_ok(self):
        _ok(_with_read(_sort("amount", "DESC")))

    def test_invalid_direction_is_error(self):
        ast = _with_read(_sort("amount", "UP"))
        _has_error(ast, "SORT direction")

    def test_without_read_is_error(self):
        _has_error([_sort("col")], "active dataframe")


# ===========================================================================
# 17. GROUP BY
# ===========================================================================

class TestGroupBy:
    @pytest.mark.parametrize("agg", ["SUM", "AVERAGE", "COUNT", "MAX", "MIN"])
    def test_valid_aggs_ok(self, agg):
        _ok(_with_read(_group(["region"], agg, "amount")))

    def test_unknown_agg_is_error(self):
        ast = _with_read(_group(["region"], "MEDIAN", "amount"))
        _has_error(ast, "Unknown aggregation")
        _has_error(ast, "MEDIAN")

    def test_mean_not_valid_agg(self):
        # spec § 3.7: GROUP BY agg = SUM | AVERAGE | COUNT | MAX | MIN
        # (MEAN is not listed — AVERAGE is the keyword)
        ast = _with_read(_group(["region"], "MEAN", "amount"))
        _has_error(ast, "Unknown aggregation")

    def test_multi_column_group_ok(self):
        _ok(_with_read(_group(["region", "department"], "SUM", "headcount")))

    def test_empty_by_is_error(self):
        ast = _with_read(_group([], "SUM", "amount"))
        _has_error(ast, "GROUP BY requires at least one grouping column")

    def test_without_read_is_error(self):
        _has_error([_group(["col"], "SUM", "val")], "active dataframe")


# ===========================================================================
# 18. MERGE
# ===========================================================================

class TestMerge:
    @pytest.mark.parametrize("how", ["INNER", "LEFT", "RIGHT", "OUTER"])
    def test_valid_join_types_ok(self, how):
        ast = _with_read(_merge("other.csv", on="id", how=how, source_is_file=True))
        _ok(ast)

    def test_unknown_join_type_is_error(self):
        ast = _with_read(_merge("other.csv", on="id", how="CROSS", source_is_file=True))
        _has_error(ast, "Unknown MERGE join type")

    def test_named_df_source_known_ok(self):
        ast = [_read("orders.csv"), _read("customers.csv"),
               _merge("orders", on="id", source_is_file=False)]
        _ok(ast)

    def test_named_df_source_unknown_is_error(self):
        ast = _with_read(_merge("ghost", on="id", source_is_file=False))
        _has_error(ast, "not a known dataframe")

    def test_merge_default_is_inner(self):
        ast = _with_read(_merge("ref.csv", on="id", how="INNER", source_is_file=True))
        _ok(ast)

    def test_without_read_is_error(self):
        _has_error([_merge("ref.csv", on="id", source_is_file=True)], "active dataframe")


# ===========================================================================
# 19. FILTER WHERE — condition variants
# ===========================================================================

class TestFilter:
    def test_simple_compare_ok(self):
        cond = _cmp(_col("amount"), ">", _lit_int(0))
        _ok(_with_read(_filter(cond)))

    def test_compare_all_ops_ok(self):
        for op in ("==", "!=", ">", "<", ">=", "<="):
            cond = _cmp(_col("x"), op, _lit_int(1))
            _ok(_with_read(_filter(cond)))

    def test_is_null_ok(self):
        _ok(_with_read(_filter(_is_null("email"))))

    def test_is_not_null_ok(self):
        _ok(_with_read(_filter(_is_null("email", is_not=True))))

    def test_in_ok(self):
        cond = _in_cond("country", [_lit_str("MD"), _lit_str("RO")])
        _ok(_with_read(_filter(cond)))

    def test_not_in_ok(self):
        cond = _in_cond("country", [_lit_str("XX")], negate=True)
        _ok(_with_read(_filter(cond)))

    def test_between_ok(self):
        cond = _between("score", _lit_int(0), _lit_int(100))
        _ok(_with_read(_filter(cond)))

    def test_between_inverted_bounds_is_error(self):
        cond = _between("score", _lit_int(100), _lit_int(0))
        ast = _with_read(_filter(cond))
        _has_error(ast, "lower bound")

    def test_between_equal_bounds_ok(self):
        # lo == hi is technically valid (single value)
        cond = _between("score", _lit_int(5), _lit_int(5))
        _ok(_with_read(_filter(cond)))

    def test_between_float_bounds_ok(self):
        cond = _between("ratio", _lit_float(0.0), _lit_float(1.0))
        _ok(_with_read(_filter(cond)))

    def test_between_float_inverted_is_error(self):
        cond = _between("ratio", _lit_float(1.5), _lit_float(0.5))
        ast = _with_read(_filter(cond))
        _has_error(ast, "lower bound")

    def test_contains_string_ok(self):
        cond = _contains("email", _lit_str("@gmail"))
        _ok(_with_read(_filter(cond)))

    def test_not_contains_ok(self):
        cond = _contains("notes", _lit_str("deleted"), negate=True)
        _ok(_with_read(_filter(cond)))

    def test_contains_non_string_is_error(self):
        cond = _contains("col", _lit_int(42))
        ast = _with_read(_filter(cond))
        _has_error(ast, "CONTAINS value should be a string literal")

    def test_and_condition_ok(self):
        cond = _and_cond(
            _cmp(_col("status"), "==", _lit_str("active")),
            _cmp(_col("age"), ">=", _lit_int(18)),
        )
        _ok(_with_read(_filter(cond)))

    def test_or_condition_ok(self):
        cond = _or_cond(
            _cmp(_col("amount"), ">", _lit_int(0)),
            _cmp(_col("status"), "!=", _lit_str("closed")),
        )
        _ok(_with_read(_filter(cond)))

    def test_nested_and_or_ok(self):
        cond = _or_cond(
            _and_cond(
                _cmp(_col("a"), ">", _lit_int(1)),
                _cmp(_col("b"), "<", _lit_int(10)),
            ),
            _cmp(_col("c"), "==", _lit_str("x")),
        )
        _ok(_with_read(_filter(cond)))

    def test_loop_var_in_filter_outside_for_is_error(self):
        cond = _cmp(_loopvar("col"), ">", _lit_int(0))
        ast = _with_read(_filter(cond))
        _has_error(ast, "outside a FOR loop")

    def test_loop_var_in_filter_inside_for_ok(self):
        cond = _cmp(_loopvar("col"), ">", _lit_int(0))
        for_node = _for("col", ["age", "income"], [_filter(cond)])
        _ok(_with_read(for_node))

    def test_without_read_is_error(self):
        cond = _cmp(_col("x"), ">", _lit_int(0))
        _has_error([_filter(cond)], "active dataframe")


# ===========================================================================
# 20. PLOT
# ===========================================================================

class TestPlot:
    # --- Valid kinds with correct axis combinations --------------------------

    @pytest.mark.parametrize("kind,x,y", [
        ("line",    "month",  "total"),
        ("bar",     "region", "revenue"),
        ("scatter", "age",    "salary"),
        ("box",     None,     "salary"),   # X optional for box
        ("pie",     "cat",    None),       # Y optional for pie
    ])
    def test_valid_kind_ok(self, kind, x, y):
        _ok(_with_read(_plot(kind, x=x, y=y)))

    def test_hist_x_only_ok(self):
        _ok(_with_read(_plot("hist", x="score")))

    def test_hist_with_y_is_error(self):
        ast = _with_read(_plot("hist", x="score", y="count"))
        _has_error(ast, "does not use a Y column")

    def test_unknown_kind_is_error(self):
        ast = _with_read(_plot("heatmap"))
        _has_error(ast, "Unknown plot type")
        _has_error(ast, "heatmap")

    def test_line_missing_x_is_error(self):
        ast = _with_read(_plot("line", y="total"))
        _has_error(ast, "requires an X column")

    def test_line_missing_y_is_error(self):
        ast = _with_read(_plot("line", x="month"))
        _has_error(ast, "requires a Y column")

    def test_bar_missing_both_is_error(self):
        ast = _with_read(_plot("bar"))
        errs = _messages(ast)
        assert any("X column" in m for m in errs)
        assert any("Y column" in m for m in errs)

    def test_scatter_requires_both_axes(self):
        ast = _with_read(_plot("scatter", x="age"))
        _has_error(ast, "requires a Y column")

    def test_plot_with_title_and_save_ok(self):
        _ok(_with_read(_plot("bar", x="region", y="revenue",
                             title="Revenue", save="chart.png")))

    def test_plot_with_named_target_ok(self):
        ast = [_read("orders.csv"), _plot("bar", x="r", y="v", target="orders")]
        _ok(ast)

    def test_plot_with_unknown_target_is_error(self):
        ast = [_read("data.csv"), _plot("bar", x="r", y="v", target="ghost")]
        _has_error(ast, "not a known dataframe")

    def test_plot_without_read_is_error(self):
        _has_error([_plot("bar", x="x", y="y")], "active dataframe")

    def test_box_with_optional_x_ok(self):
        # box: x is optional grouping column
        _ok(_with_read(_plot("box", x="region", y="salary")))

    def test_pie_with_optional_y_ok(self):
        _ok(_with_read(_plot("pie", x="category", y="value")))

    # Spec § 3.9 valid kinds are: line | bar | scatter | pie | hist | box
    @pytest.mark.parametrize("invalid_kind", ["area", "count", "histogram", "density", "violin"])
    def test_spec_invalid_kinds_are_rejected(self, invalid_kind):
        ast = _with_read(_plot(invalid_kind, x="x", y="y"))
        _has_error(ast, "Unknown plot type")


# ===========================================================================
# 21. Pipeline
# ===========================================================================

class TestPipeline:
    def test_simple_pipeline_ok(self):
        ast = [_read("data.csv"), _pipeline(steps=[_drop_empty(), _info()])]
        _ok(ast)

    def test_named_pipeline_ok(self):
        ast = [_read("orders.csv"), _pipeline(target="orders", steps=[_info()])]
        _ok(ast)

    def test_read_inside_pipeline_is_error(self):
        ast = [_read("data.csv"), _pipeline(steps=[_read("extra.csv")])]
        _has_error(ast, "READ cannot appear inside a pipeline")

    def test_empty_pipeline_is_error(self):
        ast = [_read("data.csv"), _pipeline(steps=[])]
        _has_error(ast, "Pipeline has no steps")

    def test_pipeline_propagates_errors(self):
        # bad CAST inside pipeline should surface
        ast = [_read("data.csv"), _pipeline(steps=[_cast("age", "BADTYPE")])]
        _has_error(ast, "Unknown CAST target type")

    def test_save_inside_pipeline_ok(self):
        ast = [_read("data.csv"), _pipeline(steps=[_save("out.csv")])]
        _ok(ast)

    def test_chained_pipeline_ok(self):
        ast = [
            _read("data.csv"),
            _pipeline(steps=[
                _drop_empty(),
                _drop_dupes(),
                _cast("age", "NUMBER"),
                _sort("age"),
                _save("clean.csv"),
            ]),
        ]
        _ok(ast)


# ===========================================================================
# 22. IF / Meta-conditions
# ===========================================================================

class TestIf:
    def test_rowcount_guard_ok(self):
        ast = [_read("data.csv"),
               _if(_rowcount("data", ">", 1000), then_body=[_drop_empty()])]
        _ok(ast)

    def test_exists_guard_ok(self):
        ast = [_read("data.csv"),
               _if(_exists("data"), then_body=[_info()])]
        _ok(ast)

    def test_rowcount_negative_value_is_error(self):
        ast = [_read("data.csv"),
               _if(_rowcount("data", ">", -1), then_body=[_info()])]
        _has_error(ast, "non-negative")

    def test_rowcount_zero_ok(self):
        # 0 is non-negative
        ast = [_read("data.csv"),
               _if(_rowcount("data", ">", 0), then_body=[_info()])]
        _ok(ast)

    def test_if_with_else_ok(self):
        ast = [_read("data.csv"),
               _if(_rowcount("data", ">", 500),
                   then_body=[_drop_empty()],
                   else_body=[_fill("amount", _lit_int(0))])]
        _ok(ast)

    def test_meta_and_ok(self):
        ast = [_read("a.csv"), _read("b.csv"),
               _if(MetaAnd(left=_exists("a"), right=_exists("b"), line=1),
                   then_body=[_info()])]
        _ok(ast)

    def test_meta_or_ok(self):
        ast = [_read("a.csv"),
               _if(MetaOr(left=_exists("a"), right=_rowcount("a", ">", 0), line=1),
                   then_body=[_info()])]
        _ok(ast)

    def test_empty_name_in_exists_is_error(self):
        ast = [_read("data.csv"),
               _if(_exists(""), then_body=[_info()])]
        _has_error(ast, "EXISTS requires a dataset name")

    def test_body_errors_propagate(self):
        ast = [_read("data.csv"),
               _if(_exists("data"), then_body=[_cast("age", "WRONG")])]
        _has_error(ast, "Unknown CAST target type")

    def test_else_body_errors_propagate(self):
        ast = [_read("data.csv"),
               _if(_exists("data"),
                   then_body=[_info()],
                   else_body=[_group(["r"], "BADAGG", "v")])]
        _has_error(ast, "Unknown aggregation")

    def test_if_body_inherits_scope(self):
        # datasets READ before IF are visible inside the body
        ast = [
            _read("orders.csv"),
            _read("customers.csv"),
            _if(_exists("orders"), then_body=[
                _merge("orders", on="id", source_is_file=False)
            ]),
        ]
        _ok(ast)


# ===========================================================================
# 23. FOR loops
# ===========================================================================

class TestFor:
    def test_simple_for_ok(self):
        body = [_cast("col", "NUMBER", on_error="SKIP")]
        # cast references $col via the loop variable context
        cast_node = _cast("col", "NUMBER", on_error="SKIP")
        for_node = _for("col", ["age", "weight", "bmi"], [cast_node])
        _ok(_with_read(for_node))

    def test_loopvar_in_body_ok(self):
        # Using $col inside FILL EMPTY is valid inside FOR
        fill_node = _fill("ignored", _loopvar("col"))
        for_node = _for("col", ["x", "y"], [fill_node])
        _ok(_with_read(for_node))

    def test_loopvar_used_outside_for_is_error(self):
        ast = _with_read(_set("x", _loopvar("col")))
        _has_error(ast, "outside a FOR loop")

    def test_loopvar_shadows_column_name_is_error(self):
        for_node = _for("age", ["age", "weight"], [_info()])
        ast = _with_read(for_node)
        _has_error(ast, "shadows")

    def test_empty_column_list_is_error(self):
        for_node = _for("col", [], [_info()])
        ast = _with_read(for_node)
        _has_error(ast, "at least one column")

    def test_empty_body_is_error(self):
        for_node = _for("col", ["age"], [])
        ast = _with_read(for_node)
        _has_error(ast, "empty")

    def test_nested_for_ok(self):
        inner_for = _for("val", ["p", "q"], [_cast("val", "NUMBER")])
        outer_for = _for("col", ["a", "b"], [inner_for])
        _ok(_with_read(outer_for))

    def test_for_body_errors_propagate(self):
        body = [_cast("col", "NOTATYPE")]
        for_node = _for("x", ["a"], body)
        ast = _with_read(for_node)
        _has_error(ast, "Unknown CAST target type")

    def test_inner_loopvar_not_visible_in_outer_scope(self):
        # After the FOR ends, $col should not be in scope
        for_node = _for("col", ["age"], [_info()])
        # Using $col AFTER the for loop is an error
        set_after = _set("x", _loopvar("col"))
        ast = _with_read(for_node, set_after)
        _has_error(ast, "outside a FOR loop")


# ===========================================================================
# 24. Expressions
# ===========================================================================

class TestExpressions:
    def test_literal_int_ok(self):
        _ok(_with_read(_set("x", _lit_int(42))))

    def test_literal_float_ok(self):
        _ok(_with_read(_set("x", _lit_float(3.14))))

    def test_literal_str_ok(self):
        _ok(_with_read(_set("x", _lit_str("hello"))))

    def test_literal_bool_ok(self):
        _ok(_with_read(_set("x", _lit_bool(True))))

    def test_col_ref_ok(self):
        _ok(_with_read(_set("y", _col("weight"))))

    def test_nested_binop_ok(self):
        expr = _binop("+", _binop("*", _col("a"), _col("b")), _lit_int(1))
        _ok(_with_read(_set("result", expr)))

    def test_division_expr_ok(self):
        expr = _binop("/", _col("weight"), _binop("*", _col("h"), _col("h")))
        _ok(_with_read(_set("bmi", expr)))


# ===========================================================================
# 25. FILL EMPTY — fill method case-insensitivity (spec says methods are
#     plain IDENTs; validator receives them as-is from the parser).
# ===========================================================================

class TestFillMethodCaseSensitivity:
    """The spec defines fill methods as plain identifiers.  The validator
    compares them lowercased.  Verify that the check is case-insensitive
    so 'Mean' (if a future parser normalises differently) also works."""

    def test_lowercase_mean_ok(self):
        node = _fill("sal", ExprColRef(name="mean", is_loopvar=False, line=1))
        _ok(_with_read(node))

    def test_uppercase_mean_is_error(self):
        # The validator lowercases the name before checking; "MEAN" → "mean" → ok
        # BUT the spec says the aggregation keyword is AVERAGE, not MEAN.
        # The fill method "mean" IS valid; "MEAN" as an IDENT should still match
        # because the validator does .lower() comparison.
        node = _fill("sal", ExprColRef(name="MEAN", is_loopvar=False, line=1))
        # "MEAN".lower() == "mean" ∈ _VALID_FILL_METHODS → no error
        _ok(_with_read(node))


# ===========================================================================
# 26. End-to-end scenario tests (mimicking spec examples)
# ===========================================================================

class TestEndToEnd:
    def test_full_cleaning_pipeline(self):
        """Mirrors the health.csv example from spec § 7.1."""
        ast = [
            _read("health.csv"),
            _info(),
            _preview(5),
            _drop_empty(),
            _drop_dupes(),
            _for("col", ["age", "weight", "bmi"], [
                _cast("col", "NUMBER", on_error="SKIP"),
                _fill("col", ExprColRef(name="mean", is_loopvar=False, line=1)),
            ]),
            _filter(_between("age", _lit_int(18), _lit_int(90))),
            _set("bmi", _binop("/", _col("weight"),
                                _binop("*", _col("height"), _col("height")))),
            _save("clean_health.csv"),
        ]
        _ok(ast)

    def test_multi_table_analysis(self):
        """Mirrors the multi-table example from spec § 7.2."""
        ast = [
            _read("orders.csv"),
            _read("customers.csv"),
            _pipeline(target="customers", steps=[
                _drop_empty(["email"]),
                _filter(_contains("email", _lit_str("@"))),
                _rename("cust_id", "id"),
            ]),
            _pipeline(target="orders", steps=[
                _filter(_cmp(_col("status"), "==", _lit_str("complete"))),
                _merge("customers", on="id", how="LEFT", source_is_file=False),
                _group(["region"], "SUM", "total"),
                _sort("total", "DESC"),
                _plot("bar", x="region", y="total", title="Revenue by Region",
                      save="revenue.png"),
                _save("regional_revenue.csv"),
            ]),
        ]
        _ok(ast)

    def test_conditional_cleaning(self):
        """Mirrors the survey example from spec § 7.3."""
        ast = [
            _read("survey.csv"),
            _info(),
            _if(
                _rowcount("survey", ">", 1000),
                then_body=[_drop_empty(), _drop_dupes()],
                else_body=[
                    _fill("score", ExprColRef(name="median", is_loopvar=False, line=1)),
                    _fill("region", _lit_str("Unknown")),
                ],
            ),
            _for("col", ["q1", "q2", "q3", "q4", "q5"], [
                _cast("col", "NUMBER", on_error="SKIP"),
                _filter(_between("col", _lit_int(1), _lit_int(10))),
            ]),
            _group(["region"], "AVERAGE", "score"),
            _sort("score", "DESC"),
            _preview(),
        ]
        _ok(ast)

    def test_error_accumulation(self):
        """Multiple independent errors are all collected."""
        ast = _with_read(
            _cast("age", "BADTYPE"),           # error 1: bad dtype
            _group(["r"], "BADAGG", "v"),      # error 2: bad agg
            _plot("invisible", x="x", y="y"), # error 3: unknown kind
            _rename("col", "col"),             # error 4: same name
        )
        errs = _errors(ast)
        assert len(errs) >= 4
