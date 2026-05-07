"""Unit tests for the PolarPandas v5 code generator (cd.py).

Tests construct ASTs by piping real source text through
``lexer.tokenize`` → ``parser.parse``, then call ``cd.generate`` and
assert on the resulting Python source.

A subset of tests additionally execute the generated code against a
small synthetic polars DataFrame to confirm runtime behaviour.

Run from the repo root:
    pytest tests/test_codegen.py -v
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Path setup — make the four modules importable.
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _SRC / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Eager imports
from lexer.lexer import tokenize  # noqa: E402
from parser.parser import parse  # noqa: E402

_semv = _load("semv", "semantic-validator/semv.py")
_cd = _load("cd", "code-generator/cd.py")

validate = _semv.validate
generate = _cd.generate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compile_pp(source: str) -> str:
    """Lex → parse → validate → generate.  Asserts no semantic errors."""
    tokens = tokenize(source)
    ast = parse(tokens)
    errors = validate(ast)
    assert errors == [], f"unexpected semantic errors: {errors}"
    return generate(ast)


def run_with_df(generated: str, df: pl.DataFrame) -> dict:
    """Execute generated Python after pre-binding ``df`` and the per-file alias.

    Strips the ``READ`` line so the test can supply its own dataframe.
    The dataset name in the test scripts is always ``data`` so the test
    binds both ``df`` and ``data`` to the input frame.
    """
    lines = [ln for ln in generated.splitlines() if "pl.read_" not in ln]
    code = "\n".join(lines)
    ns: dict = {"pl": pl, "df": df, "data": df}
    exec(compile(code, "<generated>", "exec"), ns)
    return ns


# ---------------------------------------------------------------------------
# Header / preamble
# ---------------------------------------------------------------------------

def test_header_always_imports_polars():
    out = compile_pp("READ 'data.csv'")
    assert "import polars as pl" in out
    assert "import pandas" not in out  # no PLOT in this script


def test_plot_triggers_pandas_and_matplotlib_imports():
    out = compile_pp(
        "READ 'data.csv'\n"
        "PLOT TYPE bar X region Y total"
    )
    assert "import pandas as pd" in out
    assert "import matplotlib.pyplot as plt" in out


# ---------------------------------------------------------------------------
# READ / SAVE
# ---------------------------------------------------------------------------

def test_read_csv_emits_pl_read_csv():
    out = compile_pp("READ 'sales.csv'")
    assert "pl.read_csv('sales.csv')" in out
    # Variable name is derived from the file stem
    assert "sales = pl.read_csv" in out
    assert "df = sales" in out


def test_read_dispatches_by_extension():
    for ext, fn in [
        (".csv", "pl.read_csv"),
        (".parquet", "pl.read_parquet"),
        (".json", "pl.read_json"),
        (".xlsx", "pl.read_excel"),
    ]:
        out = compile_pp(f"READ 'data{ext}'")
        assert fn in out, f"{ext} should dispatch to {fn}"


def test_save_dispatches_by_extension():
    out = compile_pp("READ 'a.csv'\nSAVE 'b.parquet'")
    assert "df.write_parquet('b.parquet')" in out


# ---------------------------------------------------------------------------
# Cleaning operations
# ---------------------------------------------------------------------------

def test_drop_empty_no_subset():
    out = compile_pp("READ 'a.csv'\nDROP EMPTY")
    assert "df.drop_nulls()" in out


def test_drop_empty_with_subset():
    out = compile_pp("READ 'a.csv'\nDROP EMPTY age, score")
    assert "df.drop_nulls(subset=['age', 'score'])" in out


def test_drop_duplicates():
    out = compile_pp("READ 'a.csv'\nDROP DUPLICATES")
    assert "df.unique(keep='first')" in out


def test_drop_columns():
    out = compile_pp("READ 'a.csv'\nDROP age, score")
    assert "df.drop(['age', 'score'])" in out


def test_keep_columns():
    out = compile_pp("READ 'a.csv'\nKEEP id, total")
    assert "df.select(['id', 'total'])" in out


def test_rename():
    out = compile_pp("READ 'a.csv'\nRENAME old TO new")
    assert "df.rename({'old': 'new'})" in out


# ---------------------------------------------------------------------------
# FILL EMPTY
# ---------------------------------------------------------------------------

def test_fill_empty_with_method_mean():
    out = compile_pp("READ 'a.csv'\nFILL EMPTY age WITH mean")
    assert "fill_null(pl.col('age').mean())" in out


def test_fill_empty_with_method_median():
    out = compile_pp("READ 'a.csv'\nFILL EMPTY age WITH median")
    assert "fill_null(pl.col('age').median())" in out


def test_fill_empty_with_ffill_uses_strategy():
    out = compile_pp("READ 'a.csv'\nFILL EMPTY age WITH ffill")
    assert "fill_null(strategy='forward')" in out


def test_fill_empty_with_literal():
    out = compile_pp("READ 'a.csv'\nFILL EMPTY region WITH 'Unknown'")
    assert "fill_null('Unknown')" in out


# ---------------------------------------------------------------------------
# CAST / SET / SORT / GROUP BY / FILTER
# ---------------------------------------------------------------------------

def test_cast_to_number_strict_true():
    out = compile_pp("READ 'a.csv'\nCAST age TO NUMBER")
    assert "cast(pl.Float64, strict=True)" in out


def test_cast_to_number_with_on_error_skip():
    out = compile_pp("READ 'a.csv'\nCAST age TO NUMBER (ON ERROR SKIP)")
    assert "cast(pl.Float64, strict=False)" in out


def test_set_with_arithmetic():
    out = compile_pp("READ 'a.csv'\nSET bmi = weight / height")
    assert "(pl.col('weight') / pl.col('height'))" in out
    assert ".alias('bmi')" in out


def test_sort_default_asc():
    out = compile_pp("READ 'a.csv'\nSORT BY total")
    assert "df.sort('total', descending=False)" in out


def test_sort_desc():
    out = compile_pp("READ 'a.csv'\nSORT BY total DESC")
    assert "df.sort('total', descending=True)" in out


def test_group_by_sum():
    out = compile_pp("READ 'a.csv'\nGROUP BY region SUM total")
    assert "group_by(['region'])" in out
    assert "pl.col('total').sum().alias('total')" in out


def test_group_by_average():
    out = compile_pp("READ 'a.csv'\nGROUP BY region AVERAGE score")
    assert "pl.col('score').mean().alias('score')" in out


def test_filter_simple_comparison():
    out = compile_pp("READ 'a.csv'\nFILTER WHERE age > 18")
    assert "df.filter((pl.col('age') > 18))" in out


def test_filter_between():
    out = compile_pp("READ 'a.csv'\nFILTER WHERE age BETWEEN 18 AND 90")
    assert "is_between(18, 90)" in out


def test_filter_contains():
    out = compile_pp("READ 'a.csv'\nFILTER WHERE name CONTAINS 'foo'")
    assert ".str.contains('foo', literal=True)" in out


def test_filter_is_empty():
    out = compile_pp("READ 'a.csv'\nFILTER WHERE name IS EMPTY")
    assert ".is_null()" in out


def test_filter_and_or():
    out = compile_pp("READ 'a.csv'\nFILTER WHERE age > 18 AND score < 5 OR name CONTAINS 'x'")
    assert " & " in out and " | " in out


# ---------------------------------------------------------------------------
# MERGE
# ---------------------------------------------------------------------------

def test_merge_with_known_dataframe():
    out = compile_pp("READ 'orders.csv'\nREAD 'customers.csv'\nMERGE customers ON id LEFT")
    assert "df.join(customers, on='id', how='left')" in out


# ---------------------------------------------------------------------------
# Pipelines, FOR, IF
# ---------------------------------------------------------------------------

def test_pipeline_chains_and_rebinds():
    out = compile_pp(
        "READ 'orders.csv'\n"
        "orders -> FILTER WHERE total > 0 -> SORT BY total DESC"
    )
    assert "df = orders" in out
    assert "df.filter" in out
    assert "df.sort" in out
    # Pipeline tail re-binds the named dataframe
    assert "orders = df" in out


def test_for_loop_is_unrolled_at_compile_time():
    out = compile_pp(
        "READ 'a.csv'\n"
        "FOR $col IN age, weight :\n"
        "    CAST $col TO NUMBER\n"
        "END"
    )
    # No Python `for` in the output
    assert "for " not in out
    # Both iterations materialised
    assert "pl.col('age').cast(pl.Float64" in out
    assert "pl.col('weight').cast(pl.Float64" in out


def test_if_rowcount_lowers_to_python_if():
    out = compile_pp(
        "READ 'survey.csv'\n"
        "IF ROWCOUNT survey > 1000 :\n"
        "    DROP EMPTY\n"
        "ELSE\n"
        "    DROP DUPLICATES\n"
        "END"
    )
    assert "if (survey.height > 1000):" in out
    assert "else:" in out


# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def test_plot_bar_with_title_and_save():
    out = compile_pp(
        "READ 'a.csv'\n"
        "PLOT TYPE bar X region Y total TITLE 'rev' SAVE 'fig.png'"
    )
    assert "_pdf = df.to_pandas()" in out
    assert "_ax.bar(_pdf['region'], _pdf['total'])" in out
    assert "_ax.set_title('rev')" in out
    assert "_fig.savefig('fig.png')" in out


def test_plot_hist_uses_x_only():
    out = compile_pp("READ 'a.csv'\nPLOT TYPE hist X age")
    assert "_ax.hist(_pdf['age'])" in out


# ---------------------------------------------------------------------------
# End-to-end execution against a real polars frame
# ---------------------------------------------------------------------------

def test_end_to_end_filter_executes():
    src = "READ 'data.csv'\nFILTER WHERE age > 18\nSAVE '/tmp/_pp_out.csv'"
    code = compile_pp(src)
    df = pl.DataFrame({"age": [10, 20, 30], "name": ["a", "b", "c"]})
    # Strip READ + SAVE, just run the filter
    body = "\n".join(
        ln for ln in code.splitlines()
        if "pl.read_" not in ln and "write_csv" not in ln
    )
    ns = {"pl": pl, "df": df, "data": df}
    exec(compile(body, "<gen>", "exec"), ns)
    assert ns["df"].height == 2
    assert ns["df"]["age"].to_list() == [20, 30]


def test_end_to_end_for_unrolled_cast_executes():
    src = (
        "READ 'data.csv'\n"
        "FOR $col IN a, b :\n"
        "    CAST $col TO NUMBER (ON ERROR SKIP)\n"
        "END"
    )
    code = compile_pp(src)
    df = pl.DataFrame({"a": ["1", "2"], "b": ["3", "4"]})
    body = "\n".join(ln for ln in code.splitlines() if "pl.read_" not in ln)
    ns = {"pl": pl, "df": df, "data": df}
    exec(compile(body, "<gen>", "exec"), ns)
    assert ns["df"].schema["a"] == pl.Float64
    assert ns["df"].schema["b"] == pl.Float64


def test_end_to_end_group_by_sum_executes():
    src = "READ 'data.csv'\nGROUP BY region SUM total"
    code = compile_pp(src)
    df = pl.DataFrame(
        {"region": ["A", "A", "B"], "total": [1, 2, 5]}
    )
    body = "\n".join(ln for ln in code.splitlines() if "pl.read_" not in ln)
    ns = {"pl": pl, "df": df, "data": df}
    exec(compile(body, "<gen>", "exec"), ns)
    result = ns["df"].sort("region")
    assert result["region"].to_list() == ["A", "B"]
    assert sorted(result["total"].to_list()) == [3, 5]