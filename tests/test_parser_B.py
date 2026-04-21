import pytest

from lexer.lexer import tokenize
from parser.parser import (
    ParseError,
    AstRead,
    AstSave,
    AstPreview,
    AstInfo,
    AstDropEmpty,
    AstDropDuplicates,
    AstDropColumns,
    AstFillEmpty,
    AstKeep,
    AstCast,
    AstSet,
    AstRename,
    AstSort,
    AstGroupBy,
    AstMerge,
    parse,
)


class TestIO:
    def test_read(self):
        ast = parse(tokenize("READ 'sales.csv'"))
        assert ast == [AstRead(path="sales.csv", line=1)]

    def test_save(self):
        ast = parse(tokenize("SAVE 'out.parquet'"))
        assert ast == [AstSave(path="out.parquet", line=1)]

    def test_preview_default(self):
        ast = parse(tokenize("PREVIEW"))
        assert ast == [AstPreview(rows=5, line=1)]

    def test_preview_value(self):
        ast = parse(tokenize("PREVIEW 10"))
        assert ast == [AstPreview(rows=10, line=1)]

    def test_info(self):
        ast = parse(tokenize("INFO"))
        assert ast == [AstInfo(line=1)]


class TestCleaningTransform:
    def test_drop_empty_all(self):
        ast = parse(tokenize("DROP EMPTY"))
        assert ast == [AstDropEmpty(columns=[], line=1)]

    def test_drop_empty_columns(self):
        ast = parse(tokenize("DROP EMPTY revenue, cost"))
        assert ast == [AstDropEmpty(columns=["revenue", "cost"], line=1)]

    def test_drop_duplicates(self):
        ast = parse(tokenize("DROP DUPLICATES"))
        assert ast == [AstDropDuplicates(line=1)]

    def test_drop_columns(self):
        ast = parse(tokenize("DROP a, b"))
        assert ast == [AstDropColumns(columns=["a", "b"], line=1)]

    def test_fill_empty_literal(self):
        ast = parse(tokenize("FILL EMPTY salary WITH 0"))
        assert isinstance(ast[0], AstFillEmpty)
        assert ast[0].column == "salary"

    def test_keep(self):
        ast = parse(tokenize("KEEP name, age, region"))
        assert ast == [AstKeep(columns=["name", "age", "region"], line=1)]

    def test_cast_no_policy(self):
        ast = parse(tokenize("CAST age TO NUMBER"))
        assert ast == [AstCast(column="age", dtype="NUMBER", on_error=None, line=1)]

    def test_cast_with_policy(self):
        ast = parse(tokenize("CAST age TO NUMBER (ON ERROR SKIP)"))
        assert ast == [AstCast(column="age", dtype="NUMBER", on_error="SKIP", line=1)]

    def test_set(self):
        ast = parse(tokenize("SET margin = revenue - cost"))
        assert isinstance(ast[0], AstSet)
        assert ast[0].column == "margin"

    def test_rename(self):
        ast = parse(tokenize("RENAME cust_id TO id"))
        assert ast == [AstRename(old="cust_id", new="id", line=1)]

    def test_sort(self):
        ast = parse(tokenize("SORT BY revenue DESC"))
        assert ast == [AstSort(column="revenue", direction="DESC", line=1)]

    def test_group(self):
        ast = parse(tokenize("GROUP BY region SUM total"))
        assert ast == [AstGroupBy(by=["region"], agg="SUM", column="total", line=1)]

    def test_merge_ident(self):
        ast = parse(tokenize("MERGE customers ON id LEFT"))
        assert ast == [AstMerge(source="customers", source_is_file=False, on="id", how="LEFT", line=1)]

    def test_merge_path(self):
        ast = parse(tokenize("MERGE 'regions.csv' ON region_id"))
        assert ast == [AstMerge(source="regions.csv", source_is_file=True, on="region_id", how="INNER", line=1)]


class TestErrors:
    def test_preview_rejects_float(self):
        with pytest.raises(ParseError):
            parse(tokenize("PREVIEW 2.5"))

    def test_set_requires_assign(self):
        with pytest.raises(ParseError):
            parse(tokenize("SET x 1"))
