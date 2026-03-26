"""
tests/test_parser_C.py
======================
Unit tests for Member C's parser subtask
"""

import pytest
from lexer.lexer import tokenize
from parser.parser import (
    ParseError, ExprLiteral, ExprColRef, ExprBinop,
    AstSelect, AstFilter, AstDropNulls, AstDropDups, AstDropCol,
    AstFillNulls, AstCast, AstAddCol, AstRename, AstSort,
    _Parser, parse,
)

def _parser(source: str) -> _Parser:
    return _Parser(tokenize(source))

class TestSelect:
    def test_select_explicit_columns(self):
        ast = parse(tokenize('SELECT df COLUMNS [a, b, c]'))
        node = ast[0]
        assert isinstance(node, AstSelect)
        assert node.name == "df"
        assert node.columns == ["a", "b", "c"]

    def test_select_all_columns(self):
        ast = parse(tokenize('SELECT df COLUMNS *'))
        assert ast[0].columns == []

class TestDrop:
    def test_drop_nulls_all_columns(self):
        ast = parse(tokenize('DROP NULLS FROM df'))
        assert isinstance(ast[0], AstDropNulls)
        assert ast[0].name == "df"
        assert ast[0].columns == []

    def test_drop_duplicates_all_columns(self):
        ast = parse(tokenize('DROP DUPLICATES FROM df KEEP FIRST'))
        assert isinstance(ast[0], AstDropDups)
        assert ast[0].name == "df"
        assert ast[0].keep == "first"

    def test_drop_column(self):
        ast = parse(tokenize('DROP COLUMN unused_col FROM sales_df'))
        assert isinstance(ast[0], AstDropCol)
        assert ast[0].column == "unused_col"

class TestFillNulls:
    def test_fill_nulls_all_columns_literal(self):
        ast = parse(tokenize('FILL NULLS IN df COLUMN target WITH 0'))
        node = ast[0]
        assert isinstance(node, AstFillNulls)
        assert node.column == "target"
        assert node.fill.value == "0"

class TestCast:
    def test_cast_to_int(self):
        ast = parse(tokenize('CAST sales COLUMN amount TO INT'))
        node = ast[0]
        assert isinstance(node, AstCast)
        assert node.name == "sales"
        assert node.column == "amount"
        assert node.dtype == "int"

class TestAddColumn:
    def test_add_column_literal(self):
        ast = parse(tokenize('ADD COLUMN new_col TO df AS 42'))
        node = ast[0]
        assert isinstance(node, AstAddCol)
        assert node.column == "new_col"
        assert node.expr.value == "42"

class TestRename:
    def test_rename_basic(self):
        ast = parse(tokenize('RENAME df COLUMN old_name TO new_name'))
        node = ast[0]
        assert isinstance(node, AstRename)
        assert node.old == "old_name"
        assert node.new == "new_name"

class TestSort:
    def test_sort_default_asc(self):
        ast = parse(tokenize('SORT df BY amount'))
        node = ast[0]
        assert isinstance(node, AstSort)
        assert node.column == "amount"
        assert node.direction == "ASC"

class TestExpressions:
    def test_expr_atom_integer(self):
        expr = _parser("42").parse_expr()
        assert isinstance(expr, ExprLiteral)
        assert expr.value == "42"

    def test_expr_atom_column_bare(self):
        expr = _parser("amount").parse_expr()
        assert isinstance(expr, ExprColRef)
        assert expr.name == "amount"

    def test_expr_add(self):
        expr = _parser("1 + 2").parse_expr()
        assert isinstance(expr, ExprBinop)
        assert expr.op == "+"