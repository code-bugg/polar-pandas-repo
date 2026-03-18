"""
tests/test_parser_C.py
======================
Unit tests for Member C's parser subtask:
  - Cleaning and Transformation statements (Member C)
  - SELECT, FILTER (where conditions are partially tested)
  - DROP (3 variants: NULLS, DUPLICATES, COLUMN)
  - FILL NULLS, CAST
  - ADD COLUMN, RENAME, SORT
  - Expression parsing: parse_expr, parse_expr_atom
"""

import pytest

from lexer.lexer import tokenize
from parser.parser import (
    ParseError,
    ExprLiteral, ExprColRef, ExprBinop,
    AstSelect, AstFilter, AstDropNulls, AstDropDups, AstDropCol,
    AstFillNulls, AstCast, AstAddCol, AstRename, AstSort,
    _Parser, parse,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parser(source: str) -> _Parser:
    """Return a fresh _Parser loaded with the token stream for *source*."""
    return _Parser(tokenize(source))


# ─────────────────────────────────────────────────────────────────────────────
# SELECT statement tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSelect:

    def test_select_explicit_columns(self):
        """SELECT COLUMNS [col1, col2] FROM df"""
        ast = parse(tokenize('SELECT COLUMNS [a, b, c] FROM df'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstSelect)
        assert node.name == "df"
        assert node.columns == ["a", "b", "c"]
        assert node.line == 1

    def test_select_all_columns(self):
        """SELECT COLUMNS * FROM df — empty columns list means all"""
        ast = parse(tokenize('SELECT COLUMNS * FROM df'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstSelect)
        assert node.name == "df"
        assert node.columns == []
        assert node.line == 1

    def test_select_single_column(self):
        """SELECT COLUMNS [amount] FROM sales"""
        ast = parse(tokenize('SELECT COLUMNS [amount] FROM sales'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstSelect)
        assert node.name == "sales"
        assert node.columns == ["amount"]

    def test_select_quoted_columns(self):
        """SELECT COLUMNS ["col 1", "col 2"] FROM df"""
        ast = parse(tokenize('SELECT COLUMNS ["col 1", "col 2"] FROM df'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstSelect)
        assert node.name == "df"
        assert node.columns == ["col 1", "col 2"]

    def test_select_missing_columns_keyword(self):
        """SELECT without COLUMNS keyword should fail"""
        with pytest.raises(ParseError) as exc_info:
            parse(tokenize('SELECT [a, b] FROM df'))
        assert "COLUMNS" in str(exc_info.value)

    def test_select_missing_from(self):
        """SELECT COLUMNS [a] without FROM should fail"""
        with pytest.raises(ParseError) as exc_info:
            parse(tokenize('SELECT COLUMNS [a] df'))
        assert "FROM" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# DROP statement tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDrop:

    def test_drop_nulls_all_columns(self):
        """DROP NULLS FROM df — no columns specified"""
        ast = parse(tokenize('DROP NULLS FROM df'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstDropNulls)
        assert node.name == "df"
        assert node.columns == []

    def test_drop_nulls_specific_columns(self):
        """DROP NULLS FROM df COLUMNS [a, b]"""
        ast = parse(tokenize('DROP NULLS FROM df COLUMNS [a, b]'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstDropNulls)
        assert node.name == "df"
        assert node.columns == ["a", "b"]

    def test_drop_duplicates_all_columns(self):
        """DROP DUPLICATES FROM df"""
        ast = parse(tokenize('DROP DUPLICATES FROM df'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstDropDups)
        assert node.name == "df"
        assert node.columns == []

    def test_drop_duplicates_specific_columns(self):
        """DROP DUPLICATES FROM df COLUMNS [id, name]"""
        ast = parse(tokenize('DROP DUPLICATES FROM df COLUMNS [id, name]'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstDropDups)
        assert node.name == "df"
        assert node.columns == ["id", "name"]

    def test_drop_column(self):
        """DROP COLUMN colname FROM df"""
        ast = parse(tokenize('DROP COLUMN unused_col FROM sales_df'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstDropCol)
        assert node.name == "sales_df"
        assert node.column == "unused_col"

    def test_drop_invalid_variant(self):
        """DROP with invalid variant should fail"""
        with pytest.raises(ParseError):
            parse(tokenize('DROP INVALID FROM df'))

    def test_drop_nulls_missing_from(self):
        """DROP NULLS without FROM should fail"""
        with pytest.raises(ParseError) as exc_info:
            parse(tokenize('DROP NULLS df'))
        assert "FROM" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# FILL NULLS statement tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFillNulls:

    def test_fill_nulls_all_columns_literal(self):
        """FILL NULLS IN df WITH 0"""
        ast = parse(tokenize('FILL NULLS IN df WITH 0'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstFillNulls)
        assert node.name == "df"
        assert node.columns == []
        assert isinstance(node.fill, ExprLiteral)
        assert node.fill.kind == "integer"
        assert node.fill.value == "0"

    def test_fill_nulls_specific_columns_string(self):
        """FILL NULLS IN df COLUMNS [a, b] WITH "unknown" """
        ast = parse(tokenize('FILL NULLS IN df COLUMNS [a, b] WITH "unknown"'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstFillNulls)
        assert node.name == "df"
        assert node.columns == ["a", "b"]
        assert isinstance(node.fill, ExprLiteral)
        assert node.fill.kind == "string"

    def test_fill_nulls_with_column_ref(self):
        """FILL NULLS IN df WITH other_col"""
        ast = parse(tokenize('FILL NULLS IN df WITH other_col'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstFillNulls)
        assert node.name == "df"
        assert isinstance(node.fill, ExprColRef)
        assert node.fill.name == "other_col"

    def test_fill_nulls_missing_with(self):
        """FILL NULLS without WITH clause should fail"""
        with pytest.raises(ParseError) as exc_info:
            parse(tokenize('FILL NULLS IN df'))
        assert "WITH" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# CAST statement tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCast:

    def test_cast_to_int(self):
        """CAST amount IN sales TO INT"""
        ast = parse(tokenize('CAST amount IN sales TO INT'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstCast)
        assert node.name == "sales"
        assert node.column == "amount"
        assert node.to == "INT"

    def test_cast_to_float(self):
        """CAST price IN inventory TO FLOAT"""
        ast = parse(tokenize('CAST price IN inventory TO FLOAT'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstCast)
        assert node.to == "FLOAT"

    def test_cast_to_str(self):
        """CAST id IN users TO STR"""
        ast = parse(tokenize('CAST id IN users TO STR'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstCast)
        assert node.to == "STR"

    def test_cast_missing_in(self):
        """CAST without IN should fail"""
        with pytest.raises(ParseError) as exc_info:
            parse(tokenize('CAST amount sales TO INT'))
        assert "IN" in str(exc_info.value)

    def test_cast_missing_to(self):
        """CAST without TO should fail"""
        with pytest.raises(ParseError) as exc_info:
            parse(tokenize('CAST amount IN sales INT'))
        assert "TO" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# ADD COLUMN statement tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAddColumn:

    def test_add_column_literal(self):
        """ADD COLUMN new_col TO df AS 42"""
        ast = parse(tokenize('ADD COLUMN new_col TO df AS 42'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstAddCol)
        assert node.name == "df"
        assert node.column == "new_col"
        assert isinstance(node.expr, ExprLiteral)
        assert node.expr.value == "42"

    def test_add_column_column_ref(self):
        """ADD COLUMN doubled TO df AS amount"""
        ast = parse(tokenize('ADD COLUMN doubled TO df AS amount'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstAddCol)
        assert node.column == "doubled"
        assert isinstance(node.expr, ExprColRef)
        assert node.expr.name == "amount"

    def test_add_column_expression(self):
        """ADD COLUMN result TO df AS a + b"""
        ast = parse(tokenize('ADD COLUMN result TO df AS a + b'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstAddCol)
        assert isinstance(node.expr, ExprBinop)
        assert node.expr.op == "+"

    def test_add_column_missing_to(self):
        """ADD COLUMN without TO should fail"""
        with pytest.raises(ParseError) as exc_info:
            parse(tokenize('ADD COLUMN new_col df AS 42'))
        assert "TO" in str(exc_info.value)

    def test_add_column_missing_as(self):
        """ADD COLUMN without AS should fail"""
        with pytest.raises(ParseError) as exc_info:
            parse(tokenize('ADD COLUMN new_col TO df 42'))
        assert "AS" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# RENAME statement tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRename:

    def test_rename_basic(self):
        """RENAME old_name TO new_name IN df"""
        ast = parse(tokenize('RENAME old_name TO new_name IN df'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstRename)
        assert node.name == "df"
        assert node.old == "old_name"
        assert node.new == "new_name"

    def test_rename_different_df(self):
        """RENAME amount TO total IN sales"""
        ast = parse(tokenize('RENAME amount TO total IN sales'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstRename)
        assert node.name == "sales"
        assert node.old == "amount"
        assert node.new == "total"

    def test_rename_missing_to(self):
        """RENAME without TO should fail"""
        with pytest.raises(ParseError) as exc_info:
            parse(tokenize('RENAME old_name new_name IN df'))
        assert "TO" in str(exc_info.value)

    def test_rename_missing_in(self):
        """RENAME without IN should fail"""
        with pytest.raises(ParseError) as exc_info:
            parse(tokenize('RENAME old_name TO new_name df'))
        assert "IN" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# SORT statement tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSort:

    def test_sort_default_asc(self):
        """SORT df BY column (defaults to ASC)"""
        ast = parse(tokenize('SORT df BY amount'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstSort)
        assert node.name == "df"
        assert node.column == "amount"
        assert node.direction == "ASC"

    def test_sort_explicit_asc(self):
        """SORT df BY column ASC"""
        ast = parse(tokenize('SORT df BY amount ASC'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstSort)
        assert node.direction == "ASC"

    def test_sort_desc(self):
        """SORT df BY column DESC"""
        ast = parse(tokenize('SORT df BY price DESC'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstSort)
        assert node.name == "df"
        assert node.column == "price"
        assert node.direction == "DESC"

    def test_sort_missing_by(self):
        """SORT without BY should fail"""
        with pytest.raises(ParseError) as exc_info:
            parse(tokenize('SORT df column'))
        assert "BY" in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────────
# Expression parsing tests: parse_expr and parse_expr_atom
# ─────────────────────────────────────────────────────────────────────────────

class TestExpressions:

    def test_expr_atom_integer(self):
        """Literal: 42"""
        p = _parser("42")
        expr = p.parse_expr()
        assert isinstance(expr, ExprLiteral)
        assert expr.kind == "integer"
        assert expr.value == "42"

    def test_expr_atom_float(self):
        """Literal: 3.14"""
        p = _parser("3.14")
        expr = p.parse_expr()
        assert isinstance(expr, ExprLiteral)
        assert expr.kind == "float"
        assert expr.value == "3.14"

    def test_expr_atom_string(self):
        """Literal: "hello" """
        p = _parser('"hello"')
        expr = p.parse_expr()
        assert isinstance(expr, ExprLiteral)
        assert expr.kind == "string"
        assert expr.value == '"hello"'

    def test_expr_atom_bool_true(self):
        """Literal: true"""
        p = _parser("true")
        expr = p.parse_expr()
        assert isinstance(expr, ExprLiteral)
        assert expr.kind == "bool"
        assert expr.value == "TRUE"

    def test_expr_atom_column_bare(self):
        """Column reference: amount"""
        p = _parser("amount")
        expr = p.parse_expr()
        assert isinstance(expr, ExprColRef)
        assert expr.name == "amount"
        assert expr.df is None
        assert expr.is_loopvar is False

    def test_expr_atom_column_qualified(self):
        """Column reference: df.amount"""
        p = _parser("df.amount")
        expr = p.parse_expr()
        assert isinstance(expr, ExprColRef)
        assert expr.name == "amount"
        assert expr.df == "df"
        assert expr.is_loopvar is False

    def test_expr_atom_loopvar(self):
        """Loop variable: $col"""
        p = _parser("$col")
        expr = p.parse_expr()
        assert isinstance(expr, ExprColRef)
        assert expr.name == "col"
        assert expr.df is None
        assert expr.is_loopvar is True

    def test_expr_add(self):
        """Binary: 1 + 2"""
        p = _parser("1 + 2")
        expr = p.parse_expr()
        assert isinstance(expr, ExprBinop)
        assert expr.op == "+"
        assert isinstance(expr.left, ExprLiteral)
        assert isinstance(expr.right, ExprLiteral)

    def test_expr_subtract(self):
        """Binary: a - b"""
        p = _parser("a - b")
        expr = p.parse_expr()
        assert isinstance(expr, ExprBinop)
        assert expr.op == "-"

    def test_expr_multiply(self):
        """Binary: a * 2"""
        p = _parser("a * 2")
        expr = p.parse_expr()
        assert isinstance(expr, ExprBinop)
        assert expr.op == "*"

    def test_expr_divide(self):
        """Binary: num1 / num2"""
        p = _parser("num1 / num2")
        expr = p.parse_expr()
        assert isinstance(expr, ExprBinop)
        assert expr.op == "/"

    def test_expr_modulo(self):
        """Binary: a % b"""
        p = _parser("a % b")
        expr = p.parse_expr()
        assert isinstance(expr, ExprBinop)
        assert expr.op == "%"

    def test_expr_associativity_add_left(self):
        """1 + 2 + 3 should be (1 + 2) + 3"""
        p = _parser("1 + 2 + 3")
        expr = p.parse_expr()
        assert isinstance(expr, ExprBinop)
        assert expr.op == "+"
        assert isinstance(expr.left, ExprBinop)
        assert expr.left.op == "+"

    def test_expr_precedence_multiply_before_add(self):
        """1 + 2 * 3 should be 1 + (2 * 3)"""
        p = _parser("1 + 2 * 3")
        expr = p.parse_expr()
        assert isinstance(expr, ExprBinop)
        assert expr.op == "+"
        # Left should be literal 1
        assert isinstance(expr.left, ExprLiteral)
        # Right should be multiplication
        assert isinstance(expr.right, ExprBinop)
        assert expr.right.op == "*"

    def test_expr_precedence_multiply_divide_same(self):
        """a * b / c should be (a * b) / c (left-associative)"""
        p = _parser("a * b / c")
        expr = p.parse_expr()
        assert isinstance(expr, ExprBinop)
        assert expr.op == "/"
        assert isinstance(expr.left, ExprBinop)
        assert expr.left.op == "*"

    def test_expr_column_arithmetic(self):
        """amount * 2 + tax"""
        p = _parser("amount * 2 + tax")
        expr = p.parse_expr()
        assert isinstance(expr, ExprBinop)
        assert expr.op == "+"
        assert isinstance(expr.left, ExprBinop)
        assert expr.left.op == "*"

    def test_expr_qualified_column_arithmetic(self):
        """df.amount + df.tax"""
        p = _parser("df.amount + df.tax")
        expr = p.parse_expr()
        assert isinstance(expr, ExprBinop)
        assert expr.op == "+"
        assert expr.left.df == "df"
        assert expr.right.df == "df"

    def test_expr_invalid_atom(self):
        """Unexpected token in atom should fail"""
        p = _parser("[invalid]")
        with pytest.raises(ParseError):
            p.parse_expr()


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests: Multi-line programs
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_multiple_statements(self):
        """Multiple Member C statements in sequence"""
        source = """
        SELECT COLUMNS [a, b] FROM df
        DROP NULLS FROM df
        SORT df BY a
        """
        ast = parse(tokenize(source))
        assert len(ast) == 3
        assert isinstance(ast[0], AstSelect)
        assert isinstance(ast[1], AstDropNulls)
        assert isinstance(ast[2], AstSort)

    def test_complex_expression_in_add_column(self):
        """ADD COLUMN with complex multi-operator expression"""
        ast = parse(tokenize('ADD COLUMN result TO df AS a * b + c - d / e'))
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstAddCol)
        # The expression should be a complex binop tree
        assert isinstance(node.expr, ExprBinop)
