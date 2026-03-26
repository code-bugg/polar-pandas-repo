"""
tests/test_parser_A.py
======================
Unit tests for Member A's parser subtask: AST datastructures and base Parser
"""

import pytest

from lexer.lexer import tokenize
from parser.parser import (
    ParseError, ExprLiteral, ExprColRef, ExprBinop,
    CondCompare, CondIn, CondNull, CondBetween, CondAnd, CondOr,
    AstLoad, AstExport, AstPreview, AstInfo, AstDescribe, AstSetEngine,
    AstSelect, AstFilter, AstDropNulls, AstDropDups, AstDropCol,
    AstFillNulls, AstCast, AstAddCol, AstRename, AstSort,
    AstGroup, AstCount, AstJoin, AstPlot, AstIf, AstFor, AstPipeline,
    _Parser, parse,
)

def _parser(source: str) -> _Parser:
    return _Parser(tokenize(source))

class TestExprNodes:
    def test_expr_literal_construction(self):
        n = ExprLiteral(value="42", kind="integer", line=1)
        assert n.value == "42"
        assert n.kind  == "integer"

    def test_expr_col_ref_construction(self):
        n = ExprColRef(name="amount", is_loopvar=False, line=2)
        assert n.name == "amount"
        assert n.is_loopvar is False

    def test_expr_binop_construction(self):
        left  = ExprLiteral("1", "integer", 1)
        right = ExprLiteral("2", "integer", 1)
        n = ExprBinop(op="+", left=left, right=right, line=1)
        assert n.op == "+"

class TestCondNodes:
    def _lit(self, v="1"): return ExprLiteral(v, "integer", 1)
    def _col(self, name="x"): return ExprColRef(name, False, 1)

    def test_cond_compare_construction(self):
        n = CondCompare(left=self._col(), op=">=", right=self._lit(), line=1)
        assert n.op == ">="

    def test_all_cond_nodes_are_frozen(self):
        nodes = [
            CondCompare(self._col(), "==", self._lit(), 1),
            CondNull(self._col(), False, 1),
            CondBetween(self._col(), self._lit(), self._lit(), 1),
        ]
        for n in nodes:
            with pytest.raises((AttributeError, TypeError)):
                setattr(n, "line", 999)

class TestStatementNodes:
    def test_exactly_23_statement_node_types(self):
        node_types = [
            AstLoad, AstExport, AstPreview, AstInfo, AstDescribe, AstSetEngine,
            AstSelect, AstFilter, AstDropNulls, AstDropDups, AstDropCol,
            AstFillNulls, AstCast, AstAddCol, AstRename, AstSort,
            AstGroup, AstCount, AstJoin, AstPlot,
            AstIf, AstFor, AstPipeline,
        ]
        assert len(node_types) == 23

    @pytest.mark.parametrize("node", [
        AstLoad("f", "df", "pandas", 1),
        AstExport("df", "f", "csv", 1),
        AstSelect("df", [], 1),
        AstDropDups("df", "first", 1),
        AstFillNulls("df", "col", ExprLiteral("0", "integer", 1), 1),
        AstGroup("df", ["cat"], "val", "sum", 1),
        AstCount("df", ["cat"], 1),
        AstJoin("l", "r", "id", "inner", "j", 1),
        AstPlot("df", "bar", "x", "y", None, None, "T", None, None, None, None, None, 1),
        AstIf(CondNull(ExprColRef("x", False, 1), False, 1), [], [], 1),
        AstFor("col", [], [], 1),
        AstPipeline("df", [], 1),
    ])
    def test_statement_nodes_are_frozen(self, node):
        with pytest.raises((AttributeError, TypeError)):
            setattr(node, "line", 999)

class TestDispatcherRouting:
    def test_non_keyword_token_raises_parse_error(self):
        p = _parser("123")
        with pytest.raises(ParseError):
            p.parse_statement()

class TestParseAll:
    def test_empty_source_returns_empty_list(self):
        assert parse(tokenize("")) == []