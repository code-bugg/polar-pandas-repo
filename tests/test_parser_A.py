"""
tests/test_parser_A.py
======================
Unit tests for Member A's parser subtask:
  - AST node dataclasses (22 statement nodes + 6 condition + 3 expression)
  - ParseError construction and attributes
  - _Parser cursor primitives: peek(), consume(), at_end()
  - _Parser typed helpers: expect_kw(), expect_ident(), expect_string(),
    expect_int()
  - _Parser lookahead predicates: next_is_kw(), next_is()
  - parse_statement() dispatcher: routes every keyword to the correct stub
  - _parse_all() returns an empty list on empty input

NOTE: All parse_*() methods owned by Members B, C, D are stubs that raise
      NotImplementedError. Tests here verify only that the dispatcher calls
      the correct stub — not the stub's output. End-to-end parsing is tested
      in test_parser_B.py, test_parser_C.py, test_parser_D.py.
"""

import pytest

from lexer.lexer import tokenize
from parser.parser import (
    # Error
    ParseError,
    # Expression nodes
    ExprLiteral, ExprColRef, ExprBinop,
    # Condition nodes
    CondCompare, CondIn, CondNull, CondBetween, CondAnd, CondOr,
    # Statement nodes
    AstLoad, AstExport, AstPreview, AstInfo, AstDescribe, AstSetEngine,
    AstSelect, AstFilter, AstDropNulls, AstDropDups, AstDropCol,
    AstFillNulls, AstCast, AstAddCol, AstRename, AstSort,
    AstGroup, AstCount, AstJoin, AstPlot,
    AstIf, AstFor,
    # Infrastructure
    _Parser, parse,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parser(source: str) -> _Parser:
    """Return a fresh _Parser loaded with the token stream for *source*."""
    return _Parser(tokenize(source))


def _patch_stubs(p: _Parser, *stub_names: str) -> None:
    """Replace the named stubs with no-ops that consume one token each,
    preventing infinite loops while allowing the dispatcher to be exercised."""
    def _noop():
        if not p.at_end():
            p.consume()
    for name in stub_names:
        setattr(p, name, _noop)


# ─────────────────────────────────────────────────────────────────────────────
# ParseError
# ─────────────────────────────────────────────────────────────────────────────

class TestParseError:

    def test_is_exception(self):
        assert isinstance(ParseError("oops", 1), Exception)

    def test_stores_message(self):
        err = ParseError("unexpected token", 7)
        assert err.message == "unexpected token"

    def test_stores_line(self):
        err = ParseError("unexpected token", 7)
        assert err.line == 7

    def test_str_contains_line_number(self):
        err = ParseError("bad kw", 12)
        assert "12" in str(err)

    def test_str_contains_parser_label(self):
        err = ParseError("bad kw", 3)
        assert "Parser" in str(err) or "parser" in str(err).lower()

    def test_str_contains_message(self):
        err = ParseError("expected LOAD", 2)
        assert "expected LOAD" in str(err)


# ─────────────────────────────────────────────────────────────────────────────
# Expression node dataclasses
# ─────────────────────────────────────────────────────────────────────────────

class TestExprNodes:

    def test_expr_literal_construction(self):
        n = ExprLiteral(value="42", kind="integer", line=1)
        assert n.value == "42"
        assert n.kind  == "integer"
        assert n.line  == 1

    def test_expr_literal_is_frozen(self):
        n = ExprLiteral(value="3.14", kind="float", line=1)
        with pytest.raises((AttributeError, TypeError)):
            n.value = "0"  # type: ignore

    def test_expr_col_ref_construction(self):
        n = ExprColRef(name="amount", df="df", is_loopvar=False, line=2)
        assert n.name       == "amount"
        assert n.df         == "df"
        assert n.is_loopvar is False
        assert n.line       == 2

    def test_expr_col_ref_bare(self):
        n = ExprColRef(name="amount", df=None, is_loopvar=False, line=1)
        assert n.df is None

    def test_expr_col_ref_loopvar(self):
        n = ExprColRef(name="col", df=None, is_loopvar=True, line=3)
        assert n.is_loopvar is True

    def test_expr_col_ref_is_frozen(self):
        n = ExprColRef(name="x", df=None, is_loopvar=False, line=1)
        with pytest.raises((AttributeError, TypeError)):
            n.name = "y"  # type: ignore

    def test_expr_binop_construction(self):
        left  = ExprLiteral("1", "integer", 1)
        right = ExprLiteral("2", "integer", 1)
        n = ExprBinop(op="+", left=left, right=right, line=1)
        assert n.op    == "+"
        assert n.left  is left
        assert n.right is right

    def test_expr_binop_is_frozen(self):
        left  = ExprLiteral("1", "integer", 1)
        right = ExprLiteral("2", "integer", 1)
        n = ExprBinop(op="+", left=left, right=right, line=1)
        with pytest.raises((AttributeError, TypeError)):
            n.op = "-"  # type: ignore

    def test_expr_nodes_are_hashable(self):
        n = ExprLiteral("1", "integer", 1)
        assert n in {n}


# ─────────────────────────────────────────────────────────────────────────────
# Condition node dataclasses
# ─────────────────────────────────────────────────────────────────────────────

class TestCondNodes:

    def _lit(self, v="1"):
        return ExprLiteral(v, "integer", 1)

    def _col(self, name="x"):
        return ExprColRef(name, None, False, 1)

    def test_cond_compare_construction(self):
        n = CondCompare(left=self._col(), op=">=", right=self._lit(), line=1)
        assert n.op == ">="

    def test_cond_compare_is_frozen(self):
        n = CondCompare(left=self._col(), op="==", right=self._lit(), line=1)
        with pytest.raises((AttributeError, TypeError)):
            n.op = "!="  # type: ignore

    def test_cond_in_construction(self):
        values = [ExprLiteral("a", "string", 1), ExprLiteral("b", "string", 1)]
        n = CondIn(col=self._col(), values=values, line=1)
        assert len(n.values) == 2

    def test_cond_null_construction(self):
        n = CondNull(col=self._col(), is_not=True, line=1)
        assert n.is_not is True

    def test_cond_between_construction(self):
        n = CondBetween(col=self._col(), low=self._lit("0"), high=self._lit("100"), line=1)
        assert n.low.value  == "0"
        assert n.high.value == "100"

    def test_cond_and_construction(self):
        c1 = CondCompare(self._col("a"), "==", self._lit(), 1)
        c2 = CondCompare(self._col("b"), "!=", self._lit(), 1)
        n  = CondAnd(left=c1, right=c2, line=1)
        assert n.left is c1
        assert n.right is c2

    def test_cond_or_construction(self):
        c1 = CondCompare(self._col("a"), ">", self._lit(), 1)
        c2 = CondCompare(self._col("b"), "<", self._lit(), 1)
        n  = CondOr(left=c1, right=c2, line=1)
        assert n.left is c1

    def test_all_cond_nodes_are_frozen(self):
        nodes = [
            CondCompare(self._col(), "==", self._lit(), 1),
            CondNull(self._col(), False, 1),
            CondBetween(self._col(), self._lit(), self._lit(), 1),
            CondAnd(CondCompare(self._col(), "==", self._lit(), 1),
                    CondCompare(self._col(), "==", self._lit(), 1), 1),
            CondOr(CondCompare(self._col(), "==", self._lit(), 1),
                   CondCompare(self._col(), "==", self._lit(), 1), 1),
        ]
        for n in nodes:
            with pytest.raises((AttributeError, TypeError)):
                setattr(n, "line", 999)

    def test_cond_nodes_are_hashable(self):
        n = CondNull(col=self._col(), is_not=False, line=1)
        assert n in {n}


# ─────────────────────────────────────────────────────────────────────────────
# Statement AST node dataclasses — existence, fields, frozen
# ─────────────────────────────────────────────────────────────────────────────

class TestStatementNodes:

    def test_exactly_22_statement_node_types(self):
        node_types = [
            AstLoad, AstExport, AstPreview, AstInfo, AstDescribe, AstSetEngine,
            AstSelect, AstFilter, AstDropNulls, AstDropDups, AstDropCol,
            AstFillNulls, AstCast, AstAddCol, AstRename, AstSort,
            AstGroup, AstCount, AstJoin, AstPlot,
            AstIf, AstFor,
        ]
        assert len(node_types) == 22

    def test_ast_load(self):
        n = AstLoad(file="data.csv", name="df", line=1)
        assert n.file == "data.csv"
        assert n.name == "df"
        assert n.line == 1

    def test_ast_export(self):
        n = AstExport(name="df", file="out.csv", line=2)
        assert n.name == "df"
        assert n.file == "out.csv"

    def test_ast_preview(self):
        n = AstPreview(name="df", rows=10, line=1)
        assert n.rows == 10

    def test_ast_info(self):
        n = AstInfo(name="df", line=1)
        assert n.name == "df"

    def test_ast_describe(self):
        n = AstDescribe(name="df", line=1)
        assert n.name == "df"

    def test_ast_set_engine(self):
        n = AstSetEngine(engine="polars", line=1)
        assert n.engine == "polars"

    def test_ast_select(self):
        n = AstSelect(name="df", columns=["a", "b"], line=1)
        assert n.columns == ["a", "b"]

    def test_ast_filter(self):
        cond = CondCompare(
            ExprColRef("x", None, False, 1), ">",
            ExprLiteral("0", "integer", 1), 1
        )
        n = AstFilter(name="df", condition=cond, line=1)
        assert n.condition is cond

    def test_ast_drop_nulls(self):
        n = AstDropNulls(name="df", columns=[], line=1)
        assert n.columns == []

    def test_ast_drop_dups(self):
        n = AstDropDups(name="df", columns=["id"], line=1)
        assert n.columns == ["id"]

    def test_ast_drop_col(self):
        n = AstDropCol(name="df", column="old_col", line=1)
        assert n.column == "old_col"

    def test_ast_fill_nulls(self):
        fill = ExprLiteral("0", "integer", 1)
        n = AstFillNulls(name="df", columns=[], fill=fill, line=1)
        assert n.fill is fill

    def test_ast_cast(self):
        n = AstCast(name="df", column="age", to="INT", line=1)
        assert n.to == "INT"

    def test_ast_add_col(self):
        expr = ExprLiteral("1", "integer", 1)
        n = AstAddCol(name="df", column="new_col", expr=expr, line=1)
        assert n.column == "new_col"

    def test_ast_rename(self):
        n = AstRename(name="df", old="col_a", new="col_b", line=1)
        assert n.old == "col_a" and n.new == "col_b"

    def test_ast_sort(self):
        n = AstSort(name="df", column="amount", direction="DESC", line=1)
        assert n.direction == "DESC"

    def test_ast_group(self):
        n = AstGroup(name="df", by="cat", agg="SUM", on="val", result="r", line=1)
        assert n.agg == "SUM"
        assert n.result == "r"

    def test_ast_group_result_optional(self):
        n = AstGroup(name="df", by="cat", agg="MEAN", on="val", result=None, line=1)
        assert n.result is None

    def test_ast_count(self):
        n = AstCount(name="df", result="n", line=1)
        assert n.result == "n"

    def test_ast_count_result_optional(self):
        n = AstCount(name="df", result=None, line=1)
        assert n.result is None

    def test_ast_join(self):
        n = AstJoin(left="df1", right="df2", on="id", how="LEFT", result="j", line=1)
        assert n.how == "LEFT"
        assert n.result == "j"

    def test_ast_plot(self):
        n = AstPlot(name="df", kind="BAR", x="cat", y="val",
                    title="My Chart", save=None, line=1)
        assert n.kind  == "BAR"
        assert n.title == "My Chart"
        assert n.save  is None

    def test_ast_if(self):
        cond = CondNull(ExprColRef("x", None, False, 1), False, 1)
        n = AstIf(condition=cond, then_body=[AstInfo("df", 1)],
                  else_body=[], line=1)
        assert len(n.then_body) == 1
        assert n.else_body == []

    def test_ast_for(self):
        n = AstFor(var="col", columns=["a", "b"],
                   body=[AstInfo("df", 1)], line=1)
        assert n.var == "col"
        assert n.columns == ["a", "b"]

    @pytest.mark.parametrize("node", [
        AstLoad("f", "df", 1),
        AstExport("df", "f", 1),
        AstPreview("df", 5, 1),
        AstInfo("df", 1),
        AstDescribe("df", 1),
        AstSetEngine("pandas", 1),
        AstSelect("df", [], 1),
        AstDropNulls("df", [], 1),
        AstDropDups("df", [], 1),
        AstDropCol("df", "c", 1),
        AstCast("df", "c", "INT", 1),
        AstRename("df", "a", "b", 1),
        AstSort("df", "c", "ASC", 1),
        AstGroup("df", "c", "SUM", "v", None, 1),
        AstCount("df", None, 1),
        AstJoin("l", "r", "id", "INNER", None, 1),
        AstPlot("df", "BAR", None, None, None, None, 1),
        AstIf(CondNull(ExprColRef("x", None, False, 1), False, 1), [], [], 1),
        AstFor("col", [], [], 1),
    ])
    def test_statement_nodes_are_frozen(self, node):
        with pytest.raises((AttributeError, TypeError)):
            setattr(node, "line", 999)

    @pytest.mark.parametrize("node", [
        AstLoad("f", "df", 1),
        AstInfo("df", 1),
        AstCount("df", None, 1),
    ])
    def test_statement_nodes_are_hashable(self, node):
        assert node in {node}


# ─────────────────────────────────────────────────────────────────────────────
# _Parser cursor primitives
# ─────────────────────────────────────────────────────────────────────────────

class TestParserCursor:

    def test_peek_returns_first_token(self):
        p = _parser('LOAD "data.csv" AS df')
        assert p.peek().value == "LOAD"

    def test_peek_with_offset(self):
        p = _parser('LOAD "data.csv" AS df')
        assert p.peek(0).value == "LOAD"
        assert p.peek(1).value == '"data.csv"'
        assert p.peek(2).value == "AS"

    def test_peek_past_end_returns_eof(self):
        p = _parser("LOAD")
        # Only 2 real tokens: KW("LOAD") + EOF
        assert p.peek(999).type.name == "EOF"

    def test_peek_does_not_advance(self):
        p = _parser("LOAD")
        p.peek()
        p.peek()
        assert p.peek().value == "LOAD"

    def test_consume_returns_token(self):
        p = _parser('LOAD "data.csv"')
        tok = p.consume()
        assert tok.value == "LOAD"

    def test_consume_advances_cursor(self):
        p = _parser('LOAD "data.csv"')
        p.consume()
        assert p.peek().value == '"data.csv"'

    def test_consume_at_eof_raises_parse_error(self):
        p = _parser("")
        with pytest.raises(ParseError, match="end of input"):
            p.consume()

    def test_at_end_false_when_tokens_remain(self):
        p = _parser("LOAD")
        assert not p.at_end()

    def test_at_end_true_on_empty_source(self):
        p = _parser("")
        assert p.at_end()

    def test_at_end_true_after_consuming_all(self):
        p = _parser("LOAD")
        p.consume()   # consume LOAD; only EOF remains
        assert p.at_end()

    def test_eof_sentinel_is_always_last(self):
        p = _parser('LOAD "data.csv" AS df')
        while not p.at_end():
            p.consume()
        assert p.peek().type.name == "EOF"


# ─────────────────────────────────────────────────────────────────────────────
# _Parser typed helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestExpectHelpers:

    # ── expect_kw ─────────────────────────────────────────────────────────

    def test_expect_kw_consumes_matching_keyword(self):
        p = _parser("LOAD")
        tok = p.expect_kw("LOAD")
        assert tok.value == "LOAD"
        assert p.at_end()

    def test_expect_kw_case_insensitive_match(self):
        p = _parser("LOAD")
        tok = p.expect_kw("load")   # lowercase argument
        assert tok.value == "LOAD"

    def test_expect_kw_accepts_any_of_multiple(self):
        p = _parser("DESC")
        tok = p.expect_kw("ASC", "DESC")
        assert tok.value == "DESC"

    def test_expect_kw_raises_on_wrong_keyword(self):
        p = _parser("LOAD")
        with pytest.raises(ParseError):
            p.expect_kw("FILTER")

    def test_expect_kw_raises_on_non_keyword_token(self):
        p = _parser("df")    # IDENT, not KW
        with pytest.raises(ParseError):
            p.expect_kw("LOAD")

    def test_expect_kw_error_mentions_expected_keyword(self):
        p = _parser("df")
        with pytest.raises(ParseError, match="FILTER"):
            p.expect_kw("FILTER")

    def test_expect_kw_does_not_consume_on_failure(self):
        p = _parser("LOAD")
        try:
            p.expect_kw("FILTER")
        except ParseError:
            pass
        # Cursor must not have moved
        assert p.peek().value == "LOAD"

    # ── expect_ident ──────────────────────────────────────────────────────

    def test_expect_ident_consumes_ident(self):
        p = _parser("df")
        tok = p.expect_ident()
        assert tok.value == "df"
        assert p.at_end()

    def test_expect_ident_raises_on_keyword(self):
        p = _parser("LOAD")
        with pytest.raises(ParseError):
            p.expect_ident()

    def test_expect_ident_raises_on_string(self):
        p = _parser('"data.csv"')
        with pytest.raises(ParseError):
            p.expect_ident()

    def test_expect_ident_does_not_consume_on_failure(self):
        p = _parser("LOAD")
        try:
            p.expect_ident()
        except ParseError:
            pass
        assert p.peek().value == "LOAD"

    # ── expect_string ─────────────────────────────────────────────────────

    def test_expect_string_consumes_string(self):
        p = _parser('"data.csv"')
        tok = p.expect_string()
        assert tok.value == '"data.csv"'
        assert p.at_end()

    def test_expect_string_raises_on_ident(self):
        p = _parser("df")
        with pytest.raises(ParseError):
            p.expect_string()

    def test_expect_string_raises_on_keyword(self):
        p = _parser("LOAD")
        with pytest.raises(ParseError):
            p.expect_string()

    def test_expect_string_does_not_consume_on_failure(self):
        p = _parser("df")
        try:
            p.expect_string()
        except ParseError:
            pass
        assert p.peek().value == "df"

    # ── expect_int ────────────────────────────────────────────────────────

    def test_expect_int_consumes_integer(self):
        p = _parser("42")
        tok = p.expect_int()
        assert tok.value == "42"
        assert p.at_end()

    def test_expect_int_raises_on_float(self):
        p = _parser("3.14")
        with pytest.raises(ParseError):
            p.expect_int()

    def test_expect_int_raises_on_ident(self):
        p = _parser("df")
        with pytest.raises(ParseError):
            p.expect_int()

    def test_expect_int_does_not_consume_on_failure(self):
        p = _parser("3.14")
        try:
            p.expect_int()
        except ParseError:
            pass
        assert p.peek().value == "3.14"


# ─────────────────────────────────────────────────────────────────────────────
# _Parser lookahead predicates
# ─────────────────────────────────────────────────────────────────────────────

class TestLookaheadPredicates:

    def test_next_is_kw_true(self):
        p = _parser("LOAD")
        assert p.next_is_kw("LOAD")

    def test_next_is_kw_false_wrong_keyword(self):
        p = _parser("LOAD")
        assert not p.next_is_kw("FILTER")

    def test_next_is_kw_false_non_keyword(self):
        p = _parser("df")
        assert not p.next_is_kw("LOAD")

    def test_next_is_kw_case_insensitive(self):
        p = _parser("SORT")
        assert p.next_is_kw("sort")

    def test_next_is_kw_any_of_multiple_true(self):
        p = _parser("ASC")
        assert p.next_is_kw("ASC", "DESC")

    def test_next_is_kw_any_of_multiple_false(self):
        p = _parser("LOAD")
        assert not p.next_is_kw("ASC", "DESC")

    def test_next_is_kw_does_not_consume(self):
        p = _parser("LOAD")
        p.next_is_kw("LOAD")
        assert p.peek().value == "LOAD"

    def test_next_is_true(self):
        from lexer.lexer import TokenType
        p = _parser("df")
        assert p.next_is(TokenType.IDENT)

    def test_next_is_false(self):
        from lexer.lexer import TokenType
        p = _parser("df")
        assert not p.next_is(TokenType.KW)

    def test_next_is_does_not_consume(self):
        from lexer.lexer import TokenType
        p = _parser("LOAD")
        p.next_is(TokenType.KW)
        assert p.peek().value == "LOAD"


# ─────────────────────────────────────────────────────────────────────────────
# parse_statement() dispatcher
# Each test confirms that the right stub is invoked for the leading keyword.
# We replace the target stub with a sentinel, let the dispatcher fire, then
# assert the sentinel was called.  All *other* stubs are left as-is (they
# raise NotImplementedError, which is fine — we catch it if it bubbles up).
# ─────────────────────────────────────────────────────────────────────────────

class TestDispatcherRouting:

    def _fires(self, source: str, stub_name: str) -> bool:
        """Return True if parse_statement() calls the named stub."""
        called = []
        p = _parser(source)

        def sentinel():
            called.append(True)
            # Consume the leading keyword so the cursor advances
            if not p.at_end():
                p.consume()

        setattr(p, stub_name, sentinel)
        try:
            p.parse_statement()
        except (NotImplementedError, ParseError):
            pass   # other stubs or helpers may still raise — that is OK
        return bool(called)

    # ── Member B stubs ────────────────────────────────────────────────────

    @pytest.mark.parametrize("source,stub", [
        ("LOAD",    "parse_load"),
        ("EXPORT",  "parse_export"),
        ("PREVIEW", "parse_preview"),
        ("INFO",    "parse_info"),
        ("DESCRIBE","parse_describe"),
        ("SET",     "parse_set_engine"),
    ])
    def test_member_b_dispatch(self, source, stub):
        assert self._fires(source, stub), \
            f"parse_statement() did not call {stub} for {source!r}"

    # ── Member C stubs ────────────────────────────────────────────────────

    @pytest.mark.parametrize("source,stub", [
        ("SELECT", "parse_select"),
        ("FILTER", "parse_filter"),
        ("DROP",   "parse_drop"),
        ("FILL",   "parse_fill_nulls"),
        ("CAST",   "parse_cast"),
        ("ADD",    "parse_add_col"),
        ("RENAME", "parse_rename"),
        ("SORT",   "parse_sort"),
    ])
    def test_member_c_dispatch(self, source, stub):
        assert self._fires(source, stub), \
            f"parse_statement() did not call {stub} for {source!r}"

    # ── Member D stubs ────────────────────────────────────────────────────

    @pytest.mark.parametrize("source,stub", [
        ("GROUP", "parse_group"),
        ("COUNT", "parse_count"),
        ("JOIN",  "parse_join"),
        ("PLOT",  "parse_plot"),
        ("IF",    "parse_if"),
        ("FOR",   "parse_for"),
    ])
    def test_member_d_dispatch(self, source, stub):
        assert self._fires(source, stub), \
            f"parse_statement() did not call {stub} for {source!r}"

    def test_non_keyword_token_raises_parse_error(self):
        p = _parser("df")   # IDENT, not a statement keyword
        with pytest.raises(ParseError):
            p.parse_statement()

    def test_unknown_keyword_raises_parse_error(self):
        # "USING" is a keyword in the lexer but not a statement opener
        p = _parser("USING")
        with pytest.raises(ParseError):
            p.parse_statement()

    def test_eof_raises_parse_error(self):
        p = _parser("")
        with pytest.raises(ParseError):
            p.parse_statement()


# ─────────────────────────────────────────────────────────────────────────────
# _parse_all() — the main loop
# ─────────────────────────────────────────────────────────────────────────────

class TestParseAll:

    def test_empty_source_returns_empty_list(self):
        result = parse(tokenize(""))
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        result = parse(tokenize("   \n\t  "))
        assert result == []

    def test_returns_a_list(self):
        result = parse(tokenize(""))
        assert isinstance(result, list)