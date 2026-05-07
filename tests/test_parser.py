"""Unified unit tests for the parser component.

Merged from: test_parser_A.py, test_parser_B.py, test_parser_C.py, test_parser_D.py.
"""
import pytest
from lexer.lexer import TokenType, tokenize
from parser.parser import (
    ParseError,
    ExprLiteral,
    ExprColRef,
    ExprBinop,
    CondCompare,
    CondAnd,
    CondOr,
    AstRead,
    AstSave,
    AstDropEmpty,
    AstIf,
    _Parser,
    parse,
)
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
from parser.parser import (
    ParseError,
    ExprLiteral,
    ExprColRef,
    ExprBinop,
    CondCompare,
    CondIn,
    CondNull,
    CondBetween,
    CondContains,
    CondAnd,
    CondOr,
    AstFilter,
    _Parser,
    parse,
)
from parser.parser import (
    ParseError,
    MetaExists,
    MetaRowCount,
    MetaAnd,
    AstPipeline,
    AstDropEmpty,
    AstFilter,
    AstSet,
    AstIf,
    AstFor,
    parse,
)


# ---------------------------------------------------------------------------
# Block from test_parser_A.py
# ---------------------------------------------------------------------------
def _parser(source: str) -> _Parser:
    return _Parser(tokenize(source))


class TestCoreNodes:
    def test_expr_nodes_are_frozen(self):
        n = ExprLiteral("1", "integer", 1)
        with pytest.raises((AttributeError, TypeError)):
            n.value = "2"  # type: ignore

    def test_expr_colref_loopvar(self):
        n = ExprColRef("col", True, 2)
        assert n.is_loopvar is True

    def test_expr_binop(self):
        n = ExprBinop("+", ExprLiteral("1", "integer", 1), ExprLiteral("2", "integer", 1), 1)
        assert n.op == "+"

    def test_condition_nodes(self):
        c1 = CondCompare(ExprColRef("a", False, 1), ">", ExprLiteral("0", "integer", 1), 1)
        c2 = CondCompare(ExprColRef("b", False, 1), "==", ExprLiteral('"x"', "string", 1), 1)
        assert isinstance(CondAnd(c1, c2, 1), CondAnd)
        assert isinstance(CondOr(c1, c2, 1), CondOr)


class TestParseError:
    def test_stores_message_and_line(self):
        err = ParseError("bad", 7)
        assert err.message == "bad"
        assert err.line == 7


class TestParserCursor:
    def test_peek_consume_at_end(self):
        p = _parser("READ 'x.csv'")
        assert p.peek().value == "READ"
        p.consume()
        assert not p.at_end()

    def test_expect_helpers(self):
        p = _parser("READ 'x.csv'")
        assert p.expect_kw("READ").value == "READ"
        assert p.expect_string().value == "'x.csv'"

    def test_expect_int(self):
        p = _parser("10")
        assert p.expect_int().value == "10"

    def test_expect_int_rejects_float(self):
        p = _parser("1.5")
        with pytest.raises(ParseError):
            p.expect_int()

    def test_next_is_helpers(self):
        p = _parser("READ")
        assert p.next_is_kw("READ")
        assert p.next_is(TokenType.KW)


class TestDispatcher:
    def test_parse_statement_read(self):
        p = _parser("READ 'x.csv'")
        node = p.parse_statement()
        assert isinstance(node, AstRead)

    def test_parse_statement_operation(self):
        p = _parser("SAVE 'out.csv'")
        node = p.parse_statement()
        assert isinstance(node, AstSave)

    def test_parse_statement_if(self):
        p = _parser("IF EXISTS sales : INFO END")
        node = p.parse_statement()
        assert isinstance(node, AstIf)

    def test_non_statement_token_raises(self):
        p = _parser("'x'")
        with pytest.raises(ParseError):
            p.parse_statement()


class TestParseAll:
    def test_empty_returns_empty(self):
        assert parse(tokenize("")) == []

    def test_multiple_statements(self):
        nodes = parse(tokenize("READ 'x.csv' SAVE 'y.csv' DROP EMPTY"))
        assert isinstance(nodes[0], AstRead)
        assert isinstance(nodes[1], AstSave)
        assert isinstance(nodes[2], AstDropEmpty)


# ---------------------------------------------------------------------------
# Block from test_parser_B.py
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Block from test_parser_C.py
# ---------------------------------------------------------------------------
def _parser(source: str) -> _Parser:
    return _Parser(tokenize(source))


class TestExpressions:
    def test_literal_number(self):
        p = _parser("42")
        expr = p.parse_expr()
        assert expr == ExprLiteral("42", "integer", 1)

    def test_colref(self):
        p = _parser("amount")
        expr = p.parse_expr()
        assert expr == ExprColRef("amount", False, 1)

    def test_loopvar_ref(self):
        p = _parser("$col")
        expr = p.parse_expr()
        assert expr == ExprColRef("col", True, 1)

    def test_precedence(self):
        p = _parser("a + b * c")
        expr = p.parse_expr()
        assert isinstance(expr, ExprBinop)
        assert expr.op == "+"
        assert isinstance(expr.right, ExprBinop)
        assert expr.right.op == "*"

    def test_parentheses(self):
        p = _parser("(a + b) * c")
        expr = p.parse_expr()
        assert isinstance(expr, ExprBinop)
        assert expr.op == "*"


class TestConditions:
    def test_compare(self):
        node = parse(tokenize("FILTER WHERE amount >= 100"))[0]
        assert isinstance(node, AstFilter)
        assert isinstance(node.condition, CondCompare)

    def test_in(self):
        node = parse(tokenize("FILTER WHERE country IN 'MD', 'RO'"))[0]
        assert isinstance(node.condition, CondIn)
        assert node.condition.negate is False

    def test_not_in(self):
        node = parse(tokenize("FILTER WHERE country NOT IN 'XX', 'ZZ'"))[0]
        assert isinstance(node.condition, CondIn)
        assert node.condition.negate is True

    def test_is_empty(self):
        node = parse(tokenize("FILTER WHERE email IS EMPTY"))[0]
        assert isinstance(node.condition, CondNull)
        assert node.condition.is_not is False

    def test_is_not_empty(self):
        node = parse(tokenize("FILTER WHERE email IS NOT EMPTY"))[0]
        assert isinstance(node.condition, CondNull)
        assert node.condition.is_not is True

    def test_between(self):
        node = parse(tokenize("FILTER WHERE score BETWEEN 0 AND 100"))[0]
        assert isinstance(node.condition, CondBetween)

    def test_contains(self):
        node = parse(tokenize("FILTER WHERE email CONTAINS '@gmail'"))[0]
        assert isinstance(node.condition, CondContains)
        assert node.condition.negate is False

    def test_not_contains(self):
        node = parse(tokenize("FILTER WHERE notes NOT CONTAINS 'deleted'"))[0]
        assert isinstance(node.condition, CondContains)
        assert node.condition.negate is True

    def test_and_or_precedence(self):
        node = parse(tokenize("FILTER WHERE a == 1 OR b == 2 AND c == 3"))[0]
        assert isinstance(node.condition, CondOr)
        assert isinstance(node.condition.right, CondAnd)

    def test_condition_parentheses(self):
        node = parse(tokenize("FILTER WHERE (a == 1 OR b == 2) AND c == 3"))[0]
        assert isinstance(node.condition, CondAnd)


class TestFilterErrors:
    def test_missing_where(self):
        with pytest.raises(ParseError):
            parse(tokenize("FILTER amount > 0"))

    def test_bad_not_clause(self):
        with pytest.raises(ParseError):
            parse(tokenize("FILTER WHERE x NOT 1"))


# ---------------------------------------------------------------------------
# Block from test_parser_D.py
# ---------------------------------------------------------------------------
class TestPipeline:
    def test_named_pipeline(self):
        ast = parse(tokenize("orders -> FILTER WHERE total > 0 -> DROP EMPTY"))
        node = ast[0]
        assert isinstance(node, AstPipeline)
        assert node.target == "orders"
        assert len(node.steps) == 2
        assert isinstance(node.steps[0], AstFilter)
        assert isinstance(node.steps[1], AstDropEmpty)

    def test_active_pipeline(self):
        ast = parse(tokenize("-> SET gross = amount + tax -> DROP EMPTY amount"))
        node = ast[0]
        assert isinstance(node, AstPipeline)
        assert node.target is None
        assert isinstance(node.steps[0], AstSet)


class TestIfFor:
    def test_if_exists(self):
        source = "IF EXISTS reference_table : MERGE reference_table ON code LEFT END"
        ast = parse(tokenize(source))
        node = ast[0]
        assert isinstance(node, AstIf)
        assert isinstance(node.condition, MetaExists)
        assert len(node.then_body) == 1

    def test_if_rowcount(self):
        source = "IF ROWCOUNT orders > 50000 : DROP EMPTY ELSE FILL EMPTY amount WITH 0 END"
        ast = parse(tokenize(source))
        node = ast[0]
        assert isinstance(node.condition, MetaRowCount)
        assert node.condition.op == ">"
        assert len(node.then_body) == 1
        assert len(node.else_body) == 1

    def test_if_with_logical_meta(self):
        source = "IF EXISTS a AND ROWCOUNT b > 0 : INFO END"
        node = parse(tokenize(source))[0]
        assert isinstance(node.condition, MetaAnd)

    def test_for(self):
        source = "FOR $col IN age, weight, bmi : CAST $col TO NUMBER (ON ERROR SKIP) FILL EMPTY $col WITH mean END"
        ast = parse(tokenize(source))
        node = ast[0]
        assert isinstance(node, AstFor)
        assert node.var == "col"
        assert node.columns == ["age", "weight", "bmi"]
        assert len(node.body) == 2

    def test_nested_blocks(self):
        source = "IF EXISTS x : FOR $c IN a, b : DROP EMPTY $c END END"
        ast = parse(tokenize(source))
        assert isinstance(ast[0], AstIf)
        assert isinstance(ast[0].then_body[0], AstFor)


class TestBlockErrors:
    def test_if_requires_colon(self):
        with pytest.raises(ParseError):
            parse(tokenize("IF EXISTS x INFO END"))

    def test_for_requires_loopvar(self):
        with pytest.raises(ParseError):
            parse(tokenize("FOR col IN a : INFO END"))

    def test_pipeline_requires_arrow(self):
        with pytest.raises(ParseError):
            parse(tokenize("orders FILTER WHERE a > 0"))
