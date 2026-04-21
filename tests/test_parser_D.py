import pytest

from lexer.lexer import tokenize
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
