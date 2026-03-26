"""
Test suite for Member D: Aggregation, Join, Plot, and Control Flow Parsers
"""
import pytest
from lexer.lexer import tokenize
from parser.parser import parse, AstGroup, AstCount, AstJoin, AstPlot, AstIf, AstFor, AstPipeline, CondCompare

class TestParseGroup:
    def test_group_basic(self):
        tokens = tokenize('GROUP sales BY region AGGREGATE amount AS sum')
        ast = parse(tokens)
        node = ast[0]
        assert isinstance(node, AstGroup)
        assert node.name == "sales"
        assert node.by == ["region"]
        assert node.agg == "sum"
        assert node.column == "amount"

    def test_group_short(self):
        tokens = tokenize('GROUP data BY month mean revenue')
        ast = parse(tokens)
        node = ast[0]
        assert isinstance(node, AstGroup)
        assert node.agg == "mean"
        assert node.column == "revenue"

class TestParseCount:
    def test_count_basic(self):
        tokens = tokenize('COUNT df GROUP BY region')
        ast = parse(tokens)
        node = ast[0]
        assert isinstance(node, AstCount)
        assert node.name == "df"
        assert node.by == ["region"]

class TestParseJoin:
    def test_join_basic(self):
        tokens = tokenize('JOIN customers WITH orders ON customer_id')
        ast = parse(tokens)
        node = ast[0]
        assert isinstance(node, AstJoin)
        assert node.left == "customers"
        assert node.right == "orders"
        assert node.on == "customer_id"
        assert node.how == "inner"

    def test_join_with_type_and_result(self):
        tokens = tokenize('JOIN a WITH b ON id TYPE LEFT AS combined')
        ast = parse(tokens)
        node = ast[0]
        assert node.how == "left"
        assert node.result == "combined"

class TestParsePlot:
    def test_plot_basic(self):
        tokens = tokenize('PLOT data TYPE bar X month Y sales')
        ast = parse(tokens)
        node = ast[0]
        assert isinstance(node, AstPlot)
        assert node.name == "data"
        assert node.kind == "bar"
        assert node.x == "month"
        assert node.y == "sales"

class TestParseIf:
    def test_if_basic(self):
        tokens = tokenize('''IF ROWCOUNT df > 1000 THEN
            COUNT df GROUP BY a
        END''')
        ast = parse(tokens)
        node = ast[0]
        assert isinstance(node, AstIf)
        assert len(node.then_body) == 1
        assert isinstance(node.condition, CondCompare)
        assert node.condition.op == "rowcount_>"

    def test_if_exists(self):
        tokens = tokenize('''IF EXISTS df THEN
            INFO df
        END''')
        ast = parse(tokens)
        assert isinstance(ast[0].condition, CondCompare)
        assert ast[0].condition.op == "exists"

class TestParseFor:
    def test_for_basic(self):
        tokens = tokenize('''FOR EACH $col OVER [amount, revenue] DO
            CAST df COLUMN $col TO float
        END''')
        ast = parse(tokens)
        node = ast[0]
        assert isinstance(node, AstFor)
        assert node.var == "col"
        assert node.columns == ["amount", "revenue"]
        assert len(node.body) == 1

class TestParsePipeline:
    def test_pipeline_basic(self):
        tokens = tokenize('df -> FILTER WHERE amount > 0 -> SORT BY age DESC')
        ast = parse(tokens)
        node = ast[0]
        assert isinstance(node, AstPipeline)
        assert node.name == "df"
        assert len(node.steps) == 2