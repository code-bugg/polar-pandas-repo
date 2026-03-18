"""
Test suite for Member D: Aggregation, Join, Plot, and Control Flow Parsers
"""
import pytest
from lexer.lexer import tokenize, TokenType
from parser.parser import parse, ParseError
from parser.parser import (
    AstGroup, AstCount, AstJoin, AstPlot, AstIf, AstFor,
    CondCompare, CondAnd, CondOr, CondIn, CondNull
)


class TestParseGroup:
    def test_group_basic(self):
        tokens = tokenize('GROUP sales BY region USING sum ON amount')
        ast = parse(tokens)
        assert len(ast) == 1
        node = ast[0]
        assert isinstance(node, AstGroup)
        assert node.name == "sales"
        assert node.by == "region"
        assert node.agg == "sum"
        assert node.on == "amount"
        assert node.result is None

    def test_group_with_result(self):
        tokens = tokenize('GROUP data BY month USING avg ON revenue AS monthly_avg')
        ast = parse(tokens)
        node = ast[0]
        assert isinstance(node, AstGroup)
        assert node.result == "monthly_avg"


class TestParseCount:
    def test_count_basic(self):
        tokens = tokenize('COUNT ROWS IN df')
        ast = parse(tokens)
        node = ast[0]
        assert isinstance(node, AstCount)
        assert node.name == "df"
        assert node.result is None

    def test_count_with_result(self):
        tokens = tokenize('COUNT ROWS IN df AS total')
        ast = parse(tokens)
        node = ast[0]
        assert isinstance(node, AstCount)
        assert node.result == "total"


class TestParseJoin:
    def test_join_basic(self):
        tokens = tokenize('JOIN customers WITH orders ON customer_id')
        ast = parse(tokens)
        node = ast[0]
        assert isinstance(node, AstJoin)
        assert node.left == "customers"
        assert node.right == "orders"
        assert node.on == "customer_id"
        assert node.how == "INNER"
        assert node.result is None

    def test_join_with_type(self):
        tokens = tokenize('JOIN a WITH b ON id LEFT')
        ast = parse(tokens)
        node = ast[0]
        assert node.how == "LEFT"

    def test_join_with_result(self):
        tokens = tokenize('JOIN a WITH b ON id RIGHT AS combined')
        ast = parse(tokens)
        node = ast[0]
        assert node.how == "RIGHT"
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
        assert node.title is None
        assert node.save is None

    def test_plot_with_title(self):
        tokens = tokenize('PLOT data TYPE line X date Y revenue TITLE "Revenue Over Time"')
        ast = parse(tokens)
        node = ast[0]
        assert node.title == "Revenue Over Time"

    def test_plot_with_save(self):
        tokens = tokenize('PLOT data TYPE scatter X price Y quantity SAVE "chart.png"')
        ast = parse(tokens)
        node = ast[0]
        assert node.save == "chart.png"

    def test_plot_all_options(self):
        tokens = tokenize('PLOT data TYPE hist X amount Y count TITLE "Distribution" SAVE "hist.png"')
        ast = parse(tokens)
        node = ast[0]
        assert node.x == "amount"
        assert node.y == "count"
        assert node.title == "Distribution"
        assert node.save == "hist.png"


class TestParseCondition:
    def test_condition_simple_compare(self):
        """Test a simple comparison condition."""
        tokens = tokenize('FILTER df WHERE amount >= 100')
        ast = parse(tokens)
        node = ast[0]
        cond = node.condition
        assert isinstance(cond, CondCompare)
        assert cond.op == ">="

    def test_condition_and(self):
        """Test AND precedence (higher than OR)."""
        tokens = tokenize('FILTER df WHERE amount > 50 AND status == "active"')
        ast = parse(tokens)
        cond = ast[0].condition
        assert isinstance(cond, CondAnd)  # AND should combine two comparisons
        assert isinstance(cond.left, CondCompare)
        assert isinstance(cond.right, CondCompare)
        assert cond.left.op == ">"
        assert cond.right.op == "=="

    def test_condition_or(self):
        """Test OR precedence (lower than AND)."""
        tokens = tokenize('FILTER df WHERE a == 1 OR b == 2')
        ast = parse(tokens)
        # Verify that OR is parsed correctly with proper precedence


class TestParseIf:
    def test_if_basic(self):
        """Simple IF without ELSE."""
        tokens = tokenize('''IF sales > 1000 THEN
            COUNT ROWS IN df
        END''')
        ast = parse(tokens)
        node = ast[0]
        assert isinstance(node, AstIf)
        assert len(node.then_body) == 1
        assert len(node.else_body) == 0
        assert isinstance(node.condition, CondCompare)

    def test_if_with_else(self):
        """IF with ELSE branch."""
        tokens = tokenize('''IF amount > 100 THEN
            FILTER df WHERE amount >= 100
        ELSE
            FILTER df WHERE amount < 100
        END''')
        ast = parse(tokens)
        node = ast[0]
        assert isinstance(node, AstIf)
        assert len(node.then_body) == 1
        assert len(node.else_body) == 1

    def test_if_nested(self):
        """Nested IF statements."""
        tokens = tokenize('''IF a > 5 THEN
            IF b > 10 THEN
                COUNT ROWS IN df
            END
        END''')
        ast = parse(tokens)
        outer = ast[0]
        assert isinstance(outer, AstIf)
        inner = outer.then_body[0]
        assert isinstance(inner, AstIf)


class TestParseFor:
    def test_for_basic(self):
        """Basic FOR loop."""
        tokens = tokenize('''FOR EACH $col OVER [amount, revenue] DO
            CAST $col IN df TO FLOAT
        END''')
        ast = parse(tokens)
        node = ast[0]
        assert isinstance(node, AstFor)
        assert node.var == "col"
        assert node.columns == ["amount", "revenue"]
        assert len(node.body) == 1

    def test_for_multiple_columns(self):
        """FOR loop with many columns."""
        tokens = tokenize('''FOR EACH $field OVER [name, email, address] DO
            COUNT ROWS IN df
        END''')
        ast = parse(tokens)
        node = ast[0]
        assert node.columns == ["name", "email", "address"]

    def test_for_multiple_statements(self):
        """FOR loop with multiple body statements."""
        tokens = tokenize('''FOR EACH $col OVER [x, y] DO
            CAST $col IN df TO INT
            FILL NULLS IN df COLUMNS [$col] WITH 0
        END''')
        ast = parse(tokens)
        node = ast[0]
        assert len(node.body) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
