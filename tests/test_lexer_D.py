import pytest
from lexer.lexer import tokenize, LexError, TokenType

class TestWhitespace:
    def test_skip_spaces(self):
        tokens = tokenize("LOAD   \"data.csv\"")
        assert tokens[0].value == "LOAD"
        assert tokens[1].value == '"data.csv"'

    def test_skip_tabs_and_newlines(self):
        tokens = tokenize("LOAD\t\"x.csv\"\nAS\tdf")
        assert [t.value for t in tokens if t.type != TokenType.EOF] == [
            "LOAD", '"x.csv"', "AS", "df"
        ]

class TestComments:
    def test_line_comment(self):
        tokens = tokenize("LOAD \"data.csv\" # Load the input file")
        assert tokens[0].value == "LOAD"
        assert tokens[2].type == TokenType.EOF

class TestOperators:
    def test_equals_vs_double_equals(self):
        tokens2 = tokenize("a == b")
        assert tokens2[1].value == "=="

    def test_pipeline_operator(self):
        tokens = tokenize("df -> FILTER")
        assert tokens[1].type == TokenType.PIPE
        assert tokens[1].value == "->"

    def test_greater_equal(self):
        tokens = tokenize("x >= 5")
        assert tokens[1].value == ">="
        assert tokens[1].type == TokenType.OP

class TestErrorHandling:
    def test_bare_exclamation(self):
        with pytest.raises(LexError, match="did you mean '!='"):
            tokenize("if ! x")

    def test_unknown_char_pipe(self):
        with pytest.raises(LexError):
            tokenize("LOAD x | FILTER")

    def test_backticks_error(self):
        with pytest.raises(LexError, match="backticks"):
            tokenize("SELECT COLUMNS `col` FROM df")

    def test_semicolon_error(self):
        with pytest.raises(LexError, match="semicolons"):
            tokenize("LOAD x;")

class TestIntegration:
    def test_complex_statement_with_comments(self):
        source = """
        LOAD "data.csv" AS df  # Load data
        WHERE amount >= 100    # Filter by amount
        """
        tokens = tokenize(source)
        assert tokens[-1].type == TokenType.EOF

    def test_all_operators_in_expression(self):
        tokens = tokenize("x == 1 AND y != 2 AND z >= 3 AND w <= 4")
        ops = [t.value for t in tokens if t.type == TokenType.OP]
        assert ops == ["==", "!=", ">=", "<="]