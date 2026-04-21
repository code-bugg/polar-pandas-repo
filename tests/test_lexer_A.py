import pytest

from lexer.lexer import LexError, Token, TokenType, _Lexer


class TestTokenTypeEnum:
    def test_has_exactly_15_members(self):
        assert len(TokenType) == 15

    def test_all_expected_members_exist(self):
        expected = {
            "KW",
            "IDENT",
            "STRING",
            "NUMBER",
            "BOOL",
            "OP",
            "ARITH_OP",
            "LPAREN",
            "RPAREN",
            "COMMA",
            "ASSIGN",
            "PIPE",
            "COLON",
            "LOOPVAR",
            "EOF",
        }
        assert {m.name for m in TokenType} == expected


class TestTokenAndError:
    def test_token_frozen(self):
        t = Token(TokenType.KW, "READ", 1)
        with pytest.raises((AttributeError, TypeError)):
            t.value = "SAVE"  # type: ignore

    def test_lexerror_stores_line(self):
        err = LexError("bad", 9)
        assert err.line == 9
        assert "9" in str(err)


class TestCursor:
    def test_peek_and_advance(self):
        lex = _Lexer("ab")
        assert lex.peek() == "a"
        assert lex.advance() == "a"
        assert lex.peek() == "b"

    def test_advance_tracks_newline(self):
        lex = _Lexer("a\n")
        lex.advance()
        lex.advance()
        assert lex._line == 2


class TestDispatcherSingleChar:
    def test_paren_comma_colon(self):
        tokens = _Lexer("(),:")._tokenize_all()
        assert [t.type for t in tokens[:-1]] == [
            TokenType.LPAREN,
            TokenType.RPAREN,
            TokenType.COMMA,
            TokenType.COLON,
        ]

    def test_pipe_and_arith(self):
        tokens = _Lexer("->+-*/")._tokenize_all()
        assert [t.type for t in tokens[:-1]] == [
            TokenType.PIPE,
            TokenType.ARITH_OP,
            TokenType.ARITH_OP,
            TokenType.ARITH_OP,
            TokenType.ARITH_OP,
        ]

    def test_square_brackets_raise(self):
        with pytest.raises(LexError, match="Square brackets"):
            _Lexer("[")._tokenize_all()

    def test_eof_last(self):
        tokens = _Lexer("READ")._tokenize_all()
        assert tokens[-1].type == TokenType.EOF
