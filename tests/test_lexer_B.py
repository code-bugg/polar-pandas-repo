"""
tests/test_lexer_B.py
=====================
Unit tests for Member B's lexer subtask:
  - KEYWORDS set completeness
  - _consume_word(): keyword recognition, booleans, identifiers
"""

import pytest
from lexer.lexer import TokenType, Token, LexError, _Lexer

def _patch_stubs(lex: _Lexer) -> None:
    lex._skip_whitespace = lambda: _do_skip_ws(lex)        # type: ignore
    lex._skip_comment    = lambda: _do_skip_comment(lex)   # type: ignore
    lex._consume_string  = lambda: _do_skip_string(lex)    # type: ignore
    lex._consume_number  = lambda: _do_skip_number(lex)    # type: ignore
    lex._consume_loopvar = lambda: _do_skip_loopvar(lex)   # type: ignore
    lex._consume_operator = lambda: _do_skip_operator(lex) # type: ignore
    lex._unknown_char    = lambda: _do_unknown(lex)        # type: ignore

def _do_skip_ws(lex: _Lexer) -> None:
    while not lex.at_end() and lex.peek() in (" ", "\t", "\r", "\n"):
        lex.advance()

def _do_skip_comment(lex: _Lexer) -> None:
    while not lex.at_end() and lex.peek() != "\n":
        lex.advance()

def _do_skip_string(lex: _Lexer) -> None:
    lex.advance()
    while not lex.at_end() and lex.peek() != '"':
        lex.advance()
    if not lex.at_end(): lex.advance()

def _do_skip_number(lex: _Lexer) -> None:
    while not lex.at_end() and (lex.peek() or "").isdigit():
        lex.advance()

def _do_skip_loopvar(lex: _Lexer) -> None:
    lex.advance()
    while not lex.at_end() and (lex.peek() or "").isalnum():
        lex.advance()

def _do_skip_operator(lex: _Lexer) -> None:
    lex.advance()
    if not lex.at_end() and lex.peek() in ("=", ">"):
        lex.advance()

def _do_unknown(lex: _Lexer) -> None:
    ch = lex.advance()
    raise LexError(f"Unexpected character '{ch}'", lex._line)

def _tokenize_patched(source: str) -> list[Token]:
    lex = _Lexer(source)
    _patch_stubs(lex)
    return lex._tokenize_all()

class TestKeywordsSet:
    def test_keywords_set_is_not_empty(self):
        assert len(_Lexer.KEYWORDS) > 0

    def test_keywords_at_least_66(self):
        assert len(_Lexer.KEYWORDS) >= 66

    def test_all_keywords_are_uppercase(self):
        for kw in _Lexer.KEYWORDS:
            assert kw == kw.upper()

    def test_true_false_not_in_keywords(self):
        assert "TRUE" not in _Lexer.KEYWORDS
        assert "FALSE" not in _Lexer.KEYWORDS

    @pytest.mark.parametrize("kw", [
        "LOAD", "EXPORT", "PREVIEW", "INFO", "DESCRIBE",
        "AS", "TO", "FROM", "IN",
        "SELECT", "COLUMNS", "FILTER", "WHERE",
        "DROP", "NULLS", "DUPLICATES", "FILL", "WITH",
        "CAST", "ADD", "COLUMN", "RENAME", "SORT", "BY",
        "GROUP", "COUNT", "JOIN", "ASC", "DESC", "PLOT",
        "SET", "ENGINE", "AND", "OR", "NOT",
    ])
    def test_core_keyword_present(self, kw):
        assert kw in _Lexer.KEYWORDS

    @pytest.mark.parametrize("kw", [
        "IF", "THEN", "ELSE", "END",
        "FOR", "EACH", "OVER", "DO",
        "EXISTS", "ROWCOUNT", "NULLCOUNT",
    ])
    def test_v11_keyword_present(self, kw):
        assert kw in _Lexer.KEYWORDS

class TestConsumeWordKeywords:
    def test_simple_keyword_uppercase(self):
        tokens = _tokenize_patched("LOAD")
        assert tokens[0] == Token(TokenType.KW, "LOAD", 1)

    def test_keyword_lowercase_normalised(self):
        tokens = _tokenize_patched("load")
        assert tokens[0] == Token(TokenType.KW, "LOAD", 1)

class TestConsumeWordBooleans:
    def test_true_lowercase(self):
        tokens = _tokenize_patched("true")
        assert tokens[0] == Token(TokenType.BOOL, "TRUE", 1)

    def test_false_uppercase(self):
        tokens = _tokenize_patched("FALSE")
        assert tokens[0] == Token(TokenType.BOOL, "FALSE", 1)

class TestConsumeWordIdentifiers:
    def test_simple_identifier(self):
        tokens = _tokenize_patched("df")
        assert tokens[0] == Token(TokenType.IDENT, "df", 1)

    def test_identifier_preserves_case(self):
        tokens = _tokenize_patched("myDataFrame")
        assert tokens[0].value == "myDataFrame"

class TestMaximalMunch:
    def test_keyword_prefix_not_split(self):
        tokens = _tokenize_patched("LOADING")
        assert tokens[0].type == TokenType.IDENT
        assert tokens[0].value == "LOADING"

class TestMultipleWords:
    def test_two_keywords_with_space(self):
        tokens = _tokenize_patched("DROP NULLS")
        assert tokens[0] == Token(TokenType.KW, "DROP", 1)
        assert tokens[1] == Token(TokenType.KW, "NULLS", 1)