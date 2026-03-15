"""
tests/test_lexer_A.py
=====================
Unit tests for Member A's lexer subtask:
  - TokenType enum completeness
  - Token dataclass behaviour
  - LexError construction
  - Single-character token dispatch ([ ] , * + - / %)
  - at_end() and peek() cursor behaviour
  - EOF sentinel is always last token

NOTE: Tests that require Members B/C/D stubs will raise NotImplementedError.
      Those are tested in test_lexer_B.py, test_lexer_C.py, test_lexer_D.py.
      This file only tests what Member A owns.
"""

import pytest
from lexer.lexer import TokenType, Token, LexError, tokenize, _Lexer


# ─────────────────────────────────────────────────────────────────────────────
# TokenType enum
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenTypeEnum:

    def test_has_exactly_14_members(self):
        assert len(TokenType) == 14

    def test_all_expected_members_exist(self):
        expected = {
            "KW", "IDENT", "STRING", "INTEGER", "FLOAT", "BOOL",
            "OP", "ARITH_OP", "LBRACKET", "RBRACKET", "COMMA",
            "STAR", "LOOPVAR", "EOF",
        }
        actual = {m.name for m in TokenType}
        assert actual == expected

    def test_members_are_unique(self):
        values = [m.value for m in TokenType]
        assert len(values) == len(set(values))

    def test_enum_members_are_not_strings(self):
        # TokenType uses auto() — values are ints, not strings
        for member in TokenType:
            assert isinstance(member.value, int)


# ─────────────────────────────────────────────────────────────────────────────
# Token dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenDataclass:

    def test_basic_construction(self):
        t = Token(TokenType.KW, "LOAD", 1)
        assert t.type  == TokenType.KW
        assert t.value == "LOAD"
        assert t.line  == 1

    def test_is_frozen(self):
        t = Token(TokenType.IDENT, "df", 3)
        with pytest.raises((AttributeError, TypeError)):
            t.value = "other"   # type: ignore

    def test_equality_by_value(self):
        t1 = Token(TokenType.COMMA, ",", 5)
        t2 = Token(TokenType.COMMA, ",", 5)
        assert t1 == t2

    def test_inequality_different_line(self):
        t1 = Token(TokenType.COMMA, ",", 5)
        t2 = Token(TokenType.COMMA, ",", 6)
        assert t1 != t2

    def test_repr_contains_type_and_value(self):
        t = Token(TokenType.INTEGER, "42", 2)
        r = repr(t)
        assert "INTEGER" in r
        assert "42"      in r
        assert "line=2"  in r

    def test_hashable(self):
        # frozen dataclasses should be hashable
        t = Token(TokenType.EOF, "", 1)
        s = {t}
        assert t in s


# ─────────────────────────────────────────────────────────────────────────────
# LexError
# ─────────────────────────────────────────────────────────────────────────────

class TestLexError:

    def test_is_exception(self):
        err = LexError("test error", 7)
        assert isinstance(err, Exception)

    def test_stores_message_and_line(self):
        err = LexError("unexpected char '@'", 12)
        assert err.message == "unexpected char '@'"
        assert err.line    == 12

    def test_str_contains_line_number(self):
        err = LexError("bad token", 3)
        assert "3" in str(err)

    def test_str_contains_lexer_label(self):
        err = LexError("bad token", 3)
        assert "Lexer" in str(err) or "lexer" in str(err).lower()


# ─────────────────────────────────────────────────────────────────────────────
# _Lexer cursor primitives
# ─────────────────────────────────────────────────────────────────────────────

class TestLexerCursor:

    def test_peek_returns_first_char(self):
        lex = _Lexer("abc")
        assert lex.peek() == "a"

    def test_peek_with_offset(self):
        lex = _Lexer("abc")
        assert lex.peek(0) == "a"
        assert lex.peek(1) == "b"
        assert lex.peek(2) == "c"

    def test_peek_past_end_returns_none(self):
        lex = _Lexer("a")
        assert lex.peek(1) is None
        assert lex.peek(999) is None

    def test_advance_returns_char(self):
        lex = _Lexer("xy")
        assert lex.advance() == "x"
        assert lex.advance() == "y"

    def test_advance_increments_position(self):
        lex = _Lexer("ab")
        lex.advance()
        assert lex.peek() == "b"

    def test_advance_tracks_newlines(self):
        lex = _Lexer("a\nb")
        assert lex._line == 1
        lex.advance()   # 'a'
        lex.advance()   # '\n'  — should bump line
        assert lex._line == 2

    def test_at_end_false_when_not_exhausted(self):
        lex = _Lexer("x")
        assert not lex.at_end()

    def test_at_end_true_after_full_consumption(self):
        lex = _Lexer("x")
        lex.advance()
        assert lex.at_end()

    def test_at_end_true_on_empty_source(self):
        lex = _Lexer("")
        assert lex.at_end()

    def test_advance_at_end_raises_lex_error(self):
        lex = _Lexer("")
        with pytest.raises(LexError):
            lex.advance()


# ─────────────────────────────────────────────────────────────────────────────
# Single-character tokens (Member A's dispatcher code)
# We patch the stubs so whitespace and words don't trigger NotImplementedError.
# ─────────────────────────────────────────────────────────────────────────────

def _patch_stubs(lex: _Lexer) -> None:
    """Monkeypatch all stubs to skip silently so we can test single-char tokens."""
    def _noop(self=None):
        raise LexError("unexpected in this test", lex._line)
    lex._skip_whitespace = lambda: None   # type: ignore
    lex._skip_comment    = lambda: None   # type: ignore


class TestSingleCharTokens:

    def _lex_single(self, source: str) -> list[Token]:
        """Tokenise a single-character source, bypassing stubs."""
        lex = _Lexer(source)
        _patch_stubs(lex)
        return lex._tokenize_all()

    def test_lbracket(self):
        tokens = self._lex_single("[")
        assert tokens[0].type  == TokenType.LBRACKET
        assert tokens[0].value == "["

    def test_rbracket(self):
        tokens = self._lex_single("]")
        assert tokens[0].type  == TokenType.RBRACKET
        assert tokens[0].value == "]"

    def test_comma(self):
        tokens = self._lex_single(",")
        assert tokens[0].type  == TokenType.COMMA
        assert tokens[0].value == ","

    def test_star(self):
        tokens = self._lex_single("*")
        assert tokens[0].type  == TokenType.STAR
        assert tokens[0].value == "*"

    def test_plus(self):
        tokens = self._lex_single("+")
        assert tokens[0].type  == TokenType.ARITH_OP
        assert tokens[0].value == "+"

    def test_minus(self):
        tokens = self._lex_single("-")
        assert tokens[0].type  == TokenType.ARITH_OP
        assert tokens[0].value == "-"

    def test_slash(self):
        tokens = self._lex_single("/")
        assert tokens[0].type  == TokenType.ARITH_OP
        assert tokens[0].value == "/"

    def test_percent(self):
        tokens = self._lex_single("%")
        assert tokens[0].type  == TokenType.ARITH_OP
        assert tokens[0].value == "%"

    def test_single_char_line_number(self):
        tokens = self._lex_single(",")
        assert tokens[0].line == 1

    def test_eof_always_last(self):
        tokens = self._lex_single(",")
        assert tokens[-1].type == TokenType.EOF

    def test_eof_on_empty_source(self):
        lex = _Lexer("")
        tokens = lex._tokenize_all()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_multiple_single_chars(self):
        lex = _Lexer("[,]")
        _patch_stubs(lex)
        tokens = lex._tokenize_all()
        types = [t.type for t in tokens]
        assert types == [
            TokenType.LBRACKET,
            TokenType.COMMA,
            TokenType.RBRACKET,
            TokenType.EOF,
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher routing — confirm stubs are called for B/C/D characters
# ─────────────────────────────────────────────────────────────────────────────

class TestDispatcherRouting:
    """Verify the dispatcher calls the right stub for each character class.
    We replace each stub with a sentinel function and check it was invoked."""

    def _lex_with_sentinel(self, source: str, stub_name: str) -> bool:
        called = []
        lex = _Lexer(source)
        # Replace whitespace/comment stubs to avoid interference
        lex._skip_whitespace = lambda: (lex.advance(), None)[1]  # type: ignore
        lex._skip_comment    = lambda: (lex.advance(), None)[1]  # type: ignore

        def sentinel():
            called.append(True)
            # consume one char to prevent infinite loop
            if not lex.at_end():
                lex.advance()

        setattr(lex, stub_name, sentinel)
        try:
            lex._tokenize_all()
        except NotImplementedError:
            pass  # other stubs may still be unimplemented
        return bool(called)

    def test_alpha_routes_to_consume_word(self):
        assert self._lex_with_sentinel("abc", "_consume_word")

    def test_digit_routes_to_consume_number(self):
        assert self._lex_with_sentinel("123", "_consume_number")

    def test_quote_routes_to_consume_string(self):
        assert self._lex_with_sentinel('"hi"', "_consume_string")

    def test_dollar_routes_to_consume_loopvar(self):
        assert self._lex_with_sentinel("$col", "_consume_loopvar")

    def test_hash_routes_to_skip_comment(self):
        assert self._lex_with_sentinel("# comment", "_skip_comment")

    def test_space_routes_to_skip_whitespace(self):
        called = []
        lex = _Lexer(" ")
        def sentinel():
            called.append(True)
            lex.advance()
        lex._skip_whitespace = sentinel  # type: ignore
        try:
            lex._tokenize_all()
        except NotImplementedError:
            pass
        assert bool(called)

    def test_operator_routes_to_consume_operator(self):
        assert self._lex_with_sentinel(">=", "_consume_operator")