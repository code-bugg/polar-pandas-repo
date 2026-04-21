"""PolarPandas v5 lexer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class TokenType(Enum):
    KW = auto()
    IDENT = auto()
    STRING = auto()
    NUMBER = auto()
    BOOL = auto()
    OP = auto()
    ARITH_OP = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    ASSIGN = auto()
    PIPE = auto()
    COLON = auto()
    LOOPVAR = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    line: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, line={self.line})"


class LexError(Exception):
    def __init__(self, message: str, line: int) -> None:
        super().__init__(f"[Lexer] line {line}: {message}")
        self.message = message
        self.line = line


class _Lexer:
    """Internal stateful lexer."""

    KEYWORDS: set[str] = {
        "READ",
        "SAVE",
        "PREVIEW",
        "INFO",
        "DROP",
        "EMPTY",
        "DUPLICATES",
        "FILL",
        "WITH",
        "KEEP",
        "CAST",
        "SET",
        "RENAME",
        "TO",
        "NUMBER",
        "TEXT",
        "DATE",
        "DECIMAL",
        "BOOLEAN",
        "SORT",
        "BY",
        "ASC",
        "DESC",
        "GROUP",
        "MERGE",
        "ON",
        "FILTER",
        "WHERE",
        "AND",
        "OR",
        "NOT",
        "IS",
        "BETWEEN",
        "CONTAINS",
        "IF",
        "ELSE",
        "END",
        "FOR",
        "IN",
        "EXISTS",
        "ROWCOUNT",
        "ERROR",
        "SKIP",
        "STOP",
        "PLOT",
        "TYPE",
        "X",
        "Y",
        "TITLE",
    }

    _BOOLEANS: set[str] = {"TRUE", "FALSE"}

    def __init__(self, source: str) -> None:
        self._source = source
        self._pos = 0
        self._line = 1
        self._tokens: list[Token] = []

    def peek(self, offset: int = 0) -> Optional[str]:
        idx = self._pos + offset
        return self._source[idx] if idx < len(self._source) else None

    def advance(self) -> str:
        if self._pos >= len(self._source):
            raise LexError("Unexpected end of source", self._line)
        ch = self._source[self._pos]
        self._pos += 1
        if ch == "\n":
            self._line += 1
        return ch

    def at_end(self) -> bool:
        return self._pos >= len(self._source)

    def _emit(self, ttype: TokenType, value: str, line: int) -> None:
        self._tokens.append(Token(ttype, value, line))

    def _next_token(self) -> None:
        ch = self.peek()

        if ch in (" ", "\t", "\r", "\n"):
            self._skip_whitespace()
            return
        if ch == "#":
            self._skip_comment()
            return

        if ch == "$":
            self._consume_loopvar()
            return

        if ch in ('"', "'"):
            self._consume_string()
            return

        if ch is not None and ch.isdigit():
            self._consume_number()
            return

        if ch is not None and (ch.isalpha() or ch == "_"):
            self._consume_word()
            return

        if ch in ("=", "!", ">", "<"):
            self._consume_operator()
            return

        line = self._line

        if ch == "-" and self.peek(1) == ">":
            self.advance()
            self.advance()
            self._emit(TokenType.PIPE, "->", line)
            return

        if ch in ("+", "-", "*", "/"):
            self.advance()
            self._emit(TokenType.ARITH_OP, ch, line)
            return

        if ch == "(":
            self.advance()
            self._emit(TokenType.LPAREN, "(", line)
            return

        if ch == ")":
            self.advance()
            self._emit(TokenType.RPAREN, ")", line)
            return

        if ch == ",":
            self.advance()
            self._emit(TokenType.COMMA, ",", line)
            return

        if ch == ":":
            self.advance()
            self._emit(TokenType.COLON, ":", line)
            return

        self._unknown_char()

    def _tokenize_all(self) -> list[Token]:
        while not self.at_end():
            self._next_token()
        self._emit(TokenType.EOF, "", self._line)
        return self._tokens

    def _consume_word(self) -> None:
        start_line = self._line
        start_pos = self._pos

        while not self.at_end():
            ch = self.peek()
            if ch is not None and (ch.isalnum() or ch == "_"):
                self.advance()
            else:
                break

        word = self._source[start_pos:self._pos]
        upper = word.upper()

        if upper in self._BOOLEANS:
            self._emit(TokenType.BOOL, upper.lower(), start_line)
            return

        if upper in self.KEYWORDS:
            self._emit(TokenType.KW, upper, start_line)
            return

        self._emit(TokenType.IDENT, word, start_line)

    def _consume_string(self) -> None:
        start_line = self._line
        quote = self.advance()
        chars: list[str] = []

        while True:
            ch = self.peek()
            if ch is None or ch == "\n":
                raise LexError("Unterminated string literal", start_line)

            if ch == quote:
                self.advance()
                self._emit(TokenType.STRING, f"{quote}{''.join(chars)}{quote}", start_line)
                return

            if ch == "\\":
                self.advance()
                esc = self.peek()
                if esc is None or esc == "\n":
                    raise LexError("Unterminated escape sequence in string", start_line)

                if esc == "n":
                    self.advance()
                    chars.append("\n")
                    continue
                if esc == "t":
                    self.advance()
                    chars.append("\t")
                    continue
                if esc == "\\":
                    self.advance()
                    chars.append("\\")
                    continue
                if esc == quote:
                    self.advance()
                    chars.append(quote)
                    continue
                if esc == '"':
                    self.advance()
                    chars.append('"')
                    continue

                raise LexError(f"Invalid escape sequence '\\{esc}' in string literal", self._line)

            chars.append(self.advance())

    def _consume_number(self) -> None:
        start_line = self._line
        start_pos = self._pos

        while (self.peek() or "").isdigit():
            self.advance()

        if self.peek() == "." and (self.peek(1) or "").isdigit():
            self.advance()
            while (self.peek() or "").isdigit():
                self.advance()

        self._emit(TokenType.NUMBER, self._source[start_pos:self._pos], start_line)

    def _consume_loopvar(self) -> None:
        start_line = self._line
        self.advance()

        first = self.peek()
        if first is None or not (first.isalpha() or first == "_"):
            raise LexError("'$' must be followed by a letter or underscore", start_line)

        start_pos = self._pos
        while not self.at_end():
            ch = self.peek()
            if ch is not None and (ch.isalnum() or ch == "_"):
                self.advance()
            else:
                break

        self._emit(TokenType.LOOPVAR, f"${self._source[start_pos:self._pos]}", start_line)

    def _skip_whitespace(self) -> None:
        while self.peek() in (" ", "\t", "\r", "\n"):
            self.advance()

    def _skip_comment(self) -> None:
        self.advance()
        while self.peek() is not None and self.peek() != "\n":
            self.advance()

    def _consume_operator(self) -> None:
        line = self._line
        ch = self.peek()

        if ch == "=":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, "==", line)
            else:
                self._emit(TokenType.ASSIGN, "=", line)
            return

        if ch == "!":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, "!=", line)
            else:
                raise LexError("Unexpected '!'; did you mean '!='?", line)
            return

        if ch == ">":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, ">=", line)
            else:
                self._emit(TokenType.OP, ">", line)
            return

        if ch == "<":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, "<=", line)
            else:
                self._emit(TokenType.OP, "<", line)
            return

    def _unknown_char(self) -> None:
        ch = self.peek()
        line = self._line

        hints = {
            "[": "Square brackets are not part of PolarPandas v5. Use comma-separated values without brackets.",
            "]": "Square brackets are not part of PolarPandas v5. Use comma-separated values without brackets.",
            ".": "Unexpected '.'. Dot notation is not supported by the lexer.",
            "|": "Unexpected '|'. Use '->' for pipeline or OR for logical disjunction.",
            "&": "Unexpected '&'. Use AND for logical conjunction.",
            ";": "Semicolons are not used in PolarPandas.",
            "`": "Backticks are not supported.",
            "{": "Braces are not supported in PolarPandas.",
            "}": "Braces are not supported in PolarPandas.",
            "^": "Unexpected '^'.",
            "~": "Unexpected '~'.",
        }

        msg = hints.get(ch, f"unexpected character {ch!r}")
        raise LexError(msg, line)


def tokenize(source: str) -> list[Token]:
    return _Lexer(source)._tokenize_all()
