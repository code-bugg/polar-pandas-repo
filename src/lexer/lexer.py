"""
polarpandas.lexer
=================
PolarPandas Translator — Component 1: Lexer
Member A subtask: TokenType enum, Token dataclass, tokenize() entry point,
                  and the maximal-munch character dispatcher.

Subtasks B, C, D will each add their consume_*() helpers into this same
file. The public interface they must respect is:

    tokenize(source: str) -> list[Token]

and the shared types defined here:

    TokenType  (enum)
    Token      (dataclass)
    LexError   (exception)          <- implemented by Member D

The dispatcher (_next_token) is the only place that decides *which*
consume_*() function to call. Members B, C, D implement those functions;
Member A wires them in here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. TOKEN TYPES
#    Spec reference: §2.1, §5.2 (LOOPVAR added in v1.1)
#    14 structural token types total.
# ─────────────────────────────────────────────────────────────────────────────

class TokenType(Enum):
    # ── Lexical categories ────────────────────────────────────────────────
    KW        = auto()   # Reserved keyword (case-insensitive). e.g. LOAD, WHERE, IF
    IDENT     = auto()   # User identifier (dataframe variable name). e.g. df, sales_df
    STRING    = auto()   # Double-quoted string literal. e.g. "amount", "data.csv"
    INTEGER   = auto()   # Integer literal.  e.g. 5, 100
    FLOAT     = auto()   # Float literal.    e.g. 3.14, 0.5
    BOOL      = auto()   # Boolean literal.  true | false  (case-insensitive)
    OP        = auto()   # Comparison operator: == != > < >= <=
    ARITH_OP  = auto()   # Arithmetic operator: + - * / %  (also see STAR)
    LBRACKET  = auto()   # [
    RBRACKET  = auto()   # ]
    COMMA     = auto()   # ,
    STAR      = auto()   # *  used as wildcard in SELECT … COLUMNS *
    LOOPVAR   = auto()   # $identifier  loop variable in FOR EACH $col OVER … (v1.1)
    EOF       = auto()   # Sentinel: end of input


# ─────────────────────────────────────────────────────────────────────────────
# 2. TOKEN DATACLASS
#    Immutable value type.  Members B/C/D construct Token instances using
#    the same class — do not add mutable state here.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Token:
    """A single lexical unit produced by the lexer.

    Attributes
    ----------
    type  : TokenType
        The category of the token.
    value : str
        The raw source text for this token.
        Keywords are stored in UPPER CASE (normalised by Member B).
        String literals include the enclosing double-quote characters.
        LOOPVAR tokens include the leading ``$``.
    line  : int
        1-based source line number where the token starts.
    """
    type:  TokenType
    value: str
    line:  int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, line={self.line})"


# ─────────────────────────────────────────────────────────────────────────────
# 3. LEXER ERROR
#    Stub — Member D will fill in the body (collect-all error reporting).
#    The class must exist here so the dispatcher can raise it.
# ─────────────────────────────────────────────────────────────────────────────

class LexError(Exception):
    """Raised when the lexer encounters an unrecognised character or
    malformed token (e.g. unterminated string, bare ``$`` not followed
    by an identifier).

    Attributes
    ----------
    message : str
        Human-readable description of the problem.
    line    : int
        1-based source line where the error occurred.
    """
    def __init__(self, message: str, line: int) -> None:
        super().__init__(f"[Lexer] line {line}: {message}")
        self.message = message
        self.line    = line


# ─────────────────────────────────────────────────────────────────────────────
# 4. LEXER STATE
#    _Lexer encapsulates mutable cursor state so the stateless public
#    function tokenize() can be called multiple times safely.
# ─────────────────────────────────────────────────────────────────────────────

class _Lexer:
    """Internal mutable lexer state.

    The public entry point is ``tokenize(source)`` which constructs one
    ``_Lexer`` instance per call and returns its token list.

    Members B, C, D add their consume_*() methods directly to this class.
    Member A owns:
        - __init__       (cursor initialisation)
        - peek()         (non-consuming lookahead)
        - advance()      (consuming one character, updating line counter)
        - _next_token()  (the maximal-munch dispatcher)
        - tokenize()     (the main loop)
    """

    def __init__(self, source: str) -> None:
        self._source: str        = source
        self._pos:    int        = 0       # current character index
        self._line:   int        = 1       # current 1-based line number
        self._tokens: list[Token] = []

    # ── Cursor primitives ─────────────────────────────────────────────────

    def peek(self, offset: int = 0) -> Optional[str]:
        """Return the character at ``_pos + offset`` without advancing.
        Returns None if past end-of-source.
        """
        idx = self._pos + offset
        return self._source[idx] if idx < len(self._source) else None

    def advance(self) -> str:
        """Consume and return the current character, updating the line
        counter when a newline is consumed.

        Raises
        ------
        LexError
            If called at end-of-source (should not happen in normal flow;
            callers must check peek() first).
        """
        if self._pos >= len(self._source):
            raise LexError("Unexpected end of source", self._line)
        ch = self._source[self._pos]
        self._pos += 1
        if ch == "\n":
            self._line += 1
        return ch

    def at_end(self) -> bool:
        """True when the cursor is at or past the end of the source."""
        return self._pos >= len(self._source)

    # ── Token emission helper ─────────────────────────────────────────────

    def _emit(self, ttype: TokenType, value: str, line: int) -> Token:
        """Construct a Token and append it to the internal list."""
        tok = Token(ttype, value, line)
        self._tokens.append(tok)
        return tok

    # ── Maximal-munch dispatcher ──────────────────────────────────────────

    def _next_token(self) -> None:
        """Classify and consume the next token from the source.

        This is the single dispatch point for all token types.
        It is called repeatedly by ``_tokenize_all()`` until EOF.

        Dispatch order matters for maximal-munch correctness:
            1. Whitespace / comments  (Member D) — skip silently
            2. LOOPVAR  $ident        (Member C) — must check before IDENT
            3. STRING   "…"           (Member C)
            4. NUMBER   digit…        (Member C) — FLOAT before INTEGER
            5. WORD     alpha…        (Member B) — KW / BOOL / IDENT
            6. OPERATOR multi-char    (Member D) — >= before >, == before =
            7. Single-char tokens     (Member A) — [ ] , * and ARITH_OP
            8. Unknown character      (Member D) — LexError
        """
        ch = self.peek()

        # ── 1. Whitespace and comments (Member D fills in) ────────────────
        if ch in (" ", "\t", "\r", "\n"):
            self._skip_whitespace()       # Member D
            return
        if ch == "#":
            self._skip_comment()          # Member D
            return

        # ── 2. LOOPVAR  $ident (Member C fills in) ────────────────────────
        if ch == "$":
            self._consume_loopvar()       # Member C
            return

        # ── 3. STRING literal (Member C fills in) ─────────────────────────
        if ch == '"':
            self._consume_string()        # Member C
            return

        # ── 4. NUMBER literal (Member C fills in) ─────────────────────────
        if ch is not None and ch.isdigit():
            self._consume_number()        # Member C
            return

        # ── 5. WORD: keyword, boolean, or identifier (Member B fills in) ──
        if ch is not None and (ch.isalpha() or ch == "_"):
            self._consume_word()          # Member B
            return

        # ── 6. Multi-character operators (Member D fills in) ──────────────
        if ch in ("=", "!", ">", "<"):
            self._consume_operator()      # Member D
            return

        # ── 7. Single-character tokens (Member A — implemented below) ─────
        line = self._line
        if ch == "[":
            self.advance()
            self._emit(TokenType.LBRACKET, "[", line)
            return
        if ch == "]":
            self.advance()
            self._emit(TokenType.RBRACKET, "]", line)
            return
        if ch == ",":
            self.advance()
            self._emit(TokenType.COMMA, ",", line)
            return
        if ch == "*":
            # Contextual disambiguation (STAR vs ARITH_OP) is resolved by
            # the parser, not the lexer.  The lexer always emits STAR here;
            # the parser interprets it as ARITH_OP inside <expr> context.
            # See spec §3.4.1, Source 2.
            self.advance()
            self._emit(TokenType.STAR, "*", line)
            return
        if ch in ("+", "-", "/", "%"):
            self.advance()
            self._emit(TokenType.ARITH_OP, ch, line)
            return

        # ── 8. Unknown character — delegate to Member D's error handler ───
        self._unknown_char()              # Member D

    # ── Main tokenisation loop ────────────────────────────────────────────

    def _tokenize_all(self) -> list[Token]:
        """Drive the dispatcher until EOF, then append the EOF sentinel."""
        while not self.at_end():
            self._next_token()
        self._emit(TokenType.EOF, "", self._line)
        return self._tokens

    # ── Stubs for Members B, C, D ─────────────────────────────────────────
    # These raise NotImplementedError until the respective subtasks are merged.
    # Do NOT call these stubs in production; they are replaced by real code.

    def _skip_whitespace(self) -> None:          # Member D
        """Consume and skip horizontal/vertical whitespace.

                Handles: space (U+0020), tab (U+0009), CR (U+000D), LF (U+000A)
                The dispatcher will recall _next_token() after this returns, ensuring
                the next token is processed correctly.

                Per Def 2.1: "Whitespace separates tokens and is otherwise ignored."
                Line tracking is automatic via advance() when consuming \\n.
                """
        while self.peek() in (" ", "\t", "\r", "\n"):
            self.advance()

    def _skip_comment(self) -> None:             # Member D
        """Consume a line comment from # to end-of-line.

                Per Def 2.2: A line comment begins with # and extends to the end
                of the current line (terminated by \\n or EOF).

                Comments are lexically equivalent to a single whitespace token,
                so we skip silently without emitting a token.

                The newline itself is NOT consumed here; it will be consumed
                by _skip_whitespace() in the next _next_token() call.
                This ensures the line counter is updated correctly.
                """
        # We know peek() == "#" when called
        self.advance()  # consume the #

        # Consume everything until we hit \n or EOF
        while self.peek() is not None and self.peek() != "\n":
            self.advance()

        # Do NOT consume the \n; let _skip_whitespace() handle it
        # so line tracking is consistent.

    def _consume_operator(self) -> None:         # Member D
        """Consume a multi-character comparison operator.

                Dispatch order ensures we are called when peek() is one of: = ! > <

                Maximal-munch rule: consume the longest valid token.
                - ==, !=, >=, <= are two-character forms
                - = is assignment (single-char, but only valid in specific contexts;
                  the parser will validate this)

                All emitted as TokenType.OP (comparison operators).
                Assignment (=) is parsed as a comparison-like construct by the parser.
                """
        line = self._line
        ch = self.peek()

        # Lookahead: check if the next char forms a two-character operator
        if ch == "=":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, "==", line)
            else:
                # Single = — assignment operator
                self._emit(TokenType.OP, "=", line)

        elif ch == "!":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, "!=", line)
            else:
                # Bare ! is not valid in PolarPandas
                raise LexError(
                    f"Unexpected character '!'; did you mean '!='?",
                    line
                )

        elif ch == ">":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, ">=", line)
            else:
                self._emit(TokenType.OP, ">", line)

        elif ch == "<":
            self.advance()
            if self.peek() == "=":
                self.advance()
                self._emit(TokenType.OP, "<=", line)
            else:
                self._emit(TokenType.OP, "<", line)

    def _unknown_char(self) -> None:             # Member D
        """Handle an unrecognised character.

                This is the fallback for any character that doesn't match
                the first 7 categories in _next_token()'s dispatch.

                Per Property 5.2 (Progress), we transition to an error state
                with a helpful diagnostic message.
                """
        ch = self.peek()
        line = self._line

        # Provide context-specific suggestions
        suggestions = {
            "|": "chaining without | (use keywords like WHERE, FILTER)",
            "&": "use AND for logical conjunction",
            "^": "unexpected character; did you mean ^ operator?",
            "~": "unexpected character",
            "`": "backticks are not supported; use double quotes for strings",
            "'": "single quotes not supported; use double quotes \"...\" for strings",
            "@": "unexpected character (decorators not supported)",
            "{": "braces not supported (use parentheses instead: (...))",
            "}": "unexpected }; did you mean to close with )?",
            ";": "semicolons not needed; use newlines or keywords to separate statements",
        }

        msg = suggestions.get(ch, f"unexpected character {ch!r}")
        raise LexError(msg, line)

    def _consume_word(self) -> None:             # Member B
        """Consume a keyword, boolean, or identifier token.

                A word is a sequence of alphanumeric characters and underscores,
                starting with a letter or underscore.

                Classification:
                - Reserved keywords (case-insensitive): KW token type
                - Boolean literals: true | false → BOOL token type
                - Everything else: IDENT token type

                Per Spec Def 2.6: "Boolean literals are case-insensitive."
                Per Spec Def 2.2: Keywords are normalized to UPPER CASE.
                """
        line = self._line
        start_pos = self._pos

        # Consume the word: [a-zA-Z_][a-zA-Z0-9_]*
        while self.peek() is not None and (self.peek().isalnum() or self.peek() == "_"):
            self.advance()

        word = self._source[start_pos:self._pos]

        # Normalize to uppercase for keyword/boolean checking
        word_upper = word.upper()

        # Reserved keywords (Member B must maintain this list per spec §2.3)
        reserved_keywords = {
            "LOAD", "AS", "WHERE", "SELECT", "COLUMNS", "DROP",
            "NULLS", "DUPLICATES", "FILTER", "SORT", "BY", "ASC", "DESC",
            "FILL", "ENSURE", "PARSE", "LIMIT", "CLEAN", "IF", "THEN",
            "ELSE", "AND", "OR", "NOT", "FOR", "EACH", "OVER", "IN",
            "WITH", "TO", "FROM", "OF", "ON", "AT", "END", "CASE", "WHEN"
        }

        if word_upper in reserved_keywords:
            self._emit(TokenType.KW, word_upper, line)
        elif word_upper in ("TRUE", "FALSE"):
            self._emit(TokenType.BOOL, word_upper, line)
        else:
            self._emit(TokenType.IDENT, word, line)

    def _consume_string(self) -> None:           # Member C
        raise NotImplementedError("Member C: _consume_string not yet implemented")

    def _consume_number(self) -> None:           # Member C
        raise NotImplementedError("Member C: _consume_number not yet implemented")

    def _consume_loopvar(self) -> None:          # Member C
        raise NotImplementedError("Member C: _consume_loopvar not yet implemented")
        raise NotImplementedError("Member D: _skip_whitespace not yet implemented")

    def _skip_comment(self) -> None:             # Member D
        raise NotImplementedError("Member D: _skip_comment not yet implemented")

    def _consume_operator(self) -> None:         # Member D
        raise NotImplementedError("Member D: _consume_operator not yet implemented")

    def _unknown_char(self) -> None:             # Member D
        raise NotImplementedError("Member D: _unknown_char not yet implemented")

    # ── KEYWORDS set (Member B) ──────────────────────────────────────────
    # 66 reserved words (all stored UPPERCASE). Includes v1.1 additions:
    # IF, THEN, ELSE, END, FOR, EACH, OVER, DO, EXISTS, ROWCOUNT, NULLCOUNT.

    KEYWORDS: set[str] = {
        # ── I/O & inspection ──────────────────────────────────────────────
        "LOAD", "EXPORT", "PREVIEW", "INFO", "DESCRIBE",
        # ── Data source / target ──────────────────────────────────────────
        "AS", "TO", "FROM", "IN", "INTO",
        # ── Selection & filtering ─────────────────────────────────────────
        "SELECT", "COLUMNS", "FILTER", "WHERE",
        # ── Cleaning ──────────────────────────────────────────────────────
        "DROP", "NULLS", "DUPLICATES", "FILL", "WITH",
        # ── Column operations ─────────────────────────────────────────────
        "CAST", "ADD", "COLUMN", "RENAME", "SORT", "BY",
        # ── Aggregation ───────────────────────────────────────────────────
        "GROUP", "COUNT", "JOIN",
        # ── Sorting modifiers ─────────────────────────────────────────────
        "ASC", "DESC",
        # ── Plot ──────────────────────────────────────────────────────────
        "PLOT", "TYPE", "X", "Y", "TITLE", "SAVE",
        # ── Engine ────────────────────────────────────────────────────────
        "SET", "ENGINE",
        # ── Aggregation functions ─────────────────────────────────────────
        "SUM", "MEAN", "MIN", "MAX", "MEDIAN",
        # ── Data types ────────────────────────────────────────────────────
        "INT", "FLOAT", "STR", "BOOL",
        # ── Join types ────────────────────────────────────────────────────
        "INNER", "LEFT", "RIGHT", "OUTER", "ON",
        # ── Logical operators (in conditions) ─────────────────────────────
        "AND", "OR", "NOT",
        # ── Control flow (v1.1) ───────────────────────────────────────────
        "IF", "THEN", "ELSE", "END",
        "FOR", "EACH", "OVER", "DO",
        # ── Meta-condition keywords (v1.1) ────────────────────────────────
        "EXISTS", "ROWCOUNT", "NULLCOUNT",
        # ── Plot types ────────────────────────────────────────────────────
        "BAR", "LINE", "SCATTER", "HIST",
        # ── Miscellaneous ─────────────────────────────────────────────────
        "USING", "ROWS",
    }

    def _consume_word(self) -> None:             # Member B
        """Read the longest [a-zA-Z_][a-zA-Z0-9_]* sequence, then classify:
        - ``true`` / ``false`` (case-insensitive) → BOOL token
        - word found in KEYWORDS (case-insensitive) → KW token (stored UPPER)
        - anything else → IDENT token (original casing preserved)
        """
        start_line = self._line
        start_pos = self._pos

        # Consume the full word using maximal munch
        while not self.at_end():
            ch = self.peek()
            if ch is not None and (ch.isalnum() or ch == "_"):
                self.advance()
            else:
                break

        word = self._source[start_pos:self._pos]
        upper = word.upper()

        # Booleans: true / false (case-insensitive) → BOOL
        if upper in ("TRUE", "FALSE"):
            self._emit(TokenType.BOOL, upper, start_line)
        # Keywords: case-insensitive match against KEYWORDS set → KW
        elif upper in self.KEYWORDS:
            self._emit(TokenType.KW, upper, start_line)
        # Everything else → IDENT (original casing)
        else:
            self._emit(TokenType.IDENT, word, start_line)

    def _consume_string(self) -> None:           # Member C
        start_line = self._line
        if self.peek() != '"':
            raise LexError("String literal must start with double quote", start_line)

        self.advance()  # opening quote
        chars: list[str] = []
        escapes = {
            '"': '"',
            "\\": "\\",
            "n": "\n",
            "t": "\t",
        }

        while True:
            ch = self.peek()
            if ch is None or ch == "\n":
                raise LexError("Unterminated string literal", start_line)
            if ch == '"':
                self.advance()
                self._emit(TokenType.STRING, f'"{"".join(chars)}"', start_line)
                return
            if ch == "\\":
                self.advance()
                esc = self.peek()
                if esc is None or esc == "\n":
                    raise LexError("Unterminated string literal", start_line)
                if esc not in escapes:
                    raise LexError(f"Invalid escape sequence '\\{esc}'", self._line)
                self.advance()
                chars.append(escapes[esc])
                continue

            chars.append(self.advance())

    def _consume_number(self) -> None:           # Member C
        start_line = self._line
        start_pos = self._pos

        while (self.peek() or "").isdigit():
            self.advance()

        if self.peek() == "." and (self.peek(1) or "").isdigit():
            self.advance()
            while (self.peek() or "").isdigit():
                self.advance()
            self._emit(TokenType.FLOAT, self._source[start_pos:self._pos], start_line)
            return

        self._emit(TokenType.INTEGER, self._source[start_pos:self._pos], start_line)

    def _consume_loopvar(self) -> None:          # Member C
        start_line = self._line
        if self.peek() != "$":
            raise LexError("Loop variable must start with '$'", start_line)

        self.advance()  # $
        first = self.peek()
        if first is None or not (first.isalpha() or first == "_"):
            raise LexError("'$' must be followed by a letter or underscore", start_line)

        start_pos = self._pos
        while True:
            ch = self.peek()
            if ch is not None and (ch.isalnum() or ch == "_"):
                self.advance()
            else:
                break

        self._emit(TokenType.LOOPVAR, f"${self._source[start_pos:self._pos]}", start_line)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PUBLIC ENTRY POINT
#    This is the function the Parser calls.  It is the only public symbol
#    that consumers of this module need to import.
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(source: str) -> list[Token]:
    """Lex a PolarPandas source string into an ordered list of tokens.

    The final token is always ``Token(TokenType.EOF, "", <last_line>)``.
    Raises ``LexError`` on the first unrecognised character or malformed
    token (Member D will extend this to collect-all mode).

    Parameters
    ----------
    source : str
        Full PolarPandas source text (UTF-8 string).

    Returns
    -------
    list[Token]
        Flat token sequence including the terminal EOF token.

    Raises
    ------
    LexError
        On any lexical error.

    Example
    -------
    # >>> from polarpandas.lexer import tokenize
    # >>> tokens = tokenize('LOAD "data.csv" AS df')
    # >>> [(t.type.name, t.value) for t in tokens]
    [('KW', 'LOAD'), ('STRING', '"data.csv"'), ('KW', 'AS'),
     ('IDENT', 'df'), ('EOF', '')]
    """
    return _Lexer(source)._tokenize_all()