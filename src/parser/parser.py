"""
polarpandas.parser
==================
PolarPandas Translator — Component 2: Parser

Public interface:
    parse(tokens: list[Token]) -> list[ASTNode]

All statement syntax follows the PolarPandas spec v1.1 exactly.

Changelog:
    v1.0  Members A/B/C/D — initial implementation
    v1.1  Fix — all statement syntaxes corrected to match spec:
              SELECT  : SELECT <ident> COLUMNS <col-list|*>
              CAST    : CAST <ident> COLUMN "<col>" TO <dtype>
              RENAME  : RENAME <ident> COLUMN "<old>" TO "<new>"
              GROUP   : GROUP <ident> BY <col-or-list> AGGREGATE "<col>" AS <agg>
              COUNT   : COUNT <ident> GROUP BY <col-or-list>
              JOIN    : JOIN <i> WITH <i> ON "<key>" [TYPE <jtype>] [AS <ident>]
              FILL    : FILL NULLS IN <ident> COLUMN "<col>" WITH <value|method>
              DROP    : DROP NULLS FROM <ident> [IN COLUMNS <col-list>]
          Note — DOT token removed; df.column notation not in spec (spec §2.2.6)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from lexer.lexer import Token, TokenType


# -----------------------------------------------------------------------------
# 1. PARSE ERROR
# -----------------------------------------------------------------------------

class ParseError(Exception):
    """Raised on unexpected token or malformed structure.

    Attributes
    ----------
    message : str
    line    : int   1-based source line.
    """
    def __init__(self, message: str, line: int) -> None:
        super().__init__(f"[Parser] line {line}: {message}")
        self.message = message
        self.line    = line


# -----------------------------------------------------------------------------
# 2. EXPRESSION NODES
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ExprLiteral:
    """A scalar literal value.

    Attributes
    ----------
    value : str   Raw token value, e.g. '42', '3.14', '"hello"', 'true'
    kind  : str   'integer' | 'float' | 'string' | 'bool'
    line  : int
    """
    value : str
    kind  : str
    line  : int


@dataclass(frozen=True)
class ExprColRef:
    """A column reference inside an expression.

    Forms:
        "amount"  (quoted string atom) -> ExprColRef(name='amount', is_loopvar=False)
        amount    (bare ident atom)    -> ExprColRef(name='amount', is_loopvar=False)
        $col      (loop variable)      -> ExprColRef(name='col',    is_loopvar=True)

    In the spec, column names inside expressions are quoted strings (⟨string⟩).
    Bare identifiers and loop variables are accepted as a practical extension
    for use inside FOR loop bodies.
    Dot notation (df.column) is NOT supported — see spec §2.2.6.

    Attributes
    ----------
    name       : str
    is_loopvar : bool   True if sourced from a LOOPVAR token ($col).
    line       : int
    """
    name       : str
    is_loopvar : bool
    line       : int


@dataclass(frozen=True)
class ExprBinop:
    """Binary arithmetic expression: left OP right.

    Attributes
    ----------
    op    : str   '+' | '-' | '*' | '/' | '%'
    left  : Expr
    right : Expr
    line  : int
    """
    op    : str
    left  : "Expr"
    right : "Expr"
    line  : int


Expr = Union[ExprLiteral, ExprColRef, ExprBinop]


# -----------------------------------------------------------------------------
# 3. CONDITION NODES
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CondCompare:
    """<expr> <op> <expr>   e.g.  "amount" >= 100"""
    left  : Expr
    op    : str   # '==' | '!=' | '>' | '<' | '>=' | '<='
    right : Expr
    line  : int


@dataclass(frozen=True)
class CondIn:
    """"<col>" [NOT] IN [v1, v2, ...]"""
    col    : ExprColRef
    values : list          # list[ExprLiteral]
    negate : bool          # True -> NOT IN
    line   : int


@dataclass(frozen=True)
class CondNull:
    """"<col>" IS [NOT] NULL"""
    col    : ExprColRef
    is_not : bool          # True -> IS NOT NULL
    line   : int


@dataclass(frozen=True)
class CondBetween:
    """"<col>" BETWEEN <low> AND <high>"""
    col  : ExprColRef
    low  : ExprLiteral
    high : ExprLiteral
    line : int


@dataclass(frozen=True)
class CondAnd:
    """left AND right  (higher precedence than OR)"""
    left  : "Condition"
    right : "Condition"
    line  : int


@dataclass(frozen=True)
class CondOr:
    """left OR right  (lower precedence than AND)"""
    left  : "Condition"
    right : "Condition"
    line  : int


Condition = Union[CondCompare, CondIn, CondNull, CondBetween, CondAnd, CondOr]


# -----------------------------------------------------------------------------
# 4. STATEMENT AST NODES
#    One frozen dataclass per statement type (22 total).
#    All string fields that hold column names store the raw value with
#    quotes stripped (the parser strips them; downstream code gets plain text).
# -----------------------------------------------------------------------------

# -- I/O (spec §2.2.2) --------------------------------------------------------

@dataclass(frozen=True)
class AstLoad:
    """LOAD "<file>" AS <ident> [ENGINE pandas|polars]

    Spec: LOAD "<file>" AS <ident> [ENGINE pandas|polars]
    """
    file   : str            # file path (quotes stripped)
    name   : str            # identifier to bind the dataframe to
    engine : Optional[str]  # 'pandas' | 'polars' | None (use current default)
    line   : int


@dataclass(frozen=True)
class AstExport:
    """EXPORT <ident> TO "<file>" [FORMAT csv|parquet|json|xlsx]

    Spec: EXPORT <ident> TO "<file>" [FORMAT csv|parquet|json|xlsx]
    """
    name   : str
    file   : str
    format : Optional[str]  # 'csv'|'parquet'|'json'|'xlsx'|None (infer from ext)
    line   : int


# -- Inspection (spec §2.2.3) -------------------------------------------------

@dataclass(frozen=True)
class AstPreview:
    """PREVIEW <ident> [ROWS <int>]"""
    name : str
    rows : int    # default 5
    line : int


@dataclass(frozen=True)
class AstInfo:
    """INFO <ident>"""
    name : str
    line : int


@dataclass(frozen=True)
class AstDescribe:
    """DESCRIBE <ident>"""
    name : str
    line : int


@dataclass(frozen=True)
class AstSetEngine:
    """SET ENGINE pandas|polars"""
    engine : str  # 'pandas' | 'polars'
    line   : int


# -- Selection and Filtering (spec §2.2.4) ------------------------------------

@dataclass(frozen=True)
class AstSelect:
    """SELECT <ident> COLUMNS <col-list> | *

    Spec: SELECT <ident> COLUMNS <col-list | *>
    columns = [] means wildcard (*), i.e. keep all columns.
    """
    name    : str
    columns : list   # list[str]; empty = wildcard (SELECT COLUMNS *)
    line    : int


@dataclass(frozen=True)
class AstFilter:
    """FILTER <ident> WHERE <condition>"""
    name      : str
    condition : Condition
    line      : int


# -- Cleaning (spec §2.2.5) ---------------------------------------------------

@dataclass(frozen=True)
class AstDropNulls:
    """DROP NULLS FROM <ident> [IN COLUMNS <col-list>]

    Spec: DROP NULLS FROM <ident> [IN COLUMNS <col-list>]
    columns = [] means all columns.
    """
    name    : str
    columns : list   # list[str]; empty = all columns
    line    : int


@dataclass(frozen=True)
class AstDropDups:
    """DROP DUPLICATES FROM <ident> [KEEP first|last|none]

    Spec: DROP DUPLICATES FROM <ident> [KEEP first|last|none]
    """
    name : str
    keep : str   # 'first' | 'last' | 'none'  (default 'first')
    line : int


@dataclass(frozen=True)
class AstDropCol:
    """DROP COLUMN "<col>" FROM <ident>"""
    name   : str
    column : str   # column name (quotes stripped)
    line   : int


@dataclass(frozen=True)
class AstFillNulls:
    """FILL NULLS IN <ident> COLUMN "<col>" WITH <value|method>

    Spec: FILL NULLS IN <ident> COLUMN "<col>" WITH <value|method>
    Single column per statement (use multiple FILL statements for multiple cols).
    fill is either an ExprLiteral (literal value) or an ExprColRef
    (method name like 'mean', 'ffill') stored as an IDENT.
    """
    name   : str
    column : str   # column name (quotes stripped)
    fill   : Expr  # literal value or method name (as ExprColRef with is_loopvar=False)
    line   : int


@dataclass(frozen=True)
class AstCast:
    """CAST <ident> COLUMN "<col>" TO <dtype>

    Spec: CAST <ident> COLUMN "<col>" TO <dtype>
    dtype: int | float | str | bool | datetime | date | category
    """
    name   : str
    column : str   # column name (quotes stripped)
    dtype  : str   # target type string (lowercase, e.g. 'int', 'float')
    line   : int


# -- Transformation (spec §2.2.6) ---------------------------------------------

@dataclass(frozen=True)
class AstAddCol:
    """ADD COLUMN "<col>" TO <ident> AS <expr>"""
    name   : str
    column : str   # new column name (quotes stripped)
    expr   : Expr
    line   : int


@dataclass(frozen=True)
class AstRename:
    """RENAME <ident> COLUMN "<old>" TO "<new>"

    Spec: RENAME <ident> COLUMN "<old>" TO "<new>"
    """
    name : str
    old  : str   # old column name (quotes stripped)
    new  : str   # new column name (quotes stripped)
    line : int


@dataclass(frozen=True)
class AstSort:
    """SORT <ident> BY "<col>" [ASC|DESC]"""
    name      : str
    column    : str   # column name (quotes stripped)
    direction : str   # 'ASC' | 'DESC'  (default 'ASC')
    line      : int


# -- Aggregation and Join (spec §2.2.7) ---------------------------------------

@dataclass(frozen=True)
class AstGroup:
    """GROUP <ident> BY <col-or-list> AGGREGATE "<col>" AS <agg>

    Spec: GROUP <ident> BY <col-or-list> AGGREGATE "<col>" AS <agg>
    agg: sum|mean|median|min|max|count|std|var|first|last
    (validated by semantic layer, not the parser)
    """
    name    : str
    by      : list   # list[str] — one or more group-by columns
    column  : str    # column to aggregate (quotes stripped)
    agg     : str    # aggregation function name
    line    : int


@dataclass(frozen=True)
class AstCount:
    """COUNT <ident> GROUP BY <col-or-list>

    Spec: COUNT <ident> GROUP BY <col-or-list>
    Counts rows per group (analogous to GROUP BY + COUNT(*) in SQL).
    """
    name    : str
    by      : list   # list[str]
    line    : int


@dataclass(frozen=True)
class AstJoin:
    """JOIN <ident> WITH <ident> ON "<key>" [TYPE <jtype>] [AS <ident>]

    Spec: JOIN <i> WITH <i> ON "<key>" [TYPE <jtype>] [AS <ident>]
    how: inner|left|right|outer|cross  (default 'inner')
    result: name for the output dataframe (default = left operand name)
    """
    left   : str
    right  : str
    on     : str            # join key column (quotes stripped)
    how    : str            # join type (default 'inner')
    result : Optional[str]  # result dataframe name
    line   : int


# -- Visualisation (spec §2.2.8) ----------------------------------------------

@dataclass(frozen=True)
class AstPlot:
    """PLOT <ident> TYPE <ptype> <plot-opt>*

    Spec: PLOT <ident> TYPE <ptype> <plot-opt>*
    All plot options are optional. See spec §2.2.8 for full option list.
    """
    name  : str
    kind  : str             # plot type (validated by semantic layer)
    x     : Optional[str]   # X-axis column (quotes stripped)
    y     : Optional[str]   # Y-axis column
    hue   : Optional[str]
    color : Optional[str]
    title : Optional[str]
    xlabel: Optional[str]
    ylabel: Optional[str]
    bins  : Optional[int]
    save  : Optional[str]   # output file path
    column: Optional[str]   # for plot types that operate on a single column
    line  : int


# -- Control Flow (spec §5) ---------------------------------------------------

@dataclass(frozen=True)
class AstIf:
    """IF <meta-cond> THEN <body> [ELSE <body>] END"""
    condition : Condition
    then_body : list   # list[ASTNode]
    else_body : list   # list[ASTNode]; empty if no ELSE branch
    line      : int


@dataclass(frozen=True)
class AstFor:
    """FOR EACH $<var> OVER <col-list> DO <body> END"""
    var     : str    # loop variable name (without '$')
    columns : list   # list[str] — static column list
    body    : list   # list[ASTNode]
    line    : int


ASTNode = Union[
    AstLoad, AstExport, AstPreview, AstInfo, AstDescribe, AstSetEngine,
    AstSelect, AstFilter,
    AstDropNulls, AstDropDups, AstDropCol, AstFillNulls, AstCast,
    AstAddCol, AstRename, AstSort,
    AstGroup, AstCount, AstJoin, AstPlot,
    AstIf, AstFor,
]


# -----------------------------------------------------------------------------
# 5. PARSER CLASS
# -----------------------------------------------------------------------------

class _Parser:
    """Internal mutable parser state.
    parse(tokens) constructs one _Parser per call and returns the AST list.
    """

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens: list[Token] = tokens
        self._pos:    int         = 0

    # -------------------------------------------------------------------------
    # Cursor primitives
    # -------------------------------------------------------------------------

    def peek(self, offset: int = 0) -> Token:
        """Return token at _pos+offset; always safe (returns EOF at end)."""
        idx = self._pos + offset
        if idx < len(self._tokens):
            return self._tokens[idx]
        return self._tokens[-1]  # EOF sentinel

    def consume(self) -> Token:
        """Consume and return current token. Raises ParseError at EOF."""
        tok = self.peek()
        if tok.type is TokenType.EOF:
            raise ParseError("Unexpected end of input", tok.line)
        self._pos += 1
        return tok

    def at_end(self) -> bool:
        return self.peek().type is TokenType.EOF

    # -------------------------------------------------------------------------
    # Typed consume helpers
    # -------------------------------------------------------------------------

    def expect_kw(self, *keywords: str) -> Token:
        """Consume next token if it is a KW matching one of keywords.
        Comparison is case-insensitive (keywords stored UPPER by lexer).
        """
        tok       = self.peek()
        upper_set = {kw.upper() for kw in keywords}
        if tok.type is TokenType.KW and tok.value in upper_set:
            return self.consume()
        expected = " or ".join(f"'{k}'" for k in keywords)
        raise ParseError(
            f"Expected keyword {expected}, got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    def expect_ident(self) -> Token:
        tok = self.peek()
        if tok.type is TokenType.IDENT:
            return self.consume()
        raise ParseError(
            f"Expected identifier, got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    def expect_string(self) -> Token:
        tok = self.peek()
        if tok.type is TokenType.STRING:
            return self.consume()
        raise ParseError(
            f"Expected string literal, got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    def expect_int(self) -> Token:
        tok = self.peek()
        if tok.type is TokenType.INTEGER:
            return self.consume()
        raise ParseError(
            f"Expected integer, got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    # -------------------------------------------------------------------------
    # Lookahead predicates
    # -------------------------------------------------------------------------

    def next_is_kw(self, *keywords: str) -> bool:
        tok = self.peek()
        return tok.type is TokenType.KW and tok.value in {kw.upper() for kw in keywords}

    def next_is(self, ttype: TokenType) -> bool:
        return self.peek().type is ttype

    # -------------------------------------------------------------------------
    # String value helper
    # -------------------------------------------------------------------------

    @staticmethod
    def _strip_quotes(s: str) -> str:
        """Strip enclosing double quotes from a string token value."""
        if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
            return s[1:-1]
        return s

    # -------------------------------------------------------------------------
    # Shared sub-parsers
    # -------------------------------------------------------------------------

    def parse_col_list(self) -> list:
        """Parse ["col1", "col2", ...] or [col1, col2, ...] or [$var, ...].

        Returns list[str] of column name strings (quotes stripped).
        The spec uses quoted strings for column names in lists; this parser
        also accepts unquoted IDENT and LOOPVAR for flexibility.
        """
        self.expect_kw_or_punct(TokenType.LBRACKET, "[")
        cols: list[str] = []

        while not self.next_is(TokenType.RBRACKET):
            tok = self.peek()
            if tok.type is TokenType.STRING:
                cols.append(self._strip_quotes(self.consume().value))
            elif tok.type is TokenType.IDENT:
                cols.append(self.consume().value)
            elif tok.type is TokenType.LOOPVAR:
                # $col inside a column list — store without '$'
                raw = self.consume().value
                cols.append(raw[1:] if raw.startswith("$") else raw)
            elif tok.type is TokenType.KW:
                # Accept bare keywords as column names (e.g. a column named 'type')
                cols.append(self.consume().value.lower())
            else:
                raise ParseError(
                    f"Expected column name in list, got {tok.type.name} {tok.value!r}",
                    tok.line,
                )
            if self.next_is(TokenType.COMMA):
                self.consume()
            elif not self.next_is(TokenType.RBRACKET):
                raise ParseError(
                    f"Expected ',' or ']' in column list, "
                    f"got {self.peek().type.name} {self.peek().value!r}",
                    self.peek().line,
                )

        self.expect_kw_or_punct(TokenType.RBRACKET, "]")
        return cols

    def parse_value_list(self) -> list:
        """Parse [value, value, ...].  Returns list[ExprLiteral]."""
        self.expect_kw_or_punct(TokenType.LBRACKET, "[")
        values: list[ExprLiteral] = []

        while not self.next_is(TokenType.RBRACKET):
            values.append(self.parse_value())
            if self.next_is(TokenType.COMMA):
                self.consume()
            elif not self.next_is(TokenType.RBRACKET):
                raise ParseError(
                    f"Expected ',' or ']' in value list, "
                    f"got {self.peek().type.name} {self.peek().value!r}",
                    self.peek().line,
                )

        self.expect_kw_or_punct(TokenType.RBRACKET, "]")
        return values

    def parse_value(self) -> ExprLiteral:
        """Parse a single literal value."""
        tok = self.peek()
        if tok.type is TokenType.STRING:
            return ExprLiteral(self.consume().value, "string", tok.line)
        if tok.type is TokenType.INTEGER:
            return ExprLiteral(self.consume().value, "integer", tok.line)
        if tok.type is TokenType.FLOAT:
            return ExprLiteral(self.consume().value, "float", tok.line)
        if tok.type is TokenType.BOOL:
            return ExprLiteral(self.consume().value, "bool", tok.line)
        raise ParseError(
            f"Expected literal value, got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    def expect_kw_or_punct(self, ttype: TokenType, display: str) -> Token:
        """Consume the next token if it matches ttype.  Used for [ and ]."""
        tok = self.peek()
        if tok.type is ttype:
            return self.consume()
        raise ParseError(
            f"Expected '{display}', got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    def parse_col_or_list(self) -> list:
        """Parse either a single column name (string or ident) or a col-list.

        Returns list[str] in both cases (single col -> one-element list).
        Spec: <col-or-list> ::= <string> | <col-list>
        """
        if self.next_is(TokenType.LBRACKET):
            return self.parse_col_list()
        # Single column: accept quoted string or bare identifier
        tok = self.peek()
        if tok.type is TokenType.STRING:
            return [self._strip_quotes(self.consume().value)]
        if tok.type is TokenType.IDENT:
            return [self.consume().value]
        raise ParseError(
            f"Expected column name or column list, "
            f"got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    # -------------------------------------------------------------------------
    # Statement dispatcher
    # -------------------------------------------------------------------------

    def parse_statement(self) -> ASTNode:
        """Dispatch to the correct parse_*() based on leading keyword."""
        tok = self.peek()
        if tok.type is not TokenType.KW:
            raise ParseError(
                f"Expected statement keyword, got {tok.type.name} {tok.value!r}",
                tok.line,
            )
        kw = tok.value  # already UPPER

        dispatch = {
            "LOAD":     self.parse_load,
            "EXPORT":   self.parse_export,
            "PREVIEW":  self.parse_preview,
            "INFO":     self.parse_info,
            "DESCRIBE": self.parse_describe,
            "SET":      self.parse_set_engine,
            "SELECT":   self.parse_select,
            "FILTER":   self.parse_filter,
            "DROP":     self.parse_drop,
            "FILL":     self.parse_fill_nulls,
            "CAST":     self.parse_cast,
            "ADD":      self.parse_add_col,
            "RENAME":   self.parse_rename,
            "SORT":     self.parse_sort,
            "GROUP":    self.parse_group,
            "COUNT":    self.parse_count,
            "JOIN":     self.parse_join,
            "PLOT":     self.parse_plot,
            "IF":       self.parse_if,
            "FOR":      self.parse_for,
        }
        if kw in dispatch:
            return dispatch[kw]()
        raise ParseError(f"Unknown statement keyword {kw!r}", tok.line)

    def _parse_all(self) -> list[ASTNode]:
        nodes: list[ASTNode] = []
        while not self.at_end():
            nodes.append(self.parse_statement())
        return nodes

    # =========================================================================
    # STATEMENT PARSERS
    # =========================================================================

    # -------------------------------------------------------------------------
    # I/O  (Member B)
    # -------------------------------------------------------------------------

    def parse_load(self) -> AstLoad:
        """LOAD "<file>" AS <ident> [ENGINE pandas|polars]"""
        start  = self.expect_kw("LOAD")
        file   = self._strip_quotes(self.expect_string().value)
        self.expect_kw("AS")
        name   = self.expect_ident().value
        engine = None
        if self.next_is_kw("ENGINE"):
            self.consume()
            tok = self.peek()
            if tok.type in (TokenType.IDENT, TokenType.KW):
                engine = self.consume().value.lower()
            else:
                raise ParseError(
                    f"Expected engine name after ENGINE, "
                    f"got {tok.type.name} {tok.value!r}",
                    tok.line,
                )
        return AstLoad(file=file, name=name, engine=engine, line=start.line)

    def parse_export(self) -> AstExport:
        """EXPORT <ident> TO "<file>" [FORMAT csv|parquet|json|xlsx]"""
        start  = self.expect_kw("EXPORT")
        name   = self.expect_ident().value
        self.expect_kw("TO")
        file   = self._strip_quotes(self.expect_string().value)
        fmt    = None
        if self.next_is_kw("FORMAT"):
            self.consume()
            tok = self.peek()
            if tok.type in (TokenType.IDENT, TokenType.KW):
                fmt = self.consume().value.lower()
            else:
                raise ParseError(
                    f"Expected format name after FORMAT, "
                    f"got {tok.type.name} {tok.value!r}",
                    tok.line,
                )
        return AstExport(name=name, file=file, format=fmt, line=start.line)

    def parse_preview(self) -> AstPreview:
        """PREVIEW <ident> [ROWS <int>]"""
        start = self.expect_kw("PREVIEW")
        name  = self.expect_ident().value
        rows  = 5
        if self.next_is_kw("ROWS"):
            self.consume()
            rows = int(self.expect_int().value)
        return AstPreview(name=name, rows=rows, line=start.line)

    def parse_info(self) -> AstInfo:
        """INFO <ident>"""
        start = self.expect_kw("INFO")
        return AstInfo(name=self.expect_ident().value, line=start.line)

    def parse_describe(self) -> AstDescribe:
        """DESCRIBE <ident>"""
        start = self.expect_kw("DESCRIBE")
        return AstDescribe(name=self.expect_ident().value, line=start.line)

    def parse_set_engine(self) -> AstSetEngine:
        """SET ENGINE pandas|polars"""
        start = self.expect_kw("SET")
        self.expect_kw("ENGINE")
        tok = self.peek()
        if tok.type in (TokenType.IDENT, TokenType.KW):
            engine = self.consume().value.lower()
            return AstSetEngine(engine=engine, line=start.line)
        raise ParseError(
            f"Expected engine name (pandas|polars), "
            f"got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    # -------------------------------------------------------------------------
    # Selection (Member C)
    # -------------------------------------------------------------------------

    def parse_select(self) -> AstSelect:
        """SELECT <ident> COLUMNS <col-list> | *

        Spec: SELECT <ident> COLUMNS <col-list | *>
        The dataframe identifier comes immediately after SELECT.
        """
        start = self.expect_kw("SELECT")
        name  = self.expect_ident().value
        self.expect_kw("COLUMNS")

        if self.next_is(TokenType.STAR):
            self.consume()
            columns: list[str] = []   # empty = wildcard
        else:
            columns = self.parse_col_list()

        return AstSelect(name=name, columns=columns, line=start.line)

    def parse_filter(self) -> AstFilter:
        """FILTER <ident> WHERE <condition>"""
        start = self.expect_kw("FILTER")
        name  = self.expect_ident().value
        self.expect_kw("WHERE")
        cond  = self.parse_condition()
        return AstFilter(name=name, condition=cond, line=start.line)

    # -------------------------------------------------------------------------
    # Cleaning (Member C)
    # -------------------------------------------------------------------------

    def parse_drop(self) -> ASTNode:
        """Dispatch DROP variants:
            DROP NULLS FROM <ident> [IN COLUMNS <col-list>]
            DROP DUPLICATES FROM <ident> [KEEP first|last|none]
            DROP COLUMN "<col>" FROM <ident>
        """
        start = self.expect_kw("DROP")
        tok   = self.peek()
        if tok.type is not TokenType.KW:
            raise ParseError(
                f"Expected NULLS, DUPLICATES, or COLUMN after DROP, "
                f"got {tok.type.name} {tok.value!r}",
                tok.line,
            )
        variant = tok.value

        if variant == "NULLS":
            self.consume()
            self.expect_kw("FROM")
            name    = self.expect_ident().value
            columns : list[str] = []
            if self.next_is_kw("IN"):
                self.consume()
                self.expect_kw("COLUMNS")
                columns = self.parse_col_list()
            return AstDropNulls(name=name, columns=columns, line=start.line)

        if variant == "DUPLICATES":
            self.consume()
            self.expect_kw("FROM")
            name = self.expect_ident().value
            keep = "first"
            if self.next_is_kw("KEEP"):
                self.consume()
                tok2 = self.peek()
                if tok2.type in (TokenType.IDENT, TokenType.KW):
                    keep = self.consume().value.lower()
                else:
                    raise ParseError(
                        f"Expected first|last|none after KEEP, "
                        f"got {tok2.type.name} {tok2.value!r}",
                        tok2.line,
                    )
            return AstDropDups(name=name, keep=keep, line=start.line)

        if variant == "COLUMN":
            self.consume()
            col  = self._strip_quotes(self.expect_string().value)
            self.expect_kw("FROM")
            name = self.expect_ident().value
            return AstDropCol(name=name, column=col, line=start.line)

        raise ParseError(
            f"Expected NULLS, DUPLICATES, or COLUMN after DROP, got {variant!r}",
            tok.line,
        )

    def parse_fill_nulls(self) -> AstFillNulls:
        """FILL NULLS IN <ident> COLUMN "<col>" WITH <value|method>

        Spec: FILL NULLS IN <ident> COLUMN "<col>" WITH <value|method>
        Single column per FILL statement.
        """
        start  = self.expect_kw("FILL")
        self.expect_kw("NULLS")
        self.expect_kw("IN")
        name   = self.expect_ident().value
        self.expect_kw("COLUMN")
        # Accept quoted string OR loop variable ($col) in the column position
        col_tok = self.peek()
        if col_tok.type is TokenType.STRING:
            column = self._strip_quotes(self.consume().value)
        elif col_tok.type is TokenType.LOOPVAR:
            raw = self.consume().value
            column = raw[1:] if raw.startswith("$") else raw
        elif col_tok.type is TokenType.IDENT:
            column = self.consume().value
        else:
            raise ParseError(
                f"Expected column name after COLUMN, "
                f"got {col_tok.type.name} {col_tok.value!r}",
                col_tok.line,
            )
        self.expect_kw("WITH")
        fill   = self._parse_fill_value()
        return AstFillNulls(name=name, column=column, fill=fill, line=start.line)

    def _parse_fill_value(self) -> Expr:
        """Parse the fill value or method name after WITH.

        Accepts:
          - Literal values: integers, floats, strings, booleans
          - Method names: mean, median, mode, ffill, bfill  (as IDENT tokens)
        """
        tok = self.peek()
        if tok.type in (TokenType.INTEGER, TokenType.FLOAT,
                        TokenType.STRING, TokenType.BOOL):
            return self.parse_value()
        if tok.type is TokenType.IDENT:
            # Method name — wrap as ExprColRef with is_loopvar=False
            self.consume()
            return ExprColRef(name=tok.value, is_loopvar=False, line=tok.line)
        raise ParseError(
            f"Expected fill value or method name after WITH, "
            f"got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    def parse_cast(self) -> AstCast:
        """CAST <ident> COLUMN "<col>" TO <dtype>

        Spec: CAST <ident> COLUMN "<col>" TO <dtype>
        dtype: int|float|str|bool|datetime|date|category
        """
        start  = self.expect_kw("CAST")
        name   = self.expect_ident().value
        self.expect_kw("COLUMN")
        col_tok2 = self.peek()
        if col_tok2.type is TokenType.STRING:
            column = self._strip_quotes(self.consume().value)
        elif col_tok2.type is TokenType.LOOPVAR:
            raw2 = self.consume().value
            column = raw2[1:] if raw2.startswith("$") else raw2
        elif col_tok2.type is TokenType.IDENT:
            column = self.consume().value
        else:
            raise ParseError(
                f"Expected column name after COLUMN, "
                f"got {col_tok2.type.name} {col_tok2.value!r}",
                col_tok2.line,
            )
        self.expect_kw("TO")
        tok    = self.peek()
        if tok.type in (TokenType.IDENT, TokenType.KW):
            dtype = self.consume().value.lower()
        else:
            raise ParseError(
                f"Expected dtype after TO, got {tok.type.name} {tok.value!r}",
                tok.line,
            )
        return AstCast(name=name, column=column, dtype=dtype, line=start.line)

    # -------------------------------------------------------------------------
    # Transformation (Member C)
    # -------------------------------------------------------------------------

    def parse_add_col(self) -> AstAddCol:
        """ADD COLUMN "<col>" TO <ident> AS <expr>"""
        start  = self.expect_kw("ADD")
        self.expect_kw("COLUMN")
        column = self._strip_quotes(self.expect_string().value)
        self.expect_kw("TO")
        name   = self.expect_ident().value
        self.expect_kw("AS")
        expr   = self.parse_expr()
        return AstAddCol(name=name, column=column, expr=expr, line=start.line)

    def parse_rename(self) -> AstRename:
        """RENAME <ident> COLUMN "<old>" TO "<new>"

        Spec: RENAME <ident> COLUMN "<old>" TO "<new>"
        """
        start = self.expect_kw("RENAME")
        name  = self.expect_ident().value
        self.expect_kw("COLUMN")
        old   = self._strip_quotes(self.expect_string().value)
        self.expect_kw("TO")
        new   = self._strip_quotes(self.expect_string().value)
        return AstRename(name=name, old=old, new=new, line=start.line)

    def parse_sort(self) -> AstSort:
        """SORT <ident> BY "<col>" [ASC|DESC]"""
        start     = self.expect_kw("SORT")
        name      = self.expect_ident().value
        self.expect_kw("BY")
        column    = self._strip_quotes(self.expect_string().value)
        direction = "ASC"
        if self.next_is_kw("ASC"):
            self.consume(); direction = "ASC"
        elif self.next_is_kw("DESC"):
            self.consume(); direction = "DESC"
        return AstSort(name=name, column=column, direction=direction, line=start.line)

    # -------------------------------------------------------------------------
    # Aggregation and Join  (Member D)
    # -------------------------------------------------------------------------

    def parse_group(self) -> AstGroup:
        """GROUP <ident> BY <col-or-list> AGGREGATE "<col>" AS <agg>

        Spec: GROUP <ident> BY <col-or-list> AGGREGATE "<col>" AS <agg>
        agg is any identifier (semantic validator checks allowed values).
        """
        start  = self.expect_kw("GROUP")
        name   = self.expect_ident().value
        self.expect_kw("BY")
        by     = self.parse_col_or_list()
        self.expect_kw("AGGREGATE")
        column = self._strip_quotes(self.expect_string().value)
        self.expect_kw("AS")
        tok    = self.peek()
        if tok.type in (TokenType.IDENT, TokenType.KW):
            agg = self.consume().value.lower()
        else:
            raise ParseError(
                f"Expected aggregation function name after AS, "
                f"got {tok.type.name} {tok.value!r}",
                tok.line,
            )
        return AstGroup(name=name, by=by, column=column, agg=agg, line=start.line)

    def parse_count(self) -> AstCount:
        """COUNT <ident> GROUP BY <col-or-list>

        Spec: COUNT <ident> GROUP BY <col-or-list>
        Counts rows per group (equivalent to SQL GROUP BY + COUNT(*)).
        """
        start = self.expect_kw("COUNT")
        name  = self.expect_ident().value
        self.expect_kw("GROUP")
        self.expect_kw("BY")
        by    = self.parse_col_or_list()
        return AstCount(name=name, by=by, line=start.line)

    def parse_join(self) -> AstJoin:
        """JOIN <ident> WITH <ident> ON "<key>" [TYPE <jtype>] [AS <ident>]

        Spec: JOIN <i> WITH <i> ON "<key>" [TYPE <jtype>] [AS <ident>]
        join key is a quoted string; join type via TYPE keyword.
        """
        start  = self.expect_kw("JOIN")
        left   = self.expect_ident().value
        self.expect_kw("WITH")
        right  = self.expect_ident().value
        self.expect_kw("ON")
        on     = self._strip_quotes(self.expect_string().value)
        how    = "inner"
        result = None
        if self.next_is_kw("TYPE"):
            self.consume()
            tok = self.peek()
            if tok.type in (TokenType.IDENT, TokenType.KW):
                how = self.consume().value.lower()
            else:
                raise ParseError(
                    f"Expected join type after TYPE, "
                    f"got {tok.type.name} {tok.value!r}",
                    tok.line,
                )
        if self.next_is_kw("AS"):
            self.consume()
            result = self.expect_ident().value
        return AstJoin(left=left, right=right, on=on, how=how,
                       result=result, line=start.line)

    # -------------------------------------------------------------------------
    # Visualisation  (Member D)
    # -------------------------------------------------------------------------

    def parse_plot(self) -> AstPlot:
        """PLOT <ident> TYPE <ptype> <plot-opt>*

        Spec: PLOT <ident> TYPE <ptype> <plot-opt>*
        All options are optional and may appear in any order.
        """
        start = self.expect_kw("PLOT")
        name  = self.expect_ident().value
        self.expect_kw("TYPE")
        tok   = self.peek()
        if tok.type in (TokenType.IDENT, TokenType.KW):
            kind = self.consume().value.lower()
        else:
            raise ParseError(
                f"Expected plot type after TYPE, "
                f"got {tok.type.name} {tok.value!r}",
                tok.line,
            )

        x = y = hue = color = title = xlabel = ylabel = save = column = None
        bins: Optional[int] = None

        # Parse zero or more plot options in any order
        PLOT_OPT_KWS = {"X","Y","HUE","COLOR","TITLE","XLABEL","YLABEL",
                        "BINS","FIGSIZE","SAVE","COLUMN"}
        while self.next_is_kw(*PLOT_OPT_KWS):
            kw = self.consume().value
            if kw == "X":
                x = self._strip_quotes(self.expect_string().value)
            elif kw == "Y":
                y = self._strip_quotes(self.expect_string().value)
            elif kw == "HUE":
                hue = self._strip_quotes(self.expect_string().value)
            elif kw == "COLOR":
                color = self._strip_quotes(self.expect_string().value)
            elif kw == "TITLE":
                title = self._strip_quotes(self.expect_string().value)
            elif kw == "XLABEL":
                xlabel = self._strip_quotes(self.expect_string().value)
            elif kw == "YLABEL":
                ylabel = self._strip_quotes(self.expect_string().value)
            elif kw == "BINS":
                bins = int(self.expect_int().value)
            elif kw == "FIGSIZE":
                # FIGSIZE <float> <float>  — consume two floats, ignore (stored elsewhere)
                self._consume_float()
                self._consume_float()
            elif kw == "SAVE":
                save = self._strip_quotes(self.expect_string().value)
            elif kw == "COLUMN":
                column = self._strip_quotes(self.expect_string().value)

        return AstPlot(name=name, kind=kind, x=x, y=y, hue=hue, color=color,
                       title=title, xlabel=xlabel, ylabel=ylabel, bins=bins,
                       save=save, column=column, line=start.line)

    def _consume_float(self) -> float:
        """Consume a FLOAT or INTEGER token and return its float value."""
        tok = self.peek()
        if tok.type is TokenType.FLOAT:
            return float(self.consume().value)
        if tok.type is TokenType.INTEGER:
            return float(self.consume().value)
        raise ParseError(
            f"Expected float value, got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    # -------------------------------------------------------------------------
    # Expressions  (Member C)
    # -------------------------------------------------------------------------

    def parse_expr(self) -> Expr:
        """Parse an arithmetic expression with binary operators.

        Precedence (low to high):
            + -   (additive)
            * / % (multiplicative)
            atoms (literals, column refs, dot-access)

        Spec §2.2.6: <expr> ::= <atom> [<arith-op> <atom>]
        This implementation extends to full left-associative binary exprs.
        """
        return self._parse_additive()

    def _parse_additive(self) -> Expr:
        left = self._parse_multiplicative()
        while self.next_is(TokenType.ARITH_OP) and self.peek().value in ("+", "-"):
            op   = self.consume()
            right = self._parse_multiplicative()
            left  = ExprBinop(op=op.value, left=left, right=right, line=op.line)
        return left

    def _parse_multiplicative(self) -> Expr:
        left = self._parse_atom()
        while True:
            tok = self.peek()
            if tok.type is TokenType.ARITH_OP and tok.value in ("/", "%"):
                op    = self.consume()
                right = self._parse_atom()
                left  = ExprBinop(op=op.value, left=left, right=right, line=op.line)
            elif tok.type is TokenType.STAR:
                # STAR in expression context = multiplication
                op    = self.consume()
                right = self._parse_atom()
                left  = ExprBinop(op="*", left=left, right=right, line=op.line)
            else:
                break
        return left

    def _parse_atom(self) -> Expr:
        """Parse an atomic expression: literal, column ref, or dot-access."""
        tok = self.peek()

        if tok.type is TokenType.INTEGER:
            self.consume()
            return ExprLiteral(tok.value, "integer", tok.line)
        if tok.type is TokenType.FLOAT:
            self.consume()
            return ExprLiteral(tok.value, "float", tok.line)
        if tok.type is TokenType.STRING:
            self.consume()
            return ExprLiteral(tok.value, "string", tok.line)
        if tok.type is TokenType.BOOL:
            self.consume()
            return ExprLiteral(tok.value, "bool", tok.line)

        if tok.type is TokenType.IDENT:
            name_tok = self.consume()
            return ExprColRef(name=name_tok.value, is_loopvar=False, line=name_tok.line)

        if tok.type is TokenType.LOOPVAR:
            lv = self.consume()
            name = lv.value[1:] if lv.value.startswith("$") else lv.value
            return ExprColRef(name=name, is_loopvar=True, line=lv.line)

        raise ParseError(
            f"Expected expression atom (literal or column name), "
            f"got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    # -------------------------------------------------------------------------
    # Conditions  (Member D)
    # -------------------------------------------------------------------------

    def parse_condition(self) -> Condition:
        """Entry point for condition parsing with AND > OR precedence.

        Spec §2.2.4:
            condition AND condition  (AND binds tighter than OR)
            condition OR  condition
        """
        return self._parse_cond_or()

    def _parse_cond_or(self) -> Condition:
        """OR level — lowest precedence, left-associative."""
        left = self._parse_cond_and()
        while self.next_is_kw("OR"):
            tok   = self.consume()
            right = self._parse_cond_and()
            left  = CondOr(left=left, right=right, line=tok.line)
        return left

    def _parse_cond_and(self) -> Condition:
        """AND level — higher precedence than OR, left-associative."""
        left = self._parse_cond_atom()
        while self.next_is_kw("AND"):
            tok   = self.consume()
            right = self._parse_cond_atom()
            left  = CondAnd(left=left, right=right, line=tok.line)
        return left

    def _parse_cond_atom(self) -> Condition:
        """Parse a leaf condition:
            "<col>" <op> <value>
            "<col>" [NOT] IN [...]
            "<col>" IS [NOT] NULL
            "<col>" BETWEEN <num> AND <num>

        The spec uses quoted strings for column names in conditions.
        This parser also accepts unquoted IDENTs and LOOPVARs for
        flexibility inside FOR loop bodies.
        """
        line = self.peek().line

        # Parse left-hand side (column ref or expression)
        left = self._parse_atom()

        tok = self.peek()

        # Comparison: <left> <op> <right>
        if tok.type is TokenType.OP:
            op    = self.consume().value
            right = self._parse_atom()
            return CondCompare(left=left, op=op, right=right, line=line)

        # NOT IN
        if self.next_is_kw("NOT"):
            not_tok = self.consume()
            self.expect_kw("IN")
            if not isinstance(left, ExprColRef):
                raise ParseError("NOT IN requires a column reference", line)
            values = self.parse_value_list()
            return CondIn(col=left, values=values, negate=True, line=not_tok.line)

        # IN
        if self.next_is_kw("IN"):
            self.consume()
            if not isinstance(left, ExprColRef):
                raise ParseError("IN requires a column reference", line)
            values = self.parse_value_list()
            return CondIn(col=left, values=values, negate=False, line=line)

        # IS [NOT] NULL
        if self.next_is_kw("IS"):
            self.consume()
            is_not = False
            if self.next_is_kw("NOT"):
                self.consume()
                is_not = True
            self.expect_kw("NULL")
            if not isinstance(left, ExprColRef):
                raise ParseError("IS NULL requires a column reference", line)
            return CondNull(col=left, is_not=is_not, line=line)

        # BETWEEN <low> AND <high>
        if self.next_is_kw("BETWEEN"):
            self.consume()
            low  = self.parse_value()
            self.expect_kw("AND")
            high = self.parse_value()
            if not isinstance(left, ExprColRef):
                raise ParseError("BETWEEN requires a column reference", line)
            return CondBetween(col=left, low=low, high=high, line=line)

        raise ParseError(
            f"Expected comparison operator, IN, IS, or BETWEEN after column "
            f"reference, got {tok.type.name} {tok.value!r}",
            tok.line,
        )

    # -------------------------------------------------------------------------
    # Control Flow  (Member D)
    # -------------------------------------------------------------------------


    def parse_meta_cond(self) -> Condition:
        """Parse a meta-condition for IF guards.

        Meta-condition forms (spec §5):
            EXISTS <ident>
            EXISTS COLUMN "<col>" IN <ident>
            ROWCOUNT <ident> <op> <int>
            NULLCOUNT <ident> COLUMN "<col>" <op> <int>
            <meta-cond> AND <meta-cond>
            <meta-cond> OR  <meta-cond>

        Uses the same AND>OR precedence as regular conditions.
        """
        return self._parse_meta_or()

    def _parse_meta_or(self) -> Condition:
        left = self._parse_meta_and()
        while self.next_is_kw("OR"):
            tok   = self.consume()
            right = self._parse_meta_and()
            left  = CondOr(left=left, right=right, line=tok.line)
        return left

    def _parse_meta_and(self) -> Condition:
        left = self._parse_meta_atom()
        while self.next_is_kw("AND"):
            tok   = self.consume()
            right = self._parse_meta_atom()
            left  = CondAnd(left=left, right=right, line=tok.line)
        return left

    def _parse_meta_atom(self) -> Condition:
        """Parse one leaf meta-condition."""
        line = self.peek().line

        # EXISTS <ident>
        # EXISTS COLUMN "<col>" IN <ident>
        if self.next_is_kw("EXISTS"):
            self.consume()
            if self.next_is_kw("COLUMN"):
                # EXISTS COLUMN "<col>" IN <ident>
                self.consume()
                col  = self._strip_quotes(self.expect_string().value)
                self.expect_kw("IN")
                df   = self.expect_ident().value
                # Represent as CondCompare with a sentinel op for the semantic validator
                return CondCompare(
                    left  = ExprColRef(name=col, is_loopvar=False, line=line),
                    op    = "exists_col",
                    right = ExprLiteral(value="true", kind="bool", line=line),
                    line  = line,
                )
            else:
                # EXISTS <ident>
                name = self.expect_ident().value
                return CondCompare(
                    left  = ExprColRef(name=name, is_loopvar=False, line=line),
                    op    = "exists",
                    right = ExprLiteral(value="true", kind="bool", line=line),
                    line  = line,
                )

        # ROWCOUNT <ident> <op> <int>
        if self.next_is_kw("ROWCOUNT"):
            self.consume()
            name = self.expect_ident().value
            tok  = self.peek()
            if tok.type is not TokenType.OP:
                raise ParseError(
                    f"Expected comparison operator after ROWCOUNT <ident>, "
                    f"got {tok.type.name} {tok.value!r}",
                    tok.line,
                )
            op  = self.consume().value
            n   = self.expect_int().value
            return CondCompare(
                left  = ExprColRef(name=name, is_loopvar=False, line=line),
                op    = f"rowcount_{op}",
                right = ExprLiteral(value=n, kind="integer", line=line),
                line  = line,
            )

        # NULLCOUNT <ident> COLUMN "<col>" <op> <int>
        if self.next_is_kw("NULLCOUNT"):
            self.consume()
            name = self.expect_ident().value
            self.expect_kw("COLUMN")
            col  = self._strip_quotes(self.expect_string().value)
            tok  = self.peek()
            if tok.type is not TokenType.OP:
                raise ParseError(
                    f"Expected comparison operator after NULLCOUNT ... COLUMN <col>, "
                    f"got {tok.type.name} {tok.value!r}",
                    tok.line,
                )
            op  = self.consume().value
            n   = self.expect_int().value
            return CondCompare(
                left  = ExprColRef(name=col, is_loopvar=False, line=line),
                op    = f"nullcount_{op}",
                right = ExprLiteral(value=n, kind="integer", line=line),
                line  = line,
            )

        raise ParseError(
            f"Expected EXISTS, ROWCOUNT, or NULLCOUNT in IF guard, "
            f"got {self.peek().type.name} {self.peek().value!r}",
            self.peek().line,
        )

    def parse_if(self) -> AstIf:
        """IF <meta-cond> THEN <body> [ELSE <body>] END

        meta-cond uses the same parse_condition() as regular conditions,
        but operands will be meta-condition forms (EXISTS, ROWCOUNT, etc.).
        The parser does not distinguish — the semantic validator enforces
        that IF guards only contain valid meta-condition atoms.
        """
        start = self.expect_kw("IF")
        cond  = self.parse_meta_cond()
        self.expect_kw("THEN")

        then_body: list[ASTNode] = []
        while not self.next_is_kw("ELSE", "END"):
            if self.at_end():
                raise ParseError("Expected ELSE or END after THEN body", self.peek().line)
            then_body.append(self.parse_statement())

        else_body: list[ASTNode] = []
        if self.next_is_kw("ELSE"):
            self.consume()
            while not self.next_is_kw("END"):
                if self.at_end():
                    raise ParseError("Expected END after ELSE body", self.peek().line)
                else_body.append(self.parse_statement())

        self.expect_kw("END")
        return AstIf(condition=cond, then_body=then_body,
                     else_body=else_body, line=start.line)

    def parse_for(self) -> AstFor:
        """FOR EACH $<var> OVER <col-list> DO <body> END"""
        start = self.expect_kw("FOR")
        self.expect_kw("EACH")

        lv_tok = self.peek()
        if lv_tok.type is not TokenType.LOOPVAR:
            raise ParseError(
                f"Expected loop variable ($identifier) after EACH, "
                f"got {lv_tok.type.name} {lv_tok.value!r}",
                lv_tok.line,
            )
        self.consume()
        var = lv_tok.value[1:] if lv_tok.value.startswith("$") else lv_tok.value

        self.expect_kw("OVER")
        columns = self.parse_col_list()
        self.expect_kw("DO")

        body: list[ASTNode] = []
        while not self.next_is_kw("END"):
            if self.at_end():
                raise ParseError("Expected END after FOR body", self.peek().line)
            body.append(self.parse_statement())

        self.expect_kw("END")
        return AstFor(var=var, columns=columns, body=body, line=start.line)


# -----------------------------------------------------------------------------
# 6. PUBLIC ENTRY POINT
# -----------------------------------------------------------------------------

def parse(tokens: list[Token]) -> list[ASTNode]:
    """Parse a PolarPandas token stream into an ordered AST node list.

    Parameters
    ----------
    tokens : list[Token]
        Token sequence from lexer.tokenize(), including terminal EOF.

    Returns
    -------
    list[ASTNode]
        One ASTNode per top-level statement.

    Raises
    ------
    ParseError

    Examples
    --------
    >>> from lexer.lexer import tokenize
    >>> from parser.parser import parse
    >>> parse(tokenize('LOAD "data.csv" AS df'))
    [AstLoad(file='data.csv', name='df', engine=None, line=1)]

    >>> parse(tokenize('CAST df COLUMN "age" TO int'))
    [AstCast(name='df', column='age', dtype='int', line=1)]

    >>> parse(tokenize('GROUP df BY "region" AGGREGATE "amount" AS sum'))
    [AstGroup(name='df', by=['region'], column='amount', agg='sum', line=1)]
    """
    return _Parser(tokens)._parse_all()