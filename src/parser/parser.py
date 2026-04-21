"""PolarPandas v5 parser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from lexer.lexer import Token, TokenType


class ParseError(Exception):
    def __init__(self, message: str, line: int) -> None:
        super().__init__(f"[Parser] line {line}: {message}")
        self.message = message
        self.line = line


@dataclass(frozen=True)
class ExprLiteral:
    value: str
    kind: str
    line: int


@dataclass(frozen=True)
class ExprColRef:
    name: str
    is_loopvar: bool
    line: int


@dataclass(frozen=True)
class ExprBinop:
    op: str
    left: "Expr"
    right: "Expr"
    line: int


Expr = Union[ExprLiteral, ExprColRef, ExprBinop]


@dataclass(frozen=True)
class CondCompare:
    left: Expr
    op: str
    right: Expr
    line: int


@dataclass(frozen=True)
class CondIn:
    col: ExprColRef
    values: list[Expr]
    negate: bool
    line: int


@dataclass(frozen=True)
class CondNull:
    col: ExprColRef
    is_not: bool
    line: int


@dataclass(frozen=True)
class CondBetween:
    col: ExprColRef
    low: Expr
    high: Expr
    line: int


@dataclass(frozen=True)
class CondContains:
    col: ExprColRef
    value: Expr
    negate: bool
    line: int


@dataclass(frozen=True)
class CondAnd:
    left: "Condition"
    right: "Condition"
    line: int


@dataclass(frozen=True)
class CondOr:
    left: "Condition"
    right: "Condition"
    line: int


Condition = Union[CondCompare, CondIn, CondNull, CondBetween, CondContains, CondAnd, CondOr]


@dataclass(frozen=True)
class MetaExists:
    name: str
    line: int


@dataclass(frozen=True)
class MetaRowCount:
    name: str
    op: str
    value: int
    line: int


@dataclass(frozen=True)
class MetaAnd:
    left: "MetaCondition"
    right: "MetaCondition"
    line: int


@dataclass(frozen=True)
class MetaOr:
    left: "MetaCondition"
    right: "MetaCondition"
    line: int


MetaCondition = Union[MetaExists, MetaRowCount, MetaAnd, MetaOr]


@dataclass(frozen=True)
class AstRead:
    path: str
    line: int


@dataclass(frozen=True)
class AstSave:
    path: str
    line: int


@dataclass(frozen=True)
class AstPreview:
    rows: int
    line: int


@dataclass(frozen=True)
class AstInfo:
    line: int


@dataclass(frozen=True)
class AstDropEmpty:
    columns: list[str]
    line: int


@dataclass(frozen=True)
class AstDropDuplicates:
    line: int


@dataclass(frozen=True)
class AstDropColumns:
    columns: list[str]
    line: int


@dataclass(frozen=True)
class AstFillEmpty:
    column: str
    value: Expr
    line: int


@dataclass(frozen=True)
class AstKeep:
    columns: list[str]
    line: int


@dataclass(frozen=True)
class AstCast:
    column: str
    dtype: str
    on_error: Optional[str]
    line: int


@dataclass(frozen=True)
class AstSet:
    column: str
    expr: Expr
    line: int


@dataclass(frozen=True)
class AstRename:
    old: str
    new: str
    line: int


@dataclass(frozen=True)
class AstSort:
    column: str
    direction: str
    line: int


@dataclass(frozen=True)
class AstGroupBy:
    by: list[str]
    agg: str
    column: str
    line: int


@dataclass(frozen=True)
class AstMerge:
    source: str
    source_is_file: bool
    on: str
    how: str
    line: int


@dataclass(frozen=True)
class AstFilter:
    condition: Condition
    line: int


@dataclass(frozen=True)
class AstPlot:
    target: Optional[str]
    kind: str
    x: Optional[str]
    y: Optional[str]
    title: Optional[str]
    save: Optional[str]
    line: int


@dataclass(frozen=True)
class AstPipeline:
    target: Optional[str]
    steps: list["OperationNode"]
    line: int


@dataclass(frozen=True)
class AstIf:
    condition: MetaCondition
    then_body: list["ASTNode"]
    else_body: list["ASTNode"]
    line: int


@dataclass(frozen=True)
class AstFor:
    var: str
    columns: list[str]
    body: list["ASTNode"]
    line: int


OperationNode = Union[
    AstSave,
    AstPreview,
    AstInfo,
    AstDropEmpty,
    AstDropDuplicates,
    AstDropColumns,
    AstFillEmpty,
    AstKeep,
    AstCast,
    AstSet,
    AstRename,
    AstSort,
    AstGroupBy,
    AstMerge,
    AstFilter,
    AstPlot,
]

ASTNode = Union[
    AstRead,
    AstSave,
    AstPreview,
    AstInfo,
    AstDropEmpty,
    AstDropDuplicates,
    AstDropColumns,
    AstFillEmpty,
    AstKeep,
    AstCast,
    AstSet,
    AstRename,
    AstSort,
    AstGroupBy,
    AstMerge,
    AstFilter,
    AstPlot,
    AstPipeline,
    AstIf,
    AstFor,
]


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def peek(self, offset: int = 0) -> Token:
        idx = self._pos + offset
        if idx < len(self._tokens):
            return self._tokens[idx]
        return self._tokens[-1]

    def consume(self) -> Token:
        tok = self.peek()
        if tok.type is TokenType.EOF:
            raise ParseError("Unexpected end of input", tok.line)
        self._pos += 1
        return tok

    def at_end(self) -> bool:
        return self.peek().type is TokenType.EOF

    def expect_kw(self, *keywords: str) -> Token:
        tok = self.peek()
        allowed = {k.upper() for k in keywords}
        if tok.type is TokenType.KW and tok.value in allowed:
            return self.consume()
        expected = " or ".join(f"'{k}'" for k in keywords)
        raise ParseError(f"Expected keyword {expected}, got {tok.type.name} {tok.value!r}", tok.line)

    def expect_ident(self) -> Token:
        tok = self.peek()
        if tok.type is TokenType.IDENT:
            return self.consume()
        raise ParseError(f"Expected identifier, got {tok.type.name} {tok.value!r}", tok.line)

    def expect_string(self) -> Token:
        tok = self.peek()
        if tok.type is TokenType.STRING:
            return self.consume()
        raise ParseError(f"Expected string literal, got {tok.type.name} {tok.value!r}", tok.line)

    def expect_int(self) -> Token:
        tok = self.peek()
        if tok.type is TokenType.NUMBER and "." not in tok.value:
            return self.consume()
        raise ParseError(f"Expected integer number, got {tok.type.name} {tok.value!r}", tok.line)

    def next_is_kw(self, *keywords: str) -> bool:
        tok = self.peek()
        return tok.type is TokenType.KW and tok.value in {k.upper() for k in keywords}

    def next_is(self, ttype: TokenType) -> bool:
        return self.peek().type is ttype

    @staticmethod
    def _strip_quotes(s: str) -> str:
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
            return s[1:-1]
        return s

    def _parse_name(self) -> str:
        tok = self.peek()
        if tok.type in (TokenType.IDENT, TokenType.KW):
            return self.consume().value
        if tok.type is TokenType.STRING:
            return self._strip_quotes(self.consume().value)
        if tok.type is TokenType.LOOPVAR:
            raw = self.consume().value
            return raw[1:] if raw.startswith("$") else raw
        raise ParseError(f"Expected name, got {tok.type.name} {tok.value!r}", tok.line)

    def _parse_col_list(self) -> list[str]:
        cols = [self._parse_name()]
        while self.next_is(TokenType.COMMA):
            self.consume()
            cols.append(self._parse_name())
        return cols

    def _statement_starter(self, tok: Token) -> bool:
        if tok.type is TokenType.PIPE:
            return True
        if tok.type is TokenType.IDENT and self.peek(1).type is TokenType.PIPE:
            return True
        return tok.type is TokenType.KW and tok.value in {
            "READ",
            "SAVE",
            "PREVIEW",
            "INFO",
            "DROP",
            "FILL",
            "KEEP",
            "CAST",
            "SET",
            "RENAME",
            "SORT",
            "GROUP",
            "MERGE",
            "FILTER",
            "PLOT",
            "IF",
            "FOR",
        }

    def parse_literal(self) -> ExprLiteral:
        tok = self.peek()
        if tok.type is TokenType.STRING:
            return ExprLiteral(value=self.consume().value, kind="string", line=tok.line)
        if tok.type is TokenType.NUMBER:
            kind = "float" if "." in tok.value else "integer"
            return ExprLiteral(value=self.consume().value, kind=kind, line=tok.line)
        if tok.type is TokenType.BOOL:
            return ExprLiteral(value=self.consume().value, kind="bool", line=tok.line)
        raise ParseError(f"Expected literal value, got {tok.type.name} {tok.value!r}", tok.line)

    def _parse_atom(self) -> Expr:
        tok = self.peek()
        if tok.type in (TokenType.STRING, TokenType.NUMBER, TokenType.BOOL):
            return self.parse_literal()
        if tok.type is TokenType.LOOPVAR:
            lv = self.consume().value
            return ExprColRef(name=lv[1:] if lv.startswith("$") else lv, is_loopvar=True, line=tok.line)
        if tok.type in (TokenType.IDENT, TokenType.KW):
            return ExprColRef(name=self.consume().value, is_loopvar=False, line=tok.line)
        if tok.type is TokenType.LPAREN:
            self.consume()
            inner = self.parse_expr()
            if not self.next_is(TokenType.RPAREN):
                raise ParseError("Expected ')' in expression", self.peek().line)
            self.consume()
            return inner
        raise ParseError(f"Expected expression atom, got {tok.type.name} {tok.value!r}", tok.line)

    def _parse_term(self) -> Expr:
        left = self._parse_atom()
        while self.next_is(TokenType.ARITH_OP) and self.peek().value in ("*", "/"):
            op = self.consume()
            right = self._parse_atom()
            left = ExprBinop(op=op.value, left=left, right=right, line=op.line)
        return left

    def parse_expr(self) -> Expr:
        left = self._parse_term()
        while self.next_is(TokenType.ARITH_OP) and self.peek().value in ("+", "-"):
            op = self.consume()
            right = self._parse_term()
            left = ExprBinop(op=op.value, left=left, right=right, line=op.line)
        return left

    def _parse_cond_atom(self) -> Condition:
        if self.next_is(TokenType.LPAREN):
            self.consume()
            c = self.parse_condition()
            if not self.next_is(TokenType.RPAREN):
                raise ParseError("Expected ')' in condition", self.peek().line)
            self.consume()
            return c

        left = self._parse_atom()
        line = self.peek(-1).line if self._pos > 0 else self.peek().line

        if self.next_is(TokenType.OP):
            op = self.consume().value
            right = self._parse_atom()
            return CondCompare(left=left, op=op, right=right, line=line)

        if self.next_is_kw("NOT"):
            self.consume()
            if self.next_is_kw("IN"):
                self.consume()
                if not isinstance(left, ExprColRef):
                    raise ParseError("NOT IN requires column reference", line)
                values = [self._parse_atom()]
                while self.next_is(TokenType.COMMA):
                    self.consume()
                    values.append(self._parse_atom())
                return CondIn(col=left, values=values, negate=True, line=line)
            if self.next_is_kw("CONTAINS"):
                self.consume()
                if not isinstance(left, ExprColRef):
                    raise ParseError("NOT CONTAINS requires column reference", line)
                value = self._parse_atom()
                return CondContains(col=left, value=value, negate=True, line=line)
            raise ParseError("Expected IN or CONTAINS after NOT", self.peek().line)

        if self.next_is_kw("IN"):
            self.consume()
            if not isinstance(left, ExprColRef):
                raise ParseError("IN requires column reference", line)
            values = [self._parse_atom()]
            while self.next_is(TokenType.COMMA):
                self.consume()
                values.append(self._parse_atom())
            return CondIn(col=left, values=values, negate=False, line=line)

        if self.next_is_kw("IS"):
            self.consume()
            is_not = False
            if self.next_is_kw("NOT"):
                self.consume()
                is_not = True
            self.expect_kw("EMPTY")
            if not isinstance(left, ExprColRef):
                raise ParseError("IS EMPTY requires column reference", line)
            return CondNull(col=left, is_not=is_not, line=line)

        if self.next_is_kw("BETWEEN"):
            self.consume()
            low = self._parse_atom()
            self.expect_kw("AND")
            high = self._parse_atom()
            if not isinstance(left, ExprColRef):
                raise ParseError("BETWEEN requires column reference", line)
            return CondBetween(col=left, low=low, high=high, line=line)

        if self.next_is_kw("CONTAINS"):
            self.consume()
            if not isinstance(left, ExprColRef):
                raise ParseError("CONTAINS requires column reference", line)
            value = self._parse_atom()
            return CondContains(col=left, value=value, negate=False, line=line)

        raise ParseError("Expected condition operator", self.peek().line)

    def _parse_cond_and(self) -> Condition:
        left = self._parse_cond_atom()
        while self.next_is_kw("AND"):
            tok = self.consume()
            right = self._parse_cond_atom()
            left = CondAnd(left=left, right=right, line=tok.line)
        return left

    def parse_condition(self) -> Condition:
        left = self._parse_cond_and()
        while self.next_is_kw("OR"):
            tok = self.consume()
            right = self._parse_cond_and()
            left = CondOr(left=left, right=right, line=tok.line)
        return left

    def _parse_meta_atom(self) -> MetaCondition:
        if self.next_is(TokenType.LPAREN):
            self.consume()
            c = self.parse_meta_condition()
            if not self.next_is(TokenType.RPAREN):
                raise ParseError("Expected ')' in IF condition", self.peek().line)
            self.consume()
            return c

        if self.next_is_kw("EXISTS"):
            tok = self.consume()
            name = self._parse_name()
            return MetaExists(name=name, line=tok.line)

        if self.next_is_kw("ROWCOUNT"):
            tok = self.consume()
            name = self._parse_name()
            op_tok = self.peek()
            if op_tok.type is not TokenType.OP:
                raise ParseError("Expected comparison operator after ROWCOUNT <df>", op_tok.line)
            op = self.consume().value
            n = int(self.expect_int().value)
            return MetaRowCount(name=name, op=op, value=n, line=tok.line)

        raise ParseError("IF guard must use EXISTS or ROWCOUNT", self.peek().line)

    def _parse_meta_and(self) -> MetaCondition:
        left = self._parse_meta_atom()
        while self.next_is_kw("AND"):
            tok = self.consume()
            right = self._parse_meta_atom()
            left = MetaAnd(left=left, right=right, line=tok.line)
        return left

    def parse_meta_condition(self) -> MetaCondition:
        left = self._parse_meta_and()
        while self.next_is_kw("OR"):
            tok = self.consume()
            right = self._parse_meta_and()
            left = MetaOr(left=left, right=right, line=tok.line)
        return left

    def parse_read(self) -> AstRead:
        start = self.expect_kw("READ")
        path = self._strip_quotes(self.expect_string().value)
        return AstRead(path=path, line=start.line)

    def parse_save(self) -> AstSave:
        start = self.expect_kw("SAVE")
        path = self._strip_quotes(self.expect_string().value)
        return AstSave(path=path, line=start.line)

    def parse_preview(self) -> AstPreview:
        start = self.expect_kw("PREVIEW")
        rows = 5
        if self.next_is(TokenType.NUMBER):
            tok = self.peek()
            if "." in tok.value:
                raise ParseError("PREVIEW expects an integer row count", tok.line)
            rows = int(self.consume().value)
        return AstPreview(rows=rows, line=start.line)

    def parse_info(self) -> AstInfo:
        start = self.expect_kw("INFO")
        return AstInfo(line=start.line)

    def parse_drop(self) -> Union[AstDropEmpty, AstDropDuplicates, AstDropColumns]:
        start = self.expect_kw("DROP")
        if self.next_is_kw("EMPTY"):
            self.consume()
            columns: list[str] = []
            if not self.at_end() and not self._statement_starter(self.peek()) and not self.next_is_kw("ELSE", "END"):
                columns = self._parse_col_list()
            return AstDropEmpty(columns=columns, line=start.line)
        if self.next_is_kw("DUPLICATES"):
            self.consume()
            return AstDropDuplicates(line=start.line)
        columns = self._parse_col_list()
        return AstDropColumns(columns=columns, line=start.line)

    def parse_fill(self) -> AstFillEmpty:
        start = self.expect_kw("FILL")
        self.expect_kw("EMPTY")
        column = self._parse_name()
        self.expect_kw("WITH")
        value = self._parse_atom()
        return AstFillEmpty(column=column, value=value, line=start.line)

    def parse_keep(self) -> AstKeep:
        start = self.expect_kw("KEEP")
        return AstKeep(columns=self._parse_col_list(), line=start.line)

    def parse_cast(self) -> AstCast:
        start = self.expect_kw("CAST")
        column = self._parse_name()
        self.expect_kw("TO")
        dtype = self._parse_name().upper()
        on_error = None
        if self.next_is(TokenType.LPAREN):
            self.consume()
            self.expect_kw("ON")
            self.expect_kw("ERROR")
            mode = self.expect_kw("SKIP", "STOP")
            on_error = mode.value
            if not self.next_is(TokenType.RPAREN):
                raise ParseError("Expected ')' after ON ERROR clause", self.peek().line)
            self.consume()
        return AstCast(column=column, dtype=dtype, on_error=on_error, line=start.line)

    def parse_set(self) -> AstSet:
        start = self.expect_kw("SET")
        column = self._parse_name()
        if not self.next_is(TokenType.ASSIGN):
            raise ParseError("Expected '=' in SET statement", self.peek().line)
        self.consume()
        expr = self.parse_expr()
        return AstSet(column=column, expr=expr, line=start.line)

    def parse_rename(self) -> AstRename:
        start = self.expect_kw("RENAME")
        old = self._parse_name()
        self.expect_kw("TO")
        new = self._parse_name()
        return AstRename(old=old, new=new, line=start.line)

    def parse_sort(self) -> AstSort:
        start = self.expect_kw("SORT")
        self.expect_kw("BY")
        column = self._parse_name()
        direction = "ASC"
        if self.next_is_kw("ASC", "DESC"):
            direction = self.consume().value
        return AstSort(column=column, direction=direction, line=start.line)

    def parse_group(self) -> AstGroupBy:
        start = self.expect_kw("GROUP")
        self.expect_kw("BY")
        by = self._parse_col_list()
        agg = self._parse_name().upper()
        column = self._parse_name()
        return AstGroupBy(by=by, agg=agg, column=column, line=start.line)

    def parse_merge(self) -> AstMerge:
        start = self.expect_kw("MERGE")
        source_tok = self.peek()
        source_is_file = source_tok.type is TokenType.STRING
        source = self._parse_name()
        self.expect_kw("ON")
        on = self._parse_name()
        how = "INNER"
        if self.peek().type in (TokenType.IDENT, TokenType.KW):
            candidate = self.peek().value.upper()
            if candidate in {"INNER", "LEFT", "RIGHT", "OUTER"}:
                how = self.consume().value.upper()
        return AstMerge(source=source, source_is_file=source_is_file, on=on, how=how, line=start.line)

    def parse_filter(self) -> AstFilter:
        start = self.expect_kw("FILTER")
        self.expect_kw("WHERE")
        return AstFilter(condition=self.parse_condition(), line=start.line)

    def parse_plot(self) -> AstPlot:
        start = self.expect_kw("PLOT")
        target = None
        if self.peek().type in (TokenType.IDENT, TokenType.KW) and not self.next_is_kw("TYPE"):
            target = self._parse_name()
        self.expect_kw("TYPE")
        kind = self._parse_name().lower()

        x = y = title = save = None
        while self.next_is_kw("X", "Y", "TITLE", "SAVE"):
            kw = self.consume().value
            if kw == "X":
                x = self._parse_name()
            elif kw == "Y":
                y = self._parse_name()
            elif kw == "TITLE":
                title = self._strip_quotes(self.expect_string().value)
            elif kw == "SAVE":
                save = self._strip_quotes(self.expect_string().value)
        return AstPlot(target=target, kind=kind, x=x, y=y, title=title, save=save, line=start.line)

    def parse_operation(self) -> OperationNode:
        if self.next_is_kw("SAVE"):
            return self.parse_save()
        if self.next_is_kw("PREVIEW"):
            return self.parse_preview()
        if self.next_is_kw("INFO"):
            return self.parse_info()
        if self.next_is_kw("DROP"):
            return self.parse_drop()
        if self.next_is_kw("FILL"):
            return self.parse_fill()
        if self.next_is_kw("KEEP"):
            return self.parse_keep()
        if self.next_is_kw("CAST"):
            return self.parse_cast()
        if self.next_is_kw("SET"):
            return self.parse_set()
        if self.next_is_kw("RENAME"):
            return self.parse_rename()
        if self.next_is_kw("SORT"):
            return self.parse_sort()
        if self.next_is_kw("GROUP"):
            return self.parse_group()
        if self.next_is_kw("MERGE"):
            return self.parse_merge()
        if self.next_is_kw("FILTER"):
            return self.parse_filter()
        if self.next_is_kw("PLOT"):
            return self.parse_plot()
        raise ParseError("Expected operation statement", self.peek().line)

    def parse_pipeline(self) -> AstPipeline:
        start_line = self.peek().line
        target = None
        if self.peek().type is TokenType.IDENT and self.peek(1).type is TokenType.PIPE:
            target = self.consume().value
        if not self.next_is(TokenType.PIPE):
            raise ParseError("Expected '->' to start pipeline", self.peek().line)
        self.consume()
        steps: list[OperationNode] = [self.parse_operation()]
        while self.next_is(TokenType.PIPE):
            self.consume()
            steps.append(self.parse_operation())
        return AstPipeline(target=target, steps=steps, line=start_line)

    def parse_if(self) -> AstIf:
        start = self.expect_kw("IF")
        cond = self.parse_meta_condition()
        if not self.next_is(TokenType.COLON):
            raise ParseError("Expected ':' after IF condition", self.peek().line)
        self.consume()

        then_body: list[ASTNode] = []
        while not self.at_end() and not self.next_is_kw("ELSE", "END"):
            then_body.append(self.parse_statement())

        else_body: list[ASTNode] = []
        if self.next_is_kw("ELSE"):
            self.consume()
            while not self.at_end() and not self.next_is_kw("END"):
                else_body.append(self.parse_statement())

        self.expect_kw("END")
        return AstIf(condition=cond, then_body=then_body, else_body=else_body, line=start.line)

    def parse_for(self) -> AstFor:
        start = self.expect_kw("FOR")
        lv = self.peek()
        if lv.type is not TokenType.LOOPVAR:
            raise ParseError("Expected loop variable after FOR", lv.line)
        var = self.consume().value[1:]
        self.expect_kw("IN")
        columns = self._parse_col_list()
        if not self.next_is(TokenType.COLON):
            raise ParseError("Expected ':' after FOR column list", self.peek().line)
        self.consume()

        body: list[ASTNode] = []
        while not self.at_end() and not self.next_is_kw("END"):
            body.append(self.parse_statement())
        self.expect_kw("END")
        return AstFor(var=var, columns=columns, body=body, line=start.line)

    def parse_statement(self) -> ASTNode:
        tok = self.peek()
        if tok.type is TokenType.PIPE or (tok.type is TokenType.IDENT and self.peek(1).type is TokenType.PIPE):
            return self.parse_pipeline()

        if tok.type is not TokenType.KW:
            raise ParseError(f"Expected statement keyword, got {tok.type.name} {tok.value!r}", tok.line)

        if tok.value == "READ":
            return self.parse_read()
        if tok.value in {
            "SAVE",
            "PREVIEW",
            "INFO",
            "DROP",
            "FILL",
            "KEEP",
            "CAST",
            "SET",
            "RENAME",
            "SORT",
            "GROUP",
            "MERGE",
            "FILTER",
            "PLOT",
        }:
            return self.parse_operation()
        if tok.value == "IF":
            return self.parse_if()
        if tok.value == "FOR":
            return self.parse_for()

        raise ParseError(f"Unknown statement keyword {tok.value!r}", tok.line)

    def _parse_all(self) -> list[ASTNode]:
        nodes: list[ASTNode] = []
        while not self.at_end():
            nodes.append(self.parse_statement())
        return nodes


def parse(tokens: list[Token]) -> list[ASTNode]:
    return _Parser(tokens)._parse_all()
