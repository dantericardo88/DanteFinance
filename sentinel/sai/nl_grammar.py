"""
Recursive-descent grammar for financial screening queries (dim_076).

Grammar:
  query      -> or_expr
  or_expr    -> and_expr ('OR' and_expr)*
  and_expr   -> not_expr ('AND' not_expr)*
  not_expr   -> 'NOT' not_expr | atom
  atom       -> '(' or_expr ')' | comparison
  comparison -> field op value
              | field 'BETWEEN' value 'AND' value
              | field 'IN' '(' value_list ')'
  op         -> '>' | '<' | '>=' | '<=' | '=' | '==' | '!='
  value      -> NUMBER | STRING | PERCENT | DOLLAR | IDENTIFIER

Field aliases (e.g. pe -> price_to_earnings, mcap -> market_cap) are normalised
during ``_normalize_field``.

This module is consumed by ``sentinel.sai.nl_screener_v3.QueryParser`` for
complex queries containing AND/OR/NOT/parens. Simple natural-language queries
continue to use the legacy regex-based parser for backward compatibility.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------

@dataclass
class Comparison:
    """Leaf node: ``field <op> value`` (optionally with ``value2`` for BETWEEN)."""
    field: str
    op: str   # >, <, >=, <=, ==, !=, BETWEEN, IN
    value: Any
    value2: Any = None  # for BETWEEN


@dataclass
class BoolExpr:
    """Interior node: AND/OR/NOT over child nodes."""
    op: str   # 'AND', 'OR', 'NOT'
    children: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Field-alias table (extends the canonical aliases used by NL screener V3)
# ---------------------------------------------------------------------------

_FIELD_ALIASES = {
    # Valuation
    "pe": "price_to_earnings",
    "p/e": "price_to_earnings",
    "pe_ratio": "price_to_earnings",
    "price_to_earnings": "price_to_earnings",
    "pb": "price_to_book",
    "p/b": "price_to_book",
    "ps": "price_to_sales",
    "p/s": "price_to_sales",
    "ev_ebitda": "ev_to_ebitda",
    "ev/ebitda": "ev_to_ebitda",
    # Size / cap
    "mcap": "market_cap",
    "mkt_cap": "market_cap",
    "marketcap": "market_cap",
    "market_cap": "market_cap",
    "cap": "market_cap",
    # Profitability / returns
    "roe": "return_on_equity",
    "roi": "return_on_investment",
    "roa": "return_on_assets",
    "eps": "earnings_per_share",
    # Income / cashflow
    "rev": "revenue",
    "sales": "revenue",
    "revenue": "revenue",
    "ni": "net_income",
    "net_income": "net_income",
    "fcf": "free_cash_flow",
    # Margins
    "margin": "profit_margin",
    "gpm": "gross_margin",
    "opm": "operating_margin",
    "npm": "net_margin",
    # Distributions
    "div": "dividend_yield",
    "yield": "dividend_yield",
    "dividend": "dividend_yield",
    # Leverage
    "debt": "debt_to_equity",
    "d/e": "debt_to_equity",
    "de": "debt_to_equity",
    # Categorical
    "sector": "sector",
    "industry": "industry",
    "ticker": "ticker",
    "symbol": "ticker",
    "exchange": "exchange",
    "country": "country",
}

_KEYWORDS = {"AND", "OR", "NOT", "BETWEEN", "IN"}
_OPS = {">", "<", ">=", "<=", "=", "==", "!="}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class NLGrammarParser:
    """
    Recursive-descent parser for boolean-composite screener queries.

    Public surface:
        parser = NLGrammarParser()
        ast = parser.parse("pe < 20 AND roe > 15%")
        # ast is a Comparison or BoolExpr tree
    """

    def __init__(self) -> None:
        self.tokens: List[str] = []
        self.pos: int = 0

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def parse(self, query: str) -> Union[Comparison, BoolExpr]:
        if not query or not query.strip():
            raise SyntaxError("Empty query")
        self.tokens = self._tokenize(query)
        self.pos = 0
        result = self._parse_or()
        if self.pos < len(self.tokens):
            raise SyntaxError(
                f"Unexpected token at position {self.pos}: {self.tokens[self.pos]!r}"
            )
        return result

    # ------------------------------------------------------------------
    # Lexer
    # ------------------------------------------------------------------
    def _tokenize(self, q: str) -> List[str]:
        """
        Tokenize the query into a flat list of operator/keyword/value tokens.

        Handles:
            - Multi-char comparison ops: ``>=``, ``<=``, ``==``, ``!=``
            - Single-char syntax: ``> < = ( ) ,``
            - Numeric suffixes (``%``, ``k``, ``m``, ``b``, ``t``) kept attached
            - Quoted strings (single or double quoted) preserved as one token
            - Keywords case-folded later in the parser
        """
        out: List[str] = []
        i = 0
        n = len(q)
        while i < n:
            ch = q[i]

            # Whitespace
            if ch.isspace():
                i += 1
                continue

            # Quoted string -> single token (quotes stripped later by _parse_value)
            if ch in ('"', "'"):
                quote = ch
                j = i + 1
                while j < n and q[j] != quote:
                    if q[j] == "\\" and j + 1 < n:
                        j += 2
                    else:
                        j += 1
                out.append(q[i:j + 1])
                i = j + 1
                continue

            # Two-char operators
            if i + 1 < n:
                two = q[i:i + 2]
                if two in (">=", "<=", "==", "!="):
                    out.append(two)
                    i += 2
                    continue

            # Single-char punctuation
            if ch in "()<>=,":
                out.append(ch)
                i += 1
                continue

            # Otherwise: scan identifier/number-like token until a delimiter
            j = i
            while j < n and not q[j].isspace() and q[j] not in "()<>=,!":
                j += 1
            out.append(q[i:j])
            i = j

        return out

    # ------------------------------------------------------------------
    # Recursive-descent productions
    # ------------------------------------------------------------------
    def _peek(self) -> Optional[str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _peek_upper(self) -> Optional[str]:
        tok = self._peek()
        return tok.upper() if tok is not None else None

    def _consume(self) -> str:
        if self.pos >= len(self.tokens):
            raise SyntaxError("Unexpected end of input")
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _parse_or(self) -> Union[Comparison, BoolExpr]:
        left = self._parse_and()
        while self._peek_upper() == "OR":
            self._consume()
            right = self._parse_and()
            # Flatten nested ORs into a single n-ary BoolExpr for readability
            if isinstance(left, BoolExpr) and left.op == "OR":
                left.children.append(right)
            else:
                left = BoolExpr("OR", [left, right])
        return left

    def _parse_and(self) -> Union[Comparison, BoolExpr]:
        left = self._parse_not()
        while self._peek_upper() == "AND":
            self._consume()
            right = self._parse_not()
            if isinstance(left, BoolExpr) and left.op == "AND":
                left.children.append(right)
            else:
                left = BoolExpr("AND", [left, right])
        return left

    def _parse_not(self) -> Union[Comparison, BoolExpr]:
        if self._peek_upper() == "NOT":
            self._consume()
            return BoolExpr("NOT", [self._parse_not()])
        return self._parse_atom()

    def _parse_atom(self) -> Union[Comparison, BoolExpr]:
        tok = self._peek()
        if tok is None:
            raise SyntaxError("Unexpected end of input while parsing atom")
        if tok == "(":
            self._consume()
            result = self._parse_or()
            if self._peek() != ")":
                raise SyntaxError(
                    f"Expected ')' at position {self.pos}, got "
                    f"{self._peek()!r}"
                )
            self._consume()
            return result
        return self._parse_comparison()

    def _parse_comparison(self) -> Comparison:
        field_tok = self._consume()
        canonical_field = self._normalize_field(field_tok)

        op_tok_raw = self._peek()
        if op_tok_raw is None:
            raise SyntaxError(
                f"Expected operator after field {field_tok!r}, got EOF"
            )
        op_tok = op_tok_raw.upper()

        # BETWEEN <v1> AND <v2>
        if op_tok == "BETWEEN":
            self._consume()
            v1_tok = self._consume()
            v1 = self._parse_value(v1_tok)
            and_tok = self._consume()
            if and_tok.upper() != "AND":
                raise SyntaxError(
                    f"Expected AND in BETWEEN clause, got {and_tok!r}"
                )
            v2_tok = self._consume()
            v2 = self._parse_value(v2_tok)
            return Comparison(canonical_field, "BETWEEN", v1, v2)

        # IN ( v1 , v2 , ... )
        if op_tok == "IN":
            self._consume()
            open_paren = self._consume()
            if open_paren != "(":
                raise SyntaxError(
                    f"Expected '(' after IN, got {open_paren!r}"
                )
            values: List[Any] = []
            while True:
                tok = self._peek()
                if tok is None:
                    raise SyntaxError("Unterminated IN list")
                if tok == ")":
                    self._consume()
                    break
                if tok == ",":
                    self._consume()
                    continue
                values.append(self._parse_value(self._consume()))
            return Comparison(canonical_field, "IN", values)

        # Standard comparison
        if op_tok not in _OPS:
            raise SyntaxError(
                f"Expected comparison operator after field "
                f"{field_tok!r}, got {op_tok_raw!r}"
            )
        self._consume()
        value_tok = self._consume()
        value = self._parse_value(value_tok)

        # Normalise '=' -> '==' for downstream consumers
        canonical_op = "==" if op_tok == "=" else op_tok
        return Comparison(canonical_field, canonical_op, value)

    # ------------------------------------------------------------------
    # Value coercion
    # ------------------------------------------------------------------
    def _parse_value(self, tok: str) -> Any:
        """Coerce a value token into a Python primitive.

        Handles ``$1.5b``, ``50%``, ``2.5x``, ``1_000``, ``"text"``,
        ``'text'``, plain identifiers, and bare strings.
        """
        # Quoted -> strip
        if len(tok) >= 2 and tok[0] in ('"', "'") and tok[-1] == tok[0]:
            return tok[1:-1]

        tok_lc = tok.lower().replace(",", "").replace("_", "").replace("$", "")
        multiplier = 1.0

        # Percent
        if tok_lc.endswith("%"):
            try:
                return float(tok_lc[:-1]) / 100.0
            except ValueError:
                pass

        # Multiplier suffix
        suffix_map = {
            "k": 1_000.0,
            "m": 1_000_000.0,
            "mm": 1_000_000.0,
            "b": 1_000_000_000.0,
            "bn": 1_000_000_000.0,
            "t": 1_000_000_000_000.0,
        }
        for suf, mult in sorted(suffix_map.items(), key=lambda kv: -len(kv[0])):
            if tok_lc.endswith(suf):
                stem = tok_lc[: -len(suf)]
                try:
                    return float(stem) * mult
                except ValueError:
                    break

        # "x" or "times" multiplier on ratios -> just a number
        if tok_lc.endswith("x"):
            try:
                return float(tok_lc[:-1])
            except ValueError:
                pass

        # Plain number
        try:
            return float(tok_lc)
        except ValueError:
            pass

        # Fallback: bare identifier / string
        return tok.strip("\"'")

    def _normalize_field(self, field_tok: str) -> str:
        """Map shorthand field names to canonical identifiers."""
        key = field_tok.lower()
        return _FIELD_ALIASES.get(key, key)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def parse_screener_query(query: str) -> dict:
    """
    Parse an NL query string into a nested dict representation.

    Example::

        >>> parse_screener_query("pe < 20 AND roe > 15%")
        {'and': [{'price_to_earnings': {'<': 20.0}},
                 {'return_on_equity': {'>': 0.15}}]}

        >>> parse_screener_query("pe BETWEEN 10 AND 20")
        {'price_to_earnings': {'between': [10.0, 20.0]}}
    """
    parser = NLGrammarParser()
    ast = parser.parse(query)
    return _ast_to_dict(ast)


def _ast_to_dict(node: Union[Comparison, BoolExpr]) -> dict:
    """Serialise an AST node to a nested filter-style dict."""
    if isinstance(node, Comparison):
        if node.op == "BETWEEN":
            return {node.field: {"between": [node.value, node.value2]}}
        if node.op == "IN":
            return {node.field: {"in": list(node.value)}}
        return {node.field: {node.op: node.value}}
    # BoolExpr
    op_key = node.op.lower()
    return {op_key: [_ast_to_dict(c) for c in node.children]}


def is_complex_query(query: str) -> bool:
    """
    Heuristic: does the query require the recursive-descent grammar
    (parens, NOT, BETWEEN, IN, or explicit AND/OR boolean composition)?

    The legacy regex-based parser in ``nl_screener_v3.QueryParser`` is
    sufficient (and friendlier) for plain English. This predicate lets the
    screener route only structured queries to the grammar engine.
    """
    if not query:
        return False
    q = query.strip()
    if "(" in q or ")" in q:
        return True
    upper = " " + q.upper() + " "
    if any(kw in upper for kw in (" NOT ", " BETWEEN ", " IN ")):
        return True
    if any(op in q for op in (">=", "<=", "!=", "==")):
        return True
    # Need *both* AND/OR composition AND at least one explicit operator
    if (" AND " in upper or " OR " in upper) and any(
        op in q for op in (">", "<", "=")
    ):
        return True
    return False


__all__ = [
    "Comparison",
    "BoolExpr",
    "NLGrammarParser",
    "parse_screener_query",
    "is_complex_query",
]
