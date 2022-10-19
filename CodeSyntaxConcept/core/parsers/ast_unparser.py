from __future__ import print_function, unicode_literals
"""Modified from -- >"""
"https://github.com/simonpercivall/astunparse/blob/2acce01fcdda2ea32eea835c30ccca21aaff7297/lib/astunparse/unparser.py"

"Usage: unparse.py <path to source file>"
from six.moves import cStringIO
import six
import sys
import ast
import tokenize

from numpy import unicode
from six import StringIO

# Large float and imaginary literals get turned into infinities in the AST.
# We unparse those infinities to INFSTR.
INFSTR = "1e" + repr(sys.float_info.max_10_exp + 1)

'''DEPRECATED'''
def interleave(inter, f, seq):
    """Call f on each item in seq, calling inter() in between.
    """
    seq = iter(seq)
    try:
        f(next(seq))
    except StopIteration:
        pass
    else:
        for x in seq:
            inter()
            f(x)


class CustomUnparser:
    """Methods in this class recursively traverse an AST and
    output source code for the abstract syntax; original formatting
    is disregarded. """

    def __init__(self, tree, tokens, file=sys.stdout):
        """Unparser(tree, file=sys.stdout) -> None.
         Print the source for tree to file."""
        self.tokens = tokens
        self.f = file
        self.future_imports = []
        self._indent = 0
        self.dispatch(tree)
        print("", file=self.f)
        self.f.flush()

    def fill(self, text="", family="_Ident"):
        "Indent a piece of text, according to the current indentation level"
        self.tokens.append({"token": "\n" + "    " * self._indent + text, "family": family})
        self.f.write("\n" + "    " * self._indent + text)

    def write(self, text, family):
        "Append a piece of text to the current line."
        self.tokens.append({"token": six.text_type(text), "family": family})
        self.f.write(six.text_type(text))

    def enter(self, family="_Ident"):
        "Print ':', and increase the indentation."
        self.write(":", family)
        self._indent += 1

    def leave(self):
        "Decrease the indentation level."
        self._indent -= 1

    def dispatch(self, tree):
        "Dispatcher function, dispatching tree type T to method _T."
        if isinstance(tree, list):
            for t in tree:
                self.dispatch(t)
            return
        meth = getattr(self, "_" + tree.__class__.__name__)
        meth(tree)

    ############### Unparsing methods ######################
    # There should be one method per concrete grammar type #
    # Constructors should be grouped by sum type. Ideally, #
    # this would follow the order in the grammar, but      #
    # currently doesn't.                                   #
    ########################################################

    def _Module(self, tree):
        for stmt in tree.body:
            self.dispatch(stmt)

    def _Interactive(self, tree):
        for stmt in tree.body:
            self.dispatch(stmt)

    def _Expression(self, tree):
        self.dispatch(tree.body)

    # stmt
    def _Expr(self, tree):
        self.fill(family="_Expr")
        self.dispatch(tree.value)

    def _NamedExpr(self, tree):
        self.write("(", '_NamedExpr')
        self.dispatch(tree.target)
        self.write(" := ", '_NamedExpr')
        self.dispatch(tree.value)
        self.write(")", '_NamedExpr')

    def _Import(self, t):
        self.fill("import ", family="_Import")
        interleave(lambda: self.write(", ", '_Import'), self.dispatch, t.names)

    def _ImportFrom(self, t):
        # A from __future__ import may affect unparsing, so record it.
        if t.module and t.module == '__future__':
            self.future_imports.extend(n.name for n in t.names)
        self.fill("from ", family="_ImportFrom")
        self.write("." * t.level, '_ImportFrom')
        if t.module:
            self.write(t.module, '_ImportFrom')
        self.write(" import ", '_ImportFrom')
        interleave(lambda: self.write(", ", '_ImportFrom'), self.dispatch, t.names)

    def _Assign(self, t):
        self.fill(family="_Assign")
        for target in t.targets:
            self.dispatch(target)
            self.write(" = ", '_Assign')
        self.dispatch(t.value)

    def _AugAssign(self, t):
        self.fill(family="_AugAssign")
        self.dispatch(t.target)
        self.write(" " + self.binop[t.op.__class__.__name__] + "= ", '_AugAssign')
        self.dispatch(t.value)

    def _AnnAssign(self, t):
        self.fill(family="_AnnAssign")
        if not t.simple and isinstance(t.target, ast.Name):
            self.write('(', '_AnnAssign')
        self.dispatch(t.target)
        if not t.simple and isinstance(t.target, ast.Name):
            self.write(')', '_AnnAssign')
        self.write(": ", '_AnnAssign')
        self.dispatch(t.annotation)
        if t.value:
            self.write(" = ", '_AnnAssign')
            self.dispatch(t.value)

    def _Return(self, t):
        self.fill("return", family="_Return")
        if t.value:
            self.write(" ", '_Return')
            self.dispatch(t.value)

    def _Pass(self, t):
        self.fill("pass", family="_Pass")

    def _Break(self, t):
        self.fill("break", family="_Break")

    def _Continue(self, t):
        self.fill("continue", family="_Continue")

    def _Delete(self, t):
        self.fill("del ", family="_Delete")
        interleave(lambda: self.write(", ", '_Delete'), self.dispatch, t.targets)

    def _Assert(self, t):
        self.fill("assert ", family="_Assert")
        self.dispatch(t.test)
        if t.msg:
            self.write(", ", '_Assert')
            self.dispatch(t.msg)

    def _Exec(self, t):
        self.fill("exec ", family="_Exec")
        self.dispatch(t.body)
        if t.globals:
            self.write(" in ", '_Exec')
            self.dispatch(t.globals)
        if t.locals:
            self.write(", ", '_Exec')
            self.dispatch(t.locals)

    def _Print(self, t):
        self.fill("print ", family="_Print")
        do_comma = False
        if t.dest:
            self.write(">>", '_Print')
            self.dispatch(t.dest)
            do_comma = True
        for e in t.values:
            if do_comma:
                self.write(", ", '_Print')
            else:
                do_comma = True
            self.dispatch(e)
        if not t.nl:
            self.write(",", '_Print')

    def _Global(self, t):
        self.fill("global ", family="_Global")
        interleave(lambda: self.write(", ", '_Global'), self.write, t.names)

    def _Nonlocal(self, t):
        self.fill("nonlocal ", family="_Nonlocal")
        interleave(lambda: self.write(", ", '_Nonlocal'), self.write, t.names)

    def _Await(self, t):
        self.write("(", '_Await')
        self.write("await", '_Await')
        if t.value:
            self.write(" ", '_Await')
            self.dispatch(t.value)
        self.write(")", '_Await')

    def _Yield(self, t):
        self.write("(", '_Yield')
        self.write("yield", '_Yield')
        if t.value:
            self.write(" ", '_Yield')
            self.dispatch(t.value)
        self.write(")", '_Yield')

    def _YieldFrom(self, t):
        self.write("(", '_YieldFrom')
        self.write("yield from", '_YieldFrom')
        if t.value:
            self.write(" ", '_YieldFrom')
            self.dispatch(t.value)
        self.write(")", '_YieldFrom')

    def _Raise(self, t):
        self.fill("raise", family="_Raise")
        if six.PY3:
            if not t.exc:
                assert not t.cause
                return
            self.write(" ", '_Raise')
            self.dispatch(t.exc)
            if t.cause:
                self.write(" from ", '_Raise')
                self.dispatch(t.cause)
        else:
            self.write(" ", '_Raise')
            if t.type:
                self.dispatch(t.type)
            if t.inst:
                self.write(", ", '_Raise')
                self.dispatch(t.inst)
            if t.tback:
                self.write(", ", '_Raise')
                self.dispatch(t.tback)

    def _Try(self, t):
        self.fill("try", family="_Try")
        self.enter(family="_Try")
        self.dispatch(t.body)
        self.leave()
        for ex in t.handlers:
            self.dispatch(ex)
        if t.orelse:
            self.fill("else", family="_Try")
            self.enter(family="_Try")
            self.dispatch(t.orelse)
            self.leave()
        if t.finalbody:
            self.fill("finally", family="_Try")
            self.enter(family="_Try")
            self.dispatch(t.finalbody)
            self.leave()

    def _TryExcept(self, t):
        self.fill("try", family="_TryExcept")
        self.enter(family="_TryExcept")
        self.dispatch(t.body)
        self.leave()

        for ex in t.handlers:
            self.dispatch(ex)
        if t.orelse:
            self.fill("else", family="_TryExcept")
            self.enter(family="_TryExcept")
            self.dispatch(t.orelse)
            self.leave()

    def _TryFinally(self, t):
        if len(t.body) == 1 and isinstance(t.body[0], ast.TryExcept):
            # try-except-finally
            self.dispatch(t.body)
        else:
            self.fill("try", family="_TryFinally")
            self.enter(family="_TryFinally")
            self.dispatch(t.body)
            self.leave()

        self.fill("finally", family="_TryFinally")
        self.enter(family="_TryFinally")
        self.dispatch(t.finalbody)
        self.leave()

    def _ExceptHandler(self, t):
        self.fill("except", family="_ExceptHandler")
        if t.type:
            self.write(" ", '_ExceptHandler')
            self.dispatch(t.type)
        if t.name:
            self.write(" as ", '_ExceptHandler')
            if six.PY3:
                self.write(t.name, '_ExceptHandler')
            else:
                self.dispatch(t.name)
        self.enter(family="_ExceptHandler")
        self.dispatch(t.body)
        self.leave()

    def _ClassDef(self, t):
        self.write("\n", '_ClassDef')
        for deco in t.decorator_list:
            self.fill("@", family="_ClassDef")
            self.dispatch(deco)
        self.fill("class " + t.name, family="_ClassDef")
        if six.PY3:
            self.write("(", '_ClassDef')
            comma = False
            for e in t.bases:
                if comma:
                    self.write(", ", '_ClassDef')
                else:
                    comma = True
                self.dispatch(e)
            for e in t.keywords:
                if comma:
                    self.write(", ", '_ClassDef')
                else:
                    comma = True
                self.dispatch(e)
            if sys.version_info[:2] < (3, 5):
                if t.starargs:
                    if comma:
                        self.write(", ", '_ClassDef')
                    else:
                        comma = True
                    self.write("*", '_ClassDef')
                    self.dispatch(t.starargs)
                if t.kwargs:
                    if comma:
                        self.write(", ", '_ClassDef')
                    else:
                        comma = True
                    self.write("**", '_ClassDef')
                    self.dispatch(t.kwargs)
            self.write(")", '_ClassDef')
        elif t.bases:
            self.write("(", '_ClassDef')
            for a in t.bases:
                self.dispatch(a)
                self.write(", ", '_ClassDef')
            self.write(")", '_ClassDef')
        self.enter(family="_ClassDef")
        self.dispatch(t.body)
        self.leave()

    def _FunctionDef(self, t):
        self.__FunctionDef_helper(t, "def")

    def _AsyncFunctionDef(self, t):
        self.__FunctionDef_helper(t, "async def")

    def __FunctionDef_helper(self, t, fill_suffix):
        self.write("\n", '_FunctionDef')
        for deco in t.decorator_list:
            self.fill("@", family="_FunctionDef")
            self.dispatch(deco)
        def_str = fill_suffix + " " + t.name + "("
        self.fill(def_str, family="_FunctionDef")
        self.dispatch(t.args)
        self.write(")", '_FunctionDef')
        if getattr(t, "returns", False):
            self.write(" -> ", '_FunctionDef')
            self.dispatch(t.returns)
        self.enter()
        self.dispatch(t.body)
        self.leave()

    def _For(self, t):
        self.__For_helper("for ", t)

    def _AsyncFor(self, t):
        self.__For_helper("async for ", t)

    def __For_helper(self, fill, t):
        self.fill(fill, family="_For")
        self.dispatch(t.target)
        self.write(" in ", '_For')
        self.dispatch(t.iter)
        self.enter(family="_For")
        self.dispatch(t.body)
        self.leave()
        if t.orelse:
            self.fill("else", family="_For")
            self.enter(family="_For")
            self.dispatch(t.orelse)
            self.leave()

    def _If(self, t):
        self.fill("if ", family="_If")
        self.dispatch(t.test)
        self.enter(family="_If")
        self.dispatch(t.body)
        self.leave()
        # collapse nested ifs into equivalent elifs.
        while (t.orelse and len(t.orelse) == 1 and
               isinstance(t.orelse[0], ast.If)):
            t = t.orelse[0]
            self.fill("elif ", family="_If")
            self.dispatch(t.test)
            self.enter(family="_If")
            self.dispatch(t.body)
            self.leave()
        # final else
        if t.orelse:
            self.fill("else", family="_If")
            self.enter(family="_If")
            self.dispatch(t.orelse)
            self.leave()

    def _While(self, t):
        self.fill("while ", family="_While")
        self.dispatch(t.test)
        self.enter(family="_While")
        self.dispatch(t.body)
        self.leave()
        if t.orelse:
            self.fill("else", family="_While")
            self.enter(family="_While")
            self.dispatch(t.orelse)
            self.leave()

    def _generic_With(self, t, async_=False):
        self.fill("async with " if async_ else "with ", family="_With")
        if hasattr(t, 'items'):
            interleave(lambda: self.write(", ", '_With'), self.dispatch, t.items)
        else:
            self.dispatch(t.context_expr)
            if t.optional_vars:
                self.write(" as ", '_With')
                self.dispatch(t.optional_vars)
        self.enter(family="_With")
        self.dispatch(t.body)
        self.leave()

    def _With(self, t):
        self._generic_With(t)

    def _AsyncWith(self, t):
        self._generic_With(t, async_=True)

    # expr
    def _Bytes(self, t):
        self.write(repr(t.s), '_Bytes')

    def _Str(self, tree):
        if six.PY3:
            self.write(repr(tree.s), '_Str')
        else:
            # if from __future__ import unicode_literals is in effect,
            # then we want to output string literals using a 'b' prefix
            # and unicode literals with no prefix.
            if "unicode_literals" not in self.future_imports:
                self.write(repr(tree.s), '_Str')
            elif isinstance(tree.s, str):
                self.write("b" + repr(tree.s), '_Str')
            elif isinstance(tree.s, unicode):
                self.write(repr(tree.s).lstrip("u"), '_Str')
            else:
                assert False, "shouldn't get here"

    def _JoinedStr(self, t):
        # JoinedStr(expr* values)
        self.write("f", '_JoinedStr')
        string = StringIO()
        self._fstring_JoinedStr(t, string.write)
        # Deviation from `unparse.py`: Try to find an unused quote.
        # This change is made to handle _very_ complex f-strings.
        v = string.getvalue()
        if '\n' in v or '\r' in v:
            quote_types = ["'''", '"""']
        else:
            quote_types = ["'", '"', '"""', "'''"]
        for quote_type in quote_types:
            if quote_type not in v:
                v = "{quote_type}{v}{quote_type}".format(quote_type=quote_type, v=v)
                break
        else:
            v = repr(v)
        self.write(v, '_JoinedStr')

    def _FormattedValue(self, t):
        # FormattedValue(expr value, int? conversion, expr? format_spec)
        self.write("f", '_FormattedValue')
        string = StringIO()
        self._fstring_JoinedStr(t, string.write)
        self.write(repr(string.getvalue()), '_FormattedValue')

    def _fstring_JoinedStr(self, t, write):
        for value in t.values:
            meth = getattr(self, "_fstring_" + type(value).__name__)
            meth(value, write)

    def _fstring_Str(self, t, write):
        value = t.s.replace("{", "{{").replace("}", "}}")
        write(value)

    def _fstring_Constant(self, t, write):
        assert isinstance(t.value, str)
        value = t.value.replace("{", "{{").replace("}", "}}")
        write(value)

    def _fstring_FormattedValue(self, t, write):
        write("{")
        expr = StringIO()
        CustomUnparser(t.value, expr)
        expr = expr.getvalue().rstrip("\n")
        if expr.startswith("{"):
            write(" ")  # Separate pair of opening brackets as "{ {"
        write(expr)
        if t.conversion != -1:
            conversion = chr(t.conversion)
            assert conversion in "sra"
            write("!{conversion}".format(conversion=conversion))
        if t.format_spec:
            write(":")
            meth = getattr(self, "_fstring_" + type(t.format_spec).__name__)
            meth(t.format_spec, write)
        write("}")

    def _Name(self, t):
        self.write(t.id, '_Name')

    def _NameConstant(self, t):
        self.write(repr(t.value), '_NameConstant')

    def _Repr(self, t):
        self.write("`", '_Repr')
        self.dispatch(t.value)
        self.write("`", '_Repr')

    def _write_constant(self, value, family="_Constant"):
        if isinstance(value, (float, complex)):
            # Substitute overflowing decimal literal for AST infinities.
            self.write(repr(value).replace("inf", INFSTR), family)
        else:
            self.write(repr(value), family)

    def _Constant(self, t):
        value = t.value
        if isinstance(value, tuple):
            self.write("(", '_Constant')
            if len(value) == 1:
                self._write_constant(value[0], family="_Constant")
                self.write(",", '_Constant')
            else:
                interleave(lambda: self.write(", ", '_Constant'), self._write_constant, value)
            self.write(")", '_Constant')
        elif value is Ellipsis:  # instead of `...` for Py2 compatibility
            self.write("...", '_Constant')
        else:
            if t.kind == "u":
                self.write("u", '_Constant')
            self._write_constant(t.value)

    def _Num(self, t):
        repr_n = repr(t.n)
        if six.PY3:
            self.write(repr_n.replace("inf", INFSTR), '_Num')
        else:
            # Parenthesize negative numbers, to avoid turning (-1)**2 into -1**2.
            if repr_n.startswith("-"):
                self.write("(", '_Num')
            if "inf" in repr_n and repr_n.endswith("*j"):
                repr_n = repr_n.replace("*j", "j")
            # Substitute overflowing decimal literal for AST infinities.
            self.write(repr_n.replace("inf", INFSTR), '_Num')
            if repr_n.startswith("-"):
                self.write(")", '_Num')

    def _List(self, t):
        self.write("[", '_List')
        interleave(lambda: self.write(", ", '_List'), self.dispatch, t.elts)
        self.write("]", '_List')

    def _ListComp(self, t):
        self.write("[", '_ListComp')
        self.dispatch(t.elt)
        for gen in t.generators:
            self.dispatch(gen)
        self.write("]", '_ListComp')

    def _GeneratorExp(self, t):
        self.write("(", '_GeneratorExp')
        self.dispatch(t.elt)
        for gen in t.generators:
            self.dispatch(gen)
        self.write(")", '_GeneratorExp')

    def _SetComp(self, t):
        self.write("{", '_SetComp')
        self.dispatch(t.elt)
        for gen in t.generators:
            self.dispatch(gen)
        self.write("}", '_SetComp')

    def _DictComp(self, t):
        self.write("{", '_DictComp')
        self.dispatch(t.key)
        self.write(": ", '_DictComp')
        self.dispatch(t.value)
        for gen in t.generators:
            self.dispatch(gen)
        self.write("}", '_DictComp')

    def _comprehension(self, t):
        if getattr(t, 'is_async', False):
            self.write(" async for ", '_comprehension')
        else:
            self.write(" for ", '_comprehension')
        self.dispatch(t.target)
        self.write(" in ", '_comprehension')
        self.dispatch(t.iter)
        for if_clause in t.ifs:
            self.write(" if ", '_comprehension')
            self.dispatch(if_clause)

    def _IfExp(self, t):
        self.write("(", '_IfExp')
        self.dispatch(t.body)
        self.write(" if ", '_IfExp')
        self.dispatch(t.test)
        self.write(" else ", '_IfExp')
        self.dispatch(t.orelse)
        self.write(")", '_IfExp')

    def _Set(self, t):
        assert (t.elts)  # should be at least one element
        self.write("{", '_Set')
        interleave(lambda: self.write(", ", '_Set'), self.dispatch, t.elts)
        self.write("}", '_Set')

    def _Dict(self, t):
        self.write("{", '_Dict')

        def write_key_value_pair(k, v):
            self.dispatch(k)
            self.write(": ", '_Dict')
            self.dispatch(v)

        def write_item(item):
            k, v = item
            if k is None:
                # for dictionary unpacking operator in dicts {**{'y': 2}}
                # see PEP 448 for details
                self.write("**", '_Dict')
                self.dispatch(v)
            else:
                write_key_value_pair(k, v)

        interleave(lambda: self.write(", ", '_Dict'), write_item, zip(t.keys, t.values))
        self.write("}", '_Dict')

    def _Tuple(self, t):
        self.write("(", '_Tuple')
        if len(t.elts) == 1:
            elt = t.elts[0]
            self.dispatch(elt)
            self.write(",", '_Tuple')
        else:
            interleave(lambda: self.write(", ", '_Tuple'), self.dispatch, t.elts)
        self.write(")", '_Tuple')

    unop = {"Invert": "~", "Not": "not", "UAdd": "+", "USub": "-"}

    def _UnaryOp(self, t):
        self.write("(", '_UnaryOp')
        self.write(self.unop[t.op.__class__.__name__], '_UnaryOp')
        self.write(" ", '_UnaryOp')
        if six.PY2 and isinstance(t.op, ast.USub) and isinstance(t.operand, ast.Num):
            # If we're applying unary minus to a number, parenthesize the number.
            # This is necessary: -2147483648 is different from -(2147483648) on
            # a 32-bit machine (the first is an int, the second a long), and
            # -7j is different from -(7j).  (The first has real part 0.0, the second
            # has real part -0.0.)
            self.write("(", '_UnaryOp')
            self.dispatch(t.operand)
            self.write(")", '_UnaryOp')
        else:
            self.dispatch(t.operand)
        self.write(")", '_UnaryOp')

    binop = {"Add": "+", "Sub": "-", "Mult": "*", "MatMult": "@", "Div": "/", "Mod": "%",
             "LShift": "<<", "RShift": ">>", "BitOr": "|", "BitXor": "^", "BitAnd": "&",
             "FloorDiv": "//", "Pow": "**"}

    def _BinOp(self, t):
        self.write("(", '_BinOp')
        self.dispatch(t.left)
        self.write(" " + self.binop[t.op.__class__.__name__] + " ", '_BinOp')
        self.dispatch(t.right)
        self.write(")", '_BinOp')

    cmpops = {"Eq": "==", "NotEq": "!=", "Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">=",
              "Is": "is", "IsNot": "is not", "In": "in", "NotIn": "not in"}

    def _Compare(self, t):
        self.write("(", '_Compare')
        self.dispatch(t.left)
        for o, e in zip(t.ops, t.comparators):
            self.write(" " + self.cmpops[o.__class__.__name__] + " ", '_Compare')
            self.dispatch(e)
        self.write(")", '_Compare')

    boolops = {ast.And: 'and', ast.Or: 'or'}

    def _BoolOp(self, t):
        self.write("(", '_BoolOp')
        s = " %s " % self.boolops[t.op.__class__]
        interleave(lambda: self.write(s, '_BoolOp'), self.dispatch, t.values)
        self.write(")", '_BoolOp')

    def _Attribute(self, t):
        self.dispatch(t.value)
        # Special case: 3.__abs__() is a syntax error, so if t.value
        # is an integer literal then we need to either parenthesize
        # it or add an extra space to get 3 .__abs__().
        if isinstance(t.value, getattr(ast, 'Constant', getattr(ast, 'Num', None))) and isinstance(t.value.n, int):
            self.write(" ", '_Attribute')
        self.write(".", '_Attribute')
        self.write(t.attr, '_Attribute')

    def _Call(self, t):
        self.dispatch(t.func)
        self.write("(", '_Call')
        comma = False
        for e in t.args:
            if comma:
                self.write(", ", '_Call')
            else:
                comma = True
            self.dispatch(e)
        for e in t.keywords:
            if comma:
                self.write(", ", '_Call')
            else:
                comma = True
            self.dispatch(e)
        if sys.version_info[:2] < (3, 5):
            if t.starargs:
                if comma:
                    self.write(", ", '_Call')
                else:
                    comma = True
                self.write("*", '_Call')
                self.dispatch(t.starargs)
            if t.kwargs:
                if comma:
                    self.write(", ", '_Call')
                else:
                    comma = True
                self.write("**", '_Call')
                self.dispatch(t.kwargs)
        self.write(")", '_Call')

    def _Subscript(self, t):
        self.dispatch(t.value)
        self.write("[", '_Subscript')
        self.dispatch(t.slice)
        self.write("]", '_Subscript')

    def _Starred(self, t):
        self.write("*", '_Starred')
        self.dispatch(t.value)

    # slice
    def _Ellipsis(self, t):
        self.write("...", '_Ellipsis')

    def _Index(self, t):
        self.dispatch(t.value)

    def _Slice(self, t):
        if t.lower:
            self.dispatch(t.lower)
        self.write(":", '_Slice')
        if t.upper:
            self.dispatch(t.upper)
        if t.step:
            self.write(":", '_Slice')
            self.dispatch(t.step)

    def _ExtSlice(self, t):
        interleave(lambda: self.write(', ', '_ExtSlice'), self.dispatch, t.dims)

    # argument
    def _arg(self, t):
        self.write(t.arg, '_arg')
        if t.annotation:
            self.write(": ", '_arg')
            self.dispatch(t.annotation)

    # others
    def _arguments(self, t):
        first = True
        # normal arguments
        all_args = getattr(t, 'posonlyargs', []) + t.args
        defaults = [None] * (len(all_args) - len(t.defaults)) + t.defaults
        for index, elements in enumerate(zip(all_args, defaults), 1):
            a, d = elements
            if first:
                first = False
            else:
                self.write(", ", '_arguments')
            self.dispatch(a)
            if d:
                self.write("=", '_arguments')
                self.dispatch(d)
            if index == len(getattr(t, 'posonlyargs', ())):
                self.write(", /", '_arguments')

        # varargs, or bare '*' if no varargs but keyword-only arguments present
        if t.vararg or getattr(t, "kwonlyargs", False):
            if first:
                first = False
            else:
                self.write(", ", '_arguments')
            self.write("*", '_arguments')
            if t.vararg:
                if hasattr(t.vararg, 'arg'):
                    self.write(t.vararg.arg, '_arguments')
                    if t.vararg.annotation:
                        self.write(": ", '_arguments')
                        self.dispatch(t.vararg.annotation)
                else:
                    self.write(t.vararg, '_arguments')
                    if getattr(t, 'varargannotation', None):
                        self.write(": ", '_arguments')
                        self.dispatch(t.varargannotation)

        # keyword-only arguments
        if getattr(t, "kwonlyargs", False):
            for a, d in zip(t.kwonlyargs, t.kw_defaults):
                if first:
                    first = False
                else:
                    self.write(", ", '_arguments')
                self.dispatch(a),
                if d:
                    self.write("=", '_arguments')
                    self.dispatch(d)

        # kwargs
        if t.kwarg:
            if first:
                first = False
            else:
                self.write(", ", '_arguments')
            if hasattr(t.kwarg, 'arg'):
                self.write("**" + t.kwarg.arg, '_arguments')
                if t.kwarg.annotation:
                    self.write(": ", '_arguments')
                    self.dispatch(t.kwarg.annotation)
            else:
                self.write("**" + t.kwarg, '_arguments')
                if getattr(t, 'kwargannotation', None):
                    self.write(": ", '_arguments')
                    self.dispatch(t.kwargannotation)

    def _keyword(self, t):
        if t.arg is None:
            # starting from Python 3.5 this denotes a kwargs part of the invocation
            self.write("**", '_keyword')
        else:
            self.write(t.arg, '_keyword')
            self.write("=", '_keyword')
        self.dispatch(t.value)

    def _Lambda(self, t):
        self.write("(", '_Lambda')
        self.write("lambda ", '_Lambda')
        self.dispatch(t.args)
        self.write(": ", '_Lambda')
        self.dispatch(t.body)
        self.write(")", '_Lambda')

    def _alias(self, t):
        self.write(t.name, '_alias')
        if t.asname:
            self.write(" as " + t.asname, '_alias')

    def _withitem(self, t):
        self.dispatch(t.context_expr)
        if t.optional_vars:
            self.write(" as ", '_withitem')
            self.dispatch(t.optional_vars)


def roundtrip(filename, output=sys.stdout):
    if six.PY3:
        with open(filename, "rb") as pyfile:
            encoding = tokenize.detect_encoding(pyfile.readline)[0]
        with open(filename, "r", encoding=encoding) as pyfile:
            source = pyfile.read()
    else:
        with open(filename, "r") as pyfile:
            source = pyfile.read()
    tree = compile(source, filename, "exec", ast.PyCF_ONLY_AST, dont_inherit=True)
    CustomUnparser(tree, output)


class UnparserTokenizer:
    def find_tokens(self, tree):
        v = cStringIO()
        tokens = []
        CustomUnparser(tree, tokens, file=v)
        return tokens
