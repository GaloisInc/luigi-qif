import json
import sys

_ARGS_TO_VALUES = {}
_VALUES_TO_ARGS = {}
_REVEAL = []
_NUM_INPUTS = 0


def _print_output():
    print(json.dumps({
        'REVEAL': _REVEAL,
        'MAPPING': _VALUES_TO_ARGS,
    }))


def _convert(x):
    assert isinstance(x, int) or isinstance(x, sint), repr((x, type(x)))
    if isinstance(x, int):
        return sint(x)
    else:
        return x


def _add(operator, *args):
    args = tuple([operator] + [_convert(x).of for x in args])
    if args in _ARGS_TO_VALUES:
        out = _ARGS_TO_VALUES[args]
        assert isinstance(out, int) or isinstance(out, sint)
        return out
    id = 'EXPR_%d' % len(_VALUES_TO_ARGS)
    _ARGS_TO_VALUES[args] = sint(id)
    _VALUES_TO_ARGS[id] = args
    return sint(id)


def _input():
    global _NUM_INPUTS
    id = 'INPUT_%d' % _NUM_INPUTS
    _VALUES_TO_ARGS[id] = ['INPUT', _NUM_INPUTS]
    _NUM_INPUTS += 1
    return sint(id)


class sint(object):
    def __init__(self, of):
        if isinstance(of, sint):
            self.of = of.of
        else:
            self.of = of

    def __repr__(self):
        return str(self)

    def __str__(self):
        return 'sint(%r)' % self.of

    def __add__(self, other):
        return _add('+', self, other)

    def __radd__(self, other):
        return _add('+', other, self)

    def __mul__(self, other):
        return _add('*', self, other)

    def __rmul__(self, other):
        return _add('*', other, self)

    def __sub__(self, other):
        return _add('-', self, other)

    def __rsub__(self, other):
        return _add('-', other, self)

    def __neg__(self):
        return 0 - self

    @classmethod
    def get_raw_input_from(cls, party):
        return _input()

    @classmethod
    def get_input_from(cls, party):
        return _input()

    @classmethod
    def get_private_input_from(cls, party):
        return _input()

    def if_else(self, if_true, if_false):
        return _add(
            'ITE',
            self,
            if_true,
            if_false,
        )

    def __lt__(self, other):
        return _add(
            '<',
            self,
            other,
        )

    def __le__(self, other):
        return _add(
            '<=',
            self,
            other,
        )

    def __eq__(self, other):
        return _add(
            '==',
            self,
            other,
        )

    def __ne__(self, other):
        return _add(
            '!=',
            self,
            other,
        )

    def __gt__(self, other):
        return _add(
            '>',
            self,
            other,
        )

    def __ge__(self, other):
        return _add(
            '>=',
            self,
            other,
        )

    def reveal(self):
        _REVEAL.append(self.of)
        return self


cint = sint


def print_ln(*args):
    pass
