import numpy as np
from enum import Enum


__all__ = [
    'State',
    'Input',
    'Token',
    'Terminals',
    'NonTerminals',
    'ProdRules',
]


class State(Enum):
    INITIAL = 0, 'INITIAL'
    IDENTIFIER_START = 1, 'IDENTIFIER_START'
    IDENTIFIER_END = 2, 'IDENTIFIER_END'
    INTEGER_START = 3, 'INTEGER_START'
    INTEGER_END = 4, 'INTEGER_END'
    REAL_START = 5, 'REAL_START'
    REAL_IN = 6, 'REAL_IN'
    REAL_END = 7, 'REAL_END'
    SEPARATOR = 8, 'SEPARATOR'
    OPERATOR = 9, 'OPERATOR'
    SPECIAL_CHARACTER = 10, 'SPECIAL_CHARACTER'
    DEAD = 11, 'DEAD_STATE'

    def __new__(cls, value, name):
        x = object.__new__(cls)
        x._value_ = value
        x._name = name
        return x

    @property
    def getvalue(self):
        return self.value


class Input(Enum):
    LETTER = 0, 'LETTER'
    DIGIT = 1, 'DIGIT'
    DOT = 2, 'DOT'
    SEP = 3, 'SEP'
    OP = 4, 'OP'
    SPECIAL = 5, 'SPECIAL'

    def __new__(cls, value, name):
        x = object.__new__(cls)
        x._value_ = value
        x._name = name
        return x

    @property
    def getvalue(self):
        return self.value


class Token:
    def __init__(self, **kwargs):
        self._value = kwargs['value']
        if self._value == 0:
            self._name = 'IDENTIFIER'
        elif self._value == 1:
            self._name = 'KEYWORD'
        elif self._value == 2:
            self._name = 'INTEGER'
        elif self._value == 3:
            self._name = 'REAL'
        elif self._value == 4:
            self._name = 'SEPARATOR'
        elif self._value == 5:
            self._name = 'OPERATOR'
        elif self._value == 6:
            self._name = 'SPECIAL'
        else:
            raise ValueError('value should be within 0 and 5')
        self._lexeme = kwargs['lexeme']

    @property
    def getvalue(self):
        return self._value

    @property
    def lexeme(self):
        return self._lexeme

    @property
    def name(self):
        return self._name


class NonTerminals(Enum):
    S = 0, 'S'
    V = 1, 'V'
    E = 2, 'E'
    Q = 3, 'Q'
    T = 4, 'T'
    R = 5, 'R'
    F = 6, 'F'
    D = 7, 'D'
    O = 8, 'O'
    Z = 9, 'Z'
    J = 10, 'J'
    M = 11, 'M'
    K = 12, 'K'
    NONE = 13, 'NONE'

    def __new__(cls, value, name):
        x = object.__new__(cls)
        x._value_ = value
        x._name = name
        return x

    @property
    def getvalue(self):
        return self.value

    @property
    def name(self):
        return self._name


class Terminals(Enum):
    IDENTIFIER = 0, 'IDENTIFIER'
    NUMBER = 1, 'NUMBER'
    REAL = 2, 'REAL'
    GREEK = 3, 'GREEK'
    EQUAL = 4, 'EQUAL'
    PLUS = 5, 'PLUS'
    MINUS = 6, 'MINUS'
    MULTIPLY = 7, 'MULTIPLY'
    DIVISION = 8, 'DIVISION'
    MOD = 9, 'MOD'
    LEFT_ROUNDB = 10, 'LEFT_ROUNDB'
    RIGHT_ROUNDB = 11, 'RIGHT_ROUNDB'
    LEFT_CURLYB = 12, 'LEFT_CURLYB'
    RIGHT_CURLYB = 13, 'RIGHT_CURLYB'
    UNION = 14, 'UNION'
    INTERSECTION = 15, 'INTERSECTION'
    COMMA = 16, 'COMMA'
    DOLLAR = 17, "DOLLAR"
    NONE = 18, 'NONE'

    def __new__(cls, value, name):
        x = object.__new__(cls)
        x._value_ = value
        x._name = name
        return x

    @property
    def getvalue(self):
        return self.value

    @property
    def name(self):
        return self._name


class ProdRules(Enum):
    IEV = 0, "i = V"
    E = 1, "E"
    D = 2, "D"
    TQ = 3, "T Q"
    ATQ = 4, "+ T Q"
    STQ = 5, "- T Q"
    FR = 6, "F R"
    MFR = 7, "* F R"
    DFR = 8, "/ F R"
    PFR = 9, "% F R"
    BEB = 10, "( E )"
    I = 11, "i"
    N = 12, "n"
    Re = 13, 'r'
    G = 14, 'g'
    ZO = 15, 'Z O'
    UZO = 16, 'Union ZO'
    InterZO = 17, 'Intersection ZO'
    CoJCc = 18, '{J}'
    Z = 19, 'Z'
    M = 20, 'M'
    IK = 21, 'iK'
    NK = 22, 'nK'
    ReK = 23, 'rK'
    GK = 24, 'gK'
    CommaJ = 25, ',J'
    EPS = 26, "EPSILON"
    INVALID = 27, "INVALID"

    def __new__(cls, value, name):
        x = object.__new__(cls)
        x._value_ = value
        x._name = name
        return x

    @property
    def getvalue(self):
        return self.value
