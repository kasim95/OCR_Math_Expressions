import re
import numpy as np
from enum import Enum
from syntactical_analysis.sa_utils import State, Input, Token

__all__ = [
    'Lexer',
]


class Lexer:
    def __init__(self):
        self.separators = ['(', ')', '[', ']', r'\{', r'\}', '.', ',', ':', ';', ' ', r'\cdot']
        self.operators = ['+', '-', '=', '/', '>', '<', '%', r'\%',
                          r'\&', r'\times', r'\div', r'\ast', r'\cup', r'\cap'
                          ]
        self.state_table_data = [
            [1, 3, 11, 8, 9, 10],
            [1, 2, 11, 2, 2, 2],
            [0, 0, 0, 0, 0, 0],
            [4, 3, 5, 4, 4, 4],
            [0, 0, 0, 0, 0, 0],
            [11, 6, 11, 11, 11, 11],
            [7, 6, 7, 7, 7, 7],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]
        self.state_table = [[None for _ in range(len(self.state_table_data[0]))]
                            for _ in range(len(self.state_table_data))]
        for i in range(len(self.state_table_data)):
            for j in range(len(self.state_table_data[0])):
                self.state_table[i][j] = State(self.state_table_data[i][j], None)
        self.keywords = ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'and', 'or']
        self.greeks = [r'\sigma', r'\Sigma', r'\gamma', r'\delta', r'\Delta',
                       r'\eta', r'\theta', r'\epsilon', r'\lambda', r'\mu',
                       r'\Pi', r'\rho', r'\phi', r'\omega', r'\ohm']
        self.special_chars = [r'\infty', r'\exists', r'\forall', r'\#', r'\$'] + self.greeks

    @staticmethod
    def get_characters(chars):
        result = []
        for c in chars:
            if c not in [r'\n', r'\t', r'\r']:
                result.append(c)
        return result

    @staticmethod
    def change_ip_order(exp):
        if "=" in exp:
            ix_eq = exp.index('=')
            if ix_eq > 1:
                result = exp[ix_eq + 1:] + exp[ix_eq:ix_eq + 1] + exp[:ix_eq]
                return result
            else:
                return exp
        else:
            result = ['Z', '='] + exp
            return result

    def checkInput(self, chars):
        if re.match("[a-zA-Z]", chars):
            return Input.LETTER
        elif re.match("[0-9]", chars):
            return Input.DIGIT
        elif chars in ['$']:
            return Input.DOLLAR
        elif chars in ['.', r'\cdot']:
            return Input.DOT
        elif chars in self.separators:
            return Input.SEP
        elif chars in self.operators:
            return Input.OP
        elif chars in self.special_chars:
            return Input.SPECIAL
        else:
            raise ValueError("INVALID INPUT: "+str(chars))

    def generate_tokens(self, expression):
        # changes inputs to format i= expression
        expression = self.change_ip_order(expression)
        # added this because ints, letters and real require a separator at the end to move to final state
        expression = expression + [' ']
        prev_state = State.INITIAL
        lexeme = ''
        tokens = []
        i = 0
        while i < len(expression):
            backup = False
            final = False
            alpha = expression[i]
            x = prev_state.getvalue
            y = self.checkInput(alpha).getvalue
            current_state = self.state_table[x][y]
            if current_state.getvalue == 2:
                if lexeme in self.keywords:
                    tokens.append(Token(value=1, lexeme=lexeme))
                else:
                    tokens.append(Token(value=0, lexeme=lexeme))
                final = True
                backup = True
            elif current_state.getvalue == 4:
                tokens.append(Token(value=2, lexeme=lexeme))
                final = True
                backup = True
            elif current_state.getvalue == 7:
                tokens.append(Token(value=3, lexeme=lexeme))
                final = True
                backup = True
            elif current_state.getvalue == 8:
                tokens.append(Token(value=4, lexeme=alpha))
                final = True
                backup = False
            elif current_state.getvalue == 9:
                tokens.append(Token(value=5, lexeme=alpha))
                final = True
                backup = False
            elif current_state.getvalue == 10:
                tokens.append(Token(value=6, lexeme=alpha))
                final = True
                backup = False
            elif current_state.getvalue == 11:
                raise ValueError("Reached Dead State")
            if final:
                lexeme = ''
                prev_state = State.INITIAL
            else:
                lexeme += alpha
                prev_state = current_state
            if not backup:
                i += 1

        # ignore the last separator added at the end of expression at the start of this function
        return tokens[:-1]


def main():
    expression = ['z', '=', 'x', '+', 'y']
    lexer = Lexer()
    tokens = lexer.generate_tokens(expression)
    print("Generated tokens are ")
    for i in tokens:
        print(i.name, i.lexeme)


if __name__ == '__main__':
    main()
