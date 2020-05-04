import numpy as np
import pandas as pd
from enum import Enum
from syntactical_analysis.sa_utils import *


__all__ = [
    'Parser',
]


class Parser:
    def __init__(self):
        self.predictive_parser_table_data = [
            [0, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27],
            [1, 1, 1, 1, 27, 27, 27, 27, 27, 27, 1, 27, 2, 27, 27, 27, 27, 27],
            [3, 3, 3, 3, 27, 27, 27, 27, 27, 27, 3, 27, 27, 27, 27, 27, 27, 27],
            [27, 27, 27, 27, 27, 4, 5, 27, 27, 27, 27, 26, 27, 27, 27, 27, 27, 26],
            [6, 6, 6, 6, 27, 27, 27, 27, 27, 27, 6, 27, 27, 27, 27, 27, 27, 27],
            [27, 27, 27, 27, 27, 26, 26, 7, 8, 9, 27, 26, 27, 27, 27, 27, 27, 26],
            [11, 12, 13, 14, 27, 27, 27, 27, 27, 27, 10, 27, 27, 27, 27, 27, 27, 27],
            [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 15, 27, 27, 27, 27, 27],
            [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 16, 17, 27, 26],
            [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 18, 27, 27, 27, 27, 27],
            [20, 20, 20, 20, 27, 27, 27, 27, 27, 27, 27, 27, 19, 27, 27, 27, 27, 27],
            [21, 22, 23, 24, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27],
            [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 26, 27, 27, 25, 26],
        ]
        self.predictive_parser_table = [[None for _ in range(len(self.predictive_parser_table_data[0]))]
                                        for _ in range(len(self.predictive_parser_table_data))]
        for i in range(len(self.predictive_parser_table_data)):
            for j in range(len(self.predictive_parser_table_data[0])):
                self.predictive_parser_table[i][j] = ProdRules(self.predictive_parser_table_data[i][j], None)
        self.predictive_parser_table = np.array(self.predictive_parser_table)
        self.list_terminals = [i for i in Terminals if i.name not in ["INVALID", "EPSILON"]]
        # stack keeps track of what is remaining to be checked
        self.stack = []

    @staticmethod
    def change_terminal(lexeme):
        if lexeme == "i":
            return Terminals.IDENTIFIER
        elif lexeme == "n":
            return Terminals.NUMBER
        elif lexeme == "r":
            return Terminals.REAL
        elif lexeme == "g":
            return Terminals.GREEK
        elif lexeme == "=":
            return Terminals.EQUAL
        elif lexeme == "+":
            return Terminals.PLUS
        elif lexeme == "-":
            return Terminals.MINUS
        elif lexeme in ["*", r"\times", r"\ast"]:
            return Terminals.MULTIPLY
        elif lexeme in [r"/", r"\div"]:
            return Terminals.DIVISION
        elif lexeme == "%":
            return Terminals.MOD
        elif lexeme in ["(", r"\("]:
            return Terminals.LEFT_ROUNDB
        elif lexeme in [")", r"\)"]:
            return Terminals.RIGHT_ROUNDB
        elif lexeme in ["{", r"\{"]:
            return Terminals.LEFT_CURLYB
        elif lexeme in ["}", r"\}"]:
            return Terminals.RIGHT_CURLYB
        elif lexeme in [r"\cup"]:
            return Terminals.UNION
        elif lexeme in [r"\cap"]:
            return Terminals.INTERSECTION
        elif lexeme in [',', r'\comma']:
            return Terminals.COMMA
        elif lexeme == "$":
            return Terminals.DOLLAR
        else:
            return Terminals.NONE

    @staticmethod
    def change_input(token, lexeme):
        if token == "IDENTIFIER":
            return Terminals.IDENTIFIER
        elif token == "INTEGER":
            return Terminals.NUMBER
        elif token == "REAL":
            return Terminals.REAL
        elif token == "GREEK":
            return Terminals.GREEK
        elif token == "KEYWORD":
            if lexeme in ["and", r"\&"]:
                return Terminals.AND
            elif lexeme in ["or", r"\|"]:
                return Terminals.OR
            else:
                return Terminals.NONE
        elif token == "OPERATOR":
            if lexeme == "+":
                return Terminals.PLUS
            elif lexeme == "-":
                return Terminals.MINUS
            elif lexeme in ["*", r"\times", r"\ast"]:
                return Terminals.MULTIPLY
            elif lexeme in [r"/", r"\div"]:
                return Terminals.DIVISION
            elif lexeme in ["%", r"\%"]:
                return Terminals.MOD
            elif lexeme in [r"\cup"]:
                return Terminals.UNION
            elif lexeme in [r"\cap"]:
                return Terminals.INTERSECTION
            elif lexeme == "$":
                return Terminals.DOLLAR
            elif lexeme == "=":
                return Terminals.EQUAL
            else:
                return Terminals.NONE
        elif token == "SEPARATOR":
            if lexeme in ["(", r"\("]:
                return Terminals.LEFT_ROUNDB
            elif lexeme in [")", r"\)"]:
                return Terminals.RIGHT_ROUNDB
            elif lexeme in ["{", r"\{"]:
                return Terminals.LEFT_CURLYB
            elif lexeme in ["}", r"\}"]:
                return Terminals.RIGHT_CURLYB
            elif lexeme in [",", r"\comma"]:
                return Terminals.COMMA
            elif lexeme == "$":
                return Terminals.DOLLAR
            elif lexeme == ";":
                return Terminals.DOLLAR
            else:
                return Terminals.NONE
        else:
            return Terminals.NONE

    # helper fn to push contents to stack
    def pop(self):
        temp = self.stack[-1]
        del self.stack[-1]
        return temp

    # Helper function used to push contents to the stack
    def push(self, t):
        self.stack.append(t)

    # Helper function used to push the contents to the stack according to the Production Rules
    def push_rules(self, rule):
        if rule == ProdRules(0):
            self.push(NonTerminals.V)
            self.push(Terminals.EQUAL)
            self.push(Terminals.IDENTIFIER)
        elif rule == ProdRules(1):
            self.push(NonTerminals.E)
        elif rule == ProdRules(2):
            self.push(NonTerminals.D)
        elif rule == ProdRules(3):
            self.push(NonTerminals.Q)
            self.push(NonTerminals.T)
        elif rule == ProdRules(4):
            self.push(NonTerminals.Q)
            self.push(NonTerminals.T)
            self.push(Terminals.PLUS)
        elif rule == ProdRules(5):
            self.push(NonTerminals.Q)
            self.push(NonTerminals.T)
            self.push(Terminals.MINUS)
        elif rule == ProdRules(6):
            self.push(NonTerminals.R)
            self.push(NonTerminals.F)
        elif rule == ProdRules(7):
            self.push(NonTerminals.R)
            self.push(NonTerminals.F)
            self.push(Terminals.MULTIPLY)
        elif rule == ProdRules(8):
            self.push(NonTerminals.R)
            self.push(NonTerminals.F)
            self.push(Terminals.DIVISION)
        elif rule == ProdRules(9):
            self.push(NonTerminals.R)
            self.push(NonTerminals.F)
            self.push(Terminals.MOD)
        elif rule == ProdRules(10):
            self.push(Terminals.RIGHT_ROUNDB)
            self.push(NonTerminals.E)
            self.push(Terminals.LEFT_ROUNDB)
        elif rule == ProdRules(11):
            self.push(Terminals.IDENTIFIER)
        elif rule == ProdRules(12):
            self.push(Terminals.NUMBER)
        elif rule == ProdRules(13):
            self.push(Terminals.REAL)
        elif rule == ProdRules(14):
            self.push(Terminals.GREEK)
        elif rule == ProdRules(15):
            self.push(NonTerminals.O)
            self.push(NonTerminals.Z)
        elif rule == ProdRules(16):
            self.push(NonTerminals.O)
            self.push(NonTerminals.Z)
            self.push(Terminals.UNION)
        elif rule == ProdRules(17):
            self.push(NonTerminals.O)
            self.push(NonTerminals.Z)
            self.push(Terminals.INTERSECTION)
        elif rule == ProdRules(18):
            self.push(Terminals.RIGHT_CURLYB)
            self.push(NonTerminals.J)
            self.push(Terminals.LEFT_CURLYB)
        elif rule == ProdRules(19):
            self.push(NonTerminals.Z)
        elif rule == ProdRules(20):
            self.push(NonTerminals.M)
        elif rule == ProdRules(21):
            self.push(NonTerminals.K)
            self.push(Terminals.IDENTIFIER)
        elif rule == ProdRules(22):
            self.push(NonTerminals.K)
            self.push(Terminals.NUMBER)
        elif rule == ProdRules(23):
            self.push(NonTerminals.K)
            self.push(Terminals.REAL)
        elif rule == ProdRules(24):
            self.push(NonTerminals.K)
            self.push(Terminals.GREEK)
        elif rule == ProdRules(25):
            self.push(NonTerminals.J)
            self.push(Terminals.COMMA)
        elif rule == ProdRules(26):
            # This statement does nothing, used
            # so the code does not given runtime error
            # when function is called for epsilon
            pass
        else:
            self.push(NonTerminals.NONE)

    def generate_tree(self, tokens_lexemes):
        # reset stack
        self.stack = []
        tokens_lexemes.append(Token(value=4, lexeme="$"))
        per_ip = [tokens_lexemes[0]]
        result = []
        # used to keep track of the length of the tokens_lexemes dataframe
        i = 0
        # push S to stack for the first line
        self.push(NonTerminals.S)
        # executes until the stack becomes empty and the tokens_lexemes dataframe reaches the end
        while i < len(tokens_lexemes) and len(self.stack) != 0:
            # print(stack)
            # changes input to the instance of Terminals class; will be used further as column no for parser table
            x = self.change_input(tokens_lexemes[i].name, tokens_lexemes[i].lexeme)
            # pops the first value of the stack; will be used further as row no for parser table
            y = self.pop()
            # this will raise a Syntax Error which will be caught by Error Handling
            if x == Terminals.NONE or y == NonTerminals.NONE:
                raise SyntaxError("Token: {}\nLexeme: {} \nSyntax Error: Invalid Syntax"
                                  .format(tokens_lexemes[i].name,tokens_lexemes[i].lexeme))
            elif y in self.list_terminals:  # executes if y and x are both equal
                if x == y:
                    result.append(per_ip)
                    i += 1
                else:
                    raise SyntaxError("Token: {}\nLexeme: {} \nSyntax Error: Invalid Syntax"
                                      .format(tokens_lexemes[i].name,tokens_lexemes[i].lexeme))
                # used to keep record of the token and lexeme for the current iteration;
                # will be added to op_df when the input changes
                per_ip = [tokens_lexemes[i]]
            # executes when y and x are not equal
            else:
                # calculates the Production Rule according to y and x from the parser table
                new_value = self.predictive_parser_table[y.value][x.value]
                # adds the Prod Rule used to the op_ls list
                per_ip.append((y, new_value))
                # pushes the contents of the Prod Rule to the stack
                self.push_rules(new_value)
        result.append(per_ip)
        return result


def main():
    # import inside main() fn to prevent any overwrite in the global variables
    from syntactical_analysis.lexer import Lexer
    expression = ['z', '=', 'x', '+', 'y']
    # lexer
    lexer = Lexer()
    tokens = lexer.generate_tokens(expression)
    # parser
    parser = Parser()
    tree = parser.generate_tree(tokens)
    for i in tree:
        print('Lexeme: ', i[0].lexeme, '', i)


if __name__ == '__main__':
    main()
