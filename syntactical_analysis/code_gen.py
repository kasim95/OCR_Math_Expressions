import lxml.etree as ET
from syntactical_analysis.sa_utils import *
from html import unescape, escape

class CodeGenerator:
    def __init__(self):
        self.tree = None
        # chart to convert latex to entity reference encoding used by xml
        self.latex2entityref = {
            r'+': '&plus;',
            r'-': '&minus;',
            r',': '&comma;',
            r'\times': r'&times;',
            r'\ast': '&times;',
            r'\div': '&div;',
            r'\%': '&percnt;',
            r'\comma': '&comma;',
            r'.': '&period;',
            r'\cdot': '&period;',
            r'\lt': '&lt;',
            r'\gt': '&gt;',
            r'=': '&equals;',
            r'\(': '&lpar;',
            r'(': '&lpar;',
            r'\)': '&rpar;',
            r')': '&rpar;',
            r'\\': '&bsol;',
            r'/': '&sol;',
            r'\{': '&lcub;',
            r'\}': '&rcub;',
            r'\amp': '&amp;',
            r'\forall': '&forall;',
            r'\exists': '&exist;',
            r'\cup': '&cup;',
            r'\cap': '&cap;',
        }

    def gen_mathml(self, tree_):
        math = ET.Element('math', xmlns="http://www.w3.org/1998/Math/MathML")
        self.tree = ET.ElementTree(math)
        mrow = ET.SubElement(math, 'mrow')
        elems = []
        # skip $
        for i in tree_[:-1]:
            obj = i[0]
            if obj.name == 'IDENTIFIER':
                tag = ET.SubElement(mrow, 'mi')
                tag.text = obj.lexeme
            elif obj.name in ["INTEGER", "NUMBER", "REAL"]:
                tag = ET.SubElement(mrow, 'mn')
                tag.text = obj.lexeme
            elif obj.name in ["OPERATOR", 'SEPARATOR']:
                tag = ET.SubElement(mrow, 'mo')
                if obj.lexeme in list(self.latex2entityref.keys()):
                    tag.text = unescape(self.latex2entityref[obj.lexeme])
                else:
                    tag.text = obj.lexeme
            else:
                tag = ET.SubElement(mrow, 'mtext')
                if obj.lexeme in list(self.latex2entityref.keys()):
                    tag.text = self.latex2entityref[obj.lexeme]
                else:
                    tag.text = obj.lexeme
        return self.tree


def main():
    # import inside main() fn to prevent any overwrite in the global variables
    from syntactical_analysis.lexer import Lexer
    from syntactical_analysis.parser import Parser
    expression = ['z', '=', 'x', r'\%', 'y']
    # lexer
    lexer = Lexer()
    tokens = lexer.generate_tokens(expression)
    # parser
    parser = Parser()
    tree = parser.generate_tree(tokens)
    # code generator
    codegen = CodeGenerator()
    xml_code = codegen.gen_mathml(tree)
    # print(ET.tostring(xml_code, pretty_print=True).decode('utf-8'))
    xml_code.write("output.xml", encoding='utf-8')


if __name__ == '__main__':
    main()
