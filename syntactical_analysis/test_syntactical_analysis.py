import lxml.etree as ET
import re
from syntactical_analysis.lexer import Lexer
from syntactical_analysis.parser import Parser
from syntactical_analysis.code_gen import CodeGenerator

exp1 = ['z', '=', '2']
exp2 = ['z', '=', 'x', '+', 'y']
exp3 = ['a', '+', 'b', '=', 'c']
exp4 = ['z', '=', '2', '+', '3']
exp5 = ['2', '1', '.', '5', '0', '+', '3', '2', '.', '4', '=', 'a']
exp6 = ['a', '=', 'b', '+', 'c', '-', 'd', r'\times', 'e', r'\div', 'f', r'\%', 'g']
exp7 = ['c', '=', "(", 'b', '+', 'c', '-', '2', ")", r'\div', 'z']
exp8 = ['A', '=', r'\{', '1', '2', ',', 'e', ',', '1', r'.', '5', ',', 'i', r'\}']
exp9 = ['B', '=', r'\{', '1', ',', '2', r'\}', r'\cup', r'\{', '1', '0', ',', '2', ',', '0', '\}']
exp10 = ['C', '=', r'\{', '5', '2', '5', ',', '2', '0', ',', '9', r'\}']

xml1 = '''
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow>
    <mi>z</mi>
    <mo>=</mo>
    <mn>2</mn>
  </mrow>
</math>
'''

xml2 = '''
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow>
    <mi>z</mi>
    <mo>=</mo>
    <mi>x</mi>
    <mo>+</mo>
    <mi>y</mi>
  </mrow>
</math>
'''

xml3 = '''
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow>
    <mi>c</mi>
    <mo>=</mo>
    <mi>a</mi>
    <mo>+</mo>
    <mi>b</mi>
  </mrow>
</math>
'''

xml4 = '''
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow>
    <mi>z</mi>
    <mo>=</mo>
    <mn>2</mn>
    <mo>+</mo>
    <mn>3</mn>
  </mrow>
</math>
'''

xml5 = '''
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow>
    <mi>a</mi>
    <mo>=</mo>
    <mn>21.50</mn>
    <mo>+</mo>
    <mn>32.4</mn>
  </mrow>
</math>
'''

xml6 = '''
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow>
    <mi>a</mi>
    <mo>=</mo>
    <mi>b</mi>
    <mo>+</mo>
    <mi>c</mi>
    <mo>&#8722;</mo>
    <mi>d</mi>
    <mo>&#215;</mo>
    <mi>e</mi>
    <mo>&#247;</mo>
    <mi>f</mi>
    <mo>%</mo>
    <mi>g</mi>
  </mrow>
</math>
'''

xml7 = '''
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow>
    <mi>c</mi>
    <mo>=</mo>
    <mo>(</mo>
    <mi>b</mi>
    <mo>+</mo>
    <mi>c</mi>
    <mo>&#8722;</mo>
    <mn>2</mn>
    <mo>)</mo>
    <mo>&#247;</mo>
    <mi>z</mi>
  </mrow>
</math>
'''

xml8 = '''
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow>
    <mi>A</mi>
    <mo>=</mo>
    <mo>{</mo>
    <mn>12</mn>
    <mo>,</mo>
    <mi>e</mi>
    <mo>,</mo>
    <mn>1.5</mn>
    <mo>,</mo>
    <mi>i</mi>
    <mo>}</mo>
  </mrow>
</math>
'''

xml9 = '''
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow>
    <mi>B</mi>
    <mo>=</mo>
    <mo>{</mo>
    <mn>1</mn>
    <mo>,</mo>
    <mn>2</mn>
    <mo>}</mo>
    <mo>&#8746;</mo>
    <mo>{</mo>
    <mn>10</mn>
    <mo>,</mo>
    <mn>2</mn>
    <mo>,</mo>
    <mn>0</mn>
    <mo>}</mo>
  </mrow>
</math>
'''

xml10 = '''
<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mrow>
    <mi>C</mi>
    <mo>=</mo>
    <mo>{</mo>
    <mn>525</mn>
    <mo>,</mo>
    <mn>20</mn>
    <mo>,</mo>
    <mn>9</mn>
    <mo>}</mo>
  </mrow>
</math>
'''


lexer = Lexer()
parser = Parser()
codegen = CodeGenerator()


def strip_snt(v):
    return re.sub(r'[\n\s\t]*', '', v)


def syn_analysis(exp):
    tokens = lexer.generate_tokens(exp)
    tree = parser.generate_tree(tokens)
    xml_tree = codegen.gen_mathml(tree)
    xml = ET.tostring(xml_tree, pretty_print=False).decode('utf-8')
    return xml


def test():
    exps = [exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9, exp10]
    xmls = [xml1, xml2, xml3, xml4, xml5, xml6, xml7, xml8, xml9, xml10]
    for exp, xml in zip(exps, xmls):
        assert strip_snt(syn_analysis(exp)) == strip_snt(xml)