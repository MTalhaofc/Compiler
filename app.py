import streamlit as st
import re

# Lexical Analyzer
class LexicalAnalyzer:
    def __init__(self):
        self.tokens = []

    def analyze(self, text):
        self.tokens = []  # Clear tokens for each analysis
        token_specification = [
            ('NUMBER',   r'\d+'),           # Integer or decimal number
            ('ASSIGN',   r'='),             # Assignment operator
            ('END',      r';'),             # Statement terminator
            ('ID',       r'[a-zA-Z_][a-zA-Z_0-9]*'),  # Identifiers
            ('OP',       r'[+\-*/]'),       # Arithmetic operators
            ('SKIP',     r'[ \t]+'),        # Skip over spaces and tabs
            ('NEWLINE',  r'\n'),            # Line endings
            ('MISMATCH', r'.'),             # Any other character
        ]

        tok_regex = '|'.join(f'(?P<{pair[0]}>{pair[1]})' for pair in token_specification)
        get_token = re.compile(tok_regex).match
        line_number = 1
        position = line_start = 0

        while position < len(text):
            match = get_token(text, position)
            if match is None:
                raise SyntaxError(f'Unexpected character {text[position]!r} at line {line_number}')
            type_ = match.lastgroup
            value = match.group(type_)
            if type_ == 'NEWLINE':
                line_start = position
                line_number += 1
            elif type_ != 'SKIP':
                self.tokens.append((type_, value, line_number))
            position = match.end()

        return self.tokens


# Parser
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_token_index = 0

    def peek(self):
        if self.current_token_index < len(self.tokens):
            return self.tokens[self.current_token_index]
        return None

    def consume(self):
        current = self.peek()
        self.current_token_index += 1
        return current

    def parse(self):
        ast = []
        while self.peek() is not None:
            ast.append(self.statement())
        return ast

    def statement(self):
        token = self.consume()
        if token is None:
            raise SyntaxError('Unexpected end of input while parsing statement')
        if token[0] == 'ID':
            id_node = {'type': 'ID', 'value': token[1]}
            assign_token = self.consume()
            if assign_token is None or assign_token[0] != 'ASSIGN':
                raise SyntaxError('Expected ASSIGN (=)')
            expression_node = self.expression()
            end_token = self.consume()
            if end_token is None or end_token[0] != 'END':
                raise SyntaxError('Expected END (;)')
            return {'type': 'ASSIGNMENT', 'id': id_node, 'expression': expression_node}
        else:
            raise SyntaxError('Expected ID')

    def expression(self):
        left = self.term()
        while self.peek() is not None and self.peek()[0] == 'OP' and self.peek()[1] in '+-':
            operator = self.consume()[1]
            right = self.term()
            left = {'type': 'BINARY_OP', 'operator': operator, 'left': left, 'right': right}
        return left

    def term(self):
        left = self.factor()
        while self.peek() is not None and self.peek()[0] == 'OP' and self.peek()[1] in '*/':
            operator = self.consume()[1]
            right = self.factor()
            left = {'type': 'BINARY_OP', 'operator': operator, 'left': left, 'right': right}
        return left

    def factor(self):
        token = self.consume()
        if token is None:
            raise SyntaxError('Unexpected end of input while parsing factor')
        if token[0] == 'NUMBER':
            return {'type': 'NUMBER', 'value': int(token[1])}
        elif token[0] == 'ID':
            return {'type': 'ID', 'value': token[1]}
        else:
            raise SyntaxError('Invalid Factor')


# Semantic Analyzer
class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = {}

    def analyze(self, ast):
        for node in ast:
            if node['type'] == 'ASSIGNMENT':
                self.evaluate_assignment(node)

    def evaluate_assignment(self, node):
        identifier = node['id']['value']
        value = self.evaluate_expression(node['expression'])
        self.symbol_table[identifier] = value

    def evaluate_expression(self, node):
        if node['type'] == 'NUMBER':
            return node['value']
        elif node['type'] == 'ID':
            if node['value'] in self.symbol_table:
                return self.symbol_table[node['value']]
            else:
                raise ValueError(f'Undefined variable {node["value"]}')
        elif node['type'] == 'BINARY_OP':
            left_value = self.evaluate_expression(node['left'])
            right_value = self.evaluate_expression(node['right'])
            if node['operator'] == '+':
                return left_value + right_value
            elif node['operator'] == '-':
                return left_value - right_value
            elif node['operator'] == '*':
                return left_value * right_value
            elif node['operator'] == '/':
                if right_value == 0:
                    raise ValueError('Division by zero')
                return left_value / right_value
        else:
            raise ValueError('Invalid Expression')

    def get_symbol_table(self):
        return self.symbol_table


# Streamlit App
def main():
    st.title("Simple Arithmetic Compiler")

    # User input
    st.subheader("Input Code")
    input_code = st.text_area("Enter your code here (end statements with ';')", height=200)

    if st.button("Compile"):
        try:
            # Lexical Analysis
            lexer = LexicalAnalyzer()
            tokens = lexer.analyze(input_code)

            # Display tokens
            st.subheader("Tokens")
            for token in tokens:
                st.text(token)

            # Parsing
            parser = Parser(tokens)
            ast = parser.parse()

            # Display AST
            st.subheader("Abstract Syntax Tree (AST)")
            st.json(ast)

            # Semantic Analysis
            semantic_analyzer = SemanticAnalyzer()
            semantic_analyzer.analyze(ast)

            # Display Symbol Table
            st.subheader("Symbol Table")
            symbol_table = semantic_analyzer.get_symbol_table()
            for identifier, value in symbol_table.items():
                st.text(f"{identifier} = {value}")

            # Result Output
            st.subheader("Execution Result")
            for identifier, value in symbol_table.items():
                st.text(f"{identifier} = {value}")

        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
