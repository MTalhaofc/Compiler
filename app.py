import streamlit as st
import re
from collections import defaultdict


# Lexical Analyzer
class LexicalAnalyzer:
    def __init__(self):
        self.tokens = []

    def analyze(self, text):
        token_specification = [
            ('NUMBER', r'\d+'),           # Integer or decimal number
            ('ASSIGN', r'='),             # Assignment operator
            ('END', r';'),                # Statement terminator
            ('ID', r'[a-zA-Z_][a-zA-Z_0-9]*'),  # Identifiers
            ('OP', r'[+\-*/]'),           # Arithmetic operators
            ('SKIP', r'[ \t]+'),          # Skip over spaces and tabs
            ('NEWLINE', r'\n'),           # Line endings
            ('MISMATCH', r'.'),           # Any other character
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
        self.parse_table = []  # Parse table to store grammar rules applied
        self.parse_tree = []   # Parse tree to represent the structure

    def peek(self):
        if self.current_token_index < len(self.tokens):
            return self.tokens[self.current_token_index]
        return None

    def consume(self):
        current = self.peek()
        self.current_token_index += 1
        return current

    def add_to_parse_table(self, non_terminal, rule):
        """Adds a step to the parse table."""
        self.parse_table.append({"Non-Terminal": non_terminal, "Rule": rule})

    def statement(self):
        """Parses a single statement and builds part of the parse tree."""
        token = self.consume()
        if token is None:
            raise SyntaxError('Unexpected end of input while parsing statement')
        if token[0] == 'ID':
            self.add_to_parse_table('STATEMENT', 'STATEMENT → ID = EXPRESSION ;')
            id_node = {'type': 'ID', 'value': token[1]}
            assign_token = self.consume()
            if assign_token is None or assign_token[0] != 'ASSIGN':
                raise SyntaxError('Expected ASSIGN (=)')
            expression_node = self.expression()
            end_token = self.consume()
            if end_token is None or end_token[0] != 'END':
                raise SyntaxError('Expected END (;)')
            statement_tree = {
                'type': 'ASSIGNMENT',
                'id': id_node,
                'expression': expression_node
            }
            self.parse_tree.append(statement_tree)
            return statement_tree
        else:
            raise SyntaxError('Expected ID')

    def expression(self):
        """Parses an expression."""
        left = self.term()
        while self.peek() is not None and self.peek()[0] == 'OP' and self.peek()[1] in '+-':
            operator = self.consume()[1]
            right = self.term()
            self.add_to_parse_table('EXPRESSION', 'EXPRESSION → EXPRESSION OP TERM')
            left = {'type': 'BINARY_OP', 'operator': operator, 'left': left, 'right': right}
        return left

    def term(self):
        """Parses a term."""
        left = self.factor()
        while self.peek() is not None and self.peek()[0] == 'OP' and self.peek()[1] in '*/':
            operator = self.consume()[1]
            right = self.factor()
            self.add_to_parse_table('TERM', 'TERM → TERM OP FACTOR')
            left = {'type': 'BINARY_OP', 'operator': operator, 'left': left, 'right': right}
        return left

    def factor(self):
        """Parses a factor."""
        token = self.consume()
        if token is None:
            raise SyntaxError('Unexpected end of input while parsing factor')
        if token[0] == 'NUMBER':
            self.add_to_parse_table('FACTOR', 'FACTOR → NUMBER')
            return {'type': 'NUMBER', 'value': int(token[1])}
        elif token[0] == 'ID':
            self.add_to_parse_table('FACTOR', 'FACTOR → ID')
            return {'type': 'ID', 'value': token[1]}
        else:
            raise SyntaxError('Invalid Factor')

    def parse(self):
        """Parses the input tokens into an AST and generates the parse tree."""
        ast = []
        while self.peek() is not None:
            ast.append(self.statement())
        return ast


# Streamlit App
def main():
    st.title("Simple Compiler with Parse Tree and Parse Table")

    st.sidebar.header("Compiler Options")
    code_input = st.text_area("Enter your code:", height=200, placeholder="x = 10;\ny = x + 5;")

    if st.button("Compile"):
        if not code_input.strip():
            st.error("Please enter some code!")
            return

        # Lexical Analysis
        lexer = LexicalAnalyzer()
        try:
            tokens = lexer.analyze(code_input)
        except SyntaxError as e:
            st.error(f"Lexical Error: {e}")
            return

        st.subheader("Tokens")
        for token in tokens:
            st.write(token)

        # Parsing
        parser = Parser(tokens)
        try:
            ast = parser.parse()
        except SyntaxError as e:
            st.error(f"Parsing Error: {e}")
            return

        st.subheader("Abstract Syntax Tree (AST)")
        for node in ast:
            st.write(node)

        st.subheader("Parse Tree")
        for node in parser.parse_tree:
            st.json(node)

        st.subheader("Parse Table")
        for entry in parser.parse_table:
            st.write(entry)


if __name__ == "__main__":
    main()
