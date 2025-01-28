mod lexing {

    use logos::Logos;

    use crate::ast::Span;

    #[derive(Debug, Clone, Default, PartialEq)]
    pub enum LexingError {
        IntegerOverflow {
            slice: String,
            span: Span,
        },
        #[default]
        UnknownToken,
    }

    #[derive(Clone, Debug, Logos)]
    #[logos(error = LexingError)]
    #[logos(skip r#"[ \t\n]+"#)]
    pub enum Token {
        #[token("struct")]
        Struct,
        #[token("int")]
        Int,
        #[token("void")]
        Void,
        #[token("*")]
        Asterisk,
        #[token("-")]
        Minus,
        #[token("+")]
        Plus,
        #[token("/")]
        Slash,
        #[token("{")]
        OpenCurly,
        #[token("}")]
        CloseCurly,
        #[token("(")]
        OpenParen,
        #[token(")")]
        CloseParen,
        #[token("[")]
        OpenSquare,
        #[token("]")]
        CloseSquare,
        #[regex(r#"\d+"#, |lex| lex.slice().parse().map_err(|_| LexingError::IntegerOverflow { slice: lex.slice().to_owned(), span: lex.span().into() }))]
        #[regex(r#"0x[\da-fA-F]+"#, |lex| u32::from_str_radix(lex.slice(), 16).map_err(|_| {
            LexingError::IntegerOverflow { slice: lex.slice().to_owned(), span: lex.span().into() }
        }).map(|n| n as i32))]
        NumberLiteral(i32),
        #[regex(r#"[_a-zA-Z][_a-zA-Z\d]*"#, |lex| lex.slice().to_string())]
        Ident(String),
        #[token(";")]
        Semicolon,
        #[token("=")]
        Equals,
        #[token("==")]
        DoubleEquals,
        #[token(">=")]
        GreaterThanEquals,
        #[token("<=")]
        LessThanEquals,
        #[token(">")]
        GreaterThan,
        #[token("<")]
        LessThan,
        #[token("!=")]
        NotEquals,
        #[token("~")]
        Tilde,
        #[token("^")]
        Circumflex,
        #[token("&")]
        Ampersand,
        #[token("|")]
        Pipe,
        #[token("if")]
        If,
        #[token("while")]
        While,
        #[token("for")]
        For,
        #[token("++")]
        Increment,
        #[token("--")]
        Decrement,
        #[token("->")]
        Arrow,
        #[token("+=")]
        CompoundAdd,
        #[token("-=")]
        CompoundSub,
        #[token("*=")]
        CompoundMul,
        #[token("/=")]
        CompoundDiv,
        #[token("fn")]
        Function,
        #[token("nullptr")]
        NullPtr,
        EOF,
    }

    pub trait PeekableLexer<T> {
        fn peek(&self) -> Option<Result<T, LexingError>>;
    }

    impl PeekableLexer<Token> for logos::Lexer<'_, Token> {
        fn peek(&self) -> Option<Result<Token, LexingError>> {
            self.clone().next()
        }
    }
}

mod parsing {
    use super::ast::*;
    use super::lexing::*;
    use logos::Lexer;
    use logos::Logos as _;
    #[derive(Clone, Debug)]
    struct Parser<'s> {
        lexer: Lexer<'s, Token>,
    }

    #[derive(Debug, Clone)]
    enum ParsingError {
        Lexing(LexingError),
        EOF,
        WrongToken {
            expected: Expected,
            found: Token,
            span: Span,
        },
    }

    #[derive(Debug, Clone)]
    enum Expected {
        Str(&'static str),
        Token(Token),
    }

    impl From<LexingError> for ParsingError {
        fn from(value: LexingError) -> Self {
            ParsingError::Lexing(value)
        }
    }

    #[derive(Debug, Clone)]
    struct Checkpoint<'s> {
        lexer: Lexer<'s, Token>,
    }

    impl<'s> Parser<'s> {
        pub fn new(s: &'s str) -> Self {
            Self {
                lexer: Token::lexer(s),
            }
        }
        fn checkpoint(&self) -> Checkpoint<'s> {
            Checkpoint {
                lexer: self.lexer.clone(),
            }
        }
        fn restore(&mut self, checkpoint: Checkpoint<'s>) {
            self.lexer = checkpoint.lexer;
        }
    }
    impl Parser<'_> {
        pub fn expect_semicolon(&mut self) -> Result<(), ParsingError> {
            let semi = self.lexer.next().ok_or(ParsingError::WrongToken {
                expected: Expected::Token(Token::Semicolon),
                found: Token::EOF,
                span: self.lexer.span().into(),
            })??;
            match semi {
                Token::Semicolon => Ok(()),
                other => Err(ParsingError::WrongToken {
                    expected: Expected::Token(Token::Semicolon),
                    found: other,
                    span: self.lexer.span().into(),
                }),
            }
        }
        pub fn parse_program(&mut self) -> Result<Program, ParsingError> {
            todo!()
        }
        pub fn parse_toplevel(&mut self) -> Result<TopLevel, ParsingError> {
            todo!()
        }
        pub fn parse_function_decl(&mut self) -> Result<FunctionDecl, ParsingError> {
            todo!()
        }
        pub fn parse_struct_decl(&mut self) -> Result<StructDecl, ParsingError> {
            todo!()
        }
        pub fn parse_variable_decl(&mut self) -> Result<VariableDecl, ParsingError> {
            let typ = self.parse_type()?;
            let res = self.lexer.next().ok_or(ParsingError::EOF)??;
            let base = match res {
                Token::Ident(ident) => ident,
                other => {
                    return Err(ParsingError::WrongToken {
                        expected: Expected::Str("variable name"),
                        found: other,
                        span: self.lexer.span().into(),
                    })
                }
            };
            // handle arrays
            // handle assignment sign (=)
            // handle expression
            // handle semicolon
            todo!()
        }
        pub fn parse_type(&mut self) -> Result<Type, ParsingError> {
            let res = self.lexer.next().ok_or(ParsingError::EOF)??;
            let base = match res {
                Token::Struct => {
                    let struct_name = self.lexer.next().ok_or(ParsingError::EOF)??;
                    match struct_name {
                        Token::Ident(ident) => Type::Struct(ident),
                        other => {
                            return Err(ParsingError::WrongToken {
                                expected: Expected::Str("struct name"),
                                found: other,
                                span: self.lexer.span().into(),
                            })
                        }
                    }
                }
                Token::Int => Type::Int,
                Token::Void => Type::Void,
                Token::Function => todo!("need punctuated support"),
                other => {
                    return Err(ParsingError::WrongToken {
                        expected: Expected::Str("type"),
                        found: other,
                        span: self.lexer.span().into(),
                    })
                }
            };
            let mut curr = base;
            while let Some(Ok(Token::Asterisk)) = self.lexer.peek() {
                self.lexer.next();
                curr = Type::Ptr(Box::new(curr));
            }
            Ok(curr)
        }
        pub fn parse_ident(&mut self, err: &'static str) -> Result<String, ParsingError> {
            let struct_name = self.lexer.next().ok_or(ParsingError::EOF)??;
            match struct_name {
                Token::Ident(ident) => Ok(ident),
                other => Err(ParsingError::WrongToken {
                    expected: Expected::Str("struct name"),
                    found: other,
                    span: self.lexer.span().into(),
                }),
            }
        }
        pub fn parse_expr(&mut self) -> Result<Expression, ParsingError> {
            todo!()
        }
        fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expression, ParsingError> {
            let lhs = self.lexer.next().ok_or(ParsingError::WrongToken {
                expected: Expected::Str("expression"),
                found: Token::EOF,
                span: self.lexer.span().into(),
            })??;
            let mut lhs = match lhs {
                Token::NumberLiteral(n) => Expression::IntLiteral(n),
                Token::Ident(ident) => Expression::Ident(ident),
                Token::NullPtr => Expression::NullPtr,
                // TODO parse as type cast
                Token::OpenParen => {
                    let expr = self.parse_expr_bp(0)?;
                    let closing = self.lexer.next().ok_or(ParsingError::WrongToken {
                        expected: Expected::Str("expression"),
                        found: Token::EOF,
                        span: self.lexer.span().into(),
                    })??;
                    match closing {
                        Token::CloseParen => (),
                        other => {
                            return Err(ParsingError::WrongToken {
                                expected: Expected::Token(Token::CloseParen),
                                found: other,
                                span: self.lexer.span().into(),
                            })
                        }
                    }
                    expr
                }

                token @ (Token::Asterisk
                | Token::Minus
                | Token::Plus
                | Token::OpenSquare
                | Token::Tilde
                | Token::Increment
                | Token::Decrement) => {
                    let ((), rbp) = prefix_binding_power(&token);
                    let inner = self.parse_expr_bp(rbp)?;
                    match token {
                        Token::Asterisk => Expression::AsteriskDereference(Box::new(inner)),
                        Token::Minus => Expression::UnaryNegation(Box::new(inner)),
                        Token::Plus => Expression::UnaryPlus(Box::new(inner)),
                        Token::Tilde => Expression::BitwiseNot(Box::new(inner)),
                        Token::Increment => Expression::PreIncrement(Box::new(inner)),
                        Token::Decrement => Expression::PreDecrement(Box::new(inner)),
                        _ => unreachable!(),
                    }
                }
                other => {
                    return Err(ParsingError::WrongToken {
                        expected: Expected::Str("expression"),
                        found: other,
                        span: self.lexer.span().into(),
                    })
                }
            };
            while let Some(op) = self.lexer.peek() {
                let op = match op? {
                    Token::Arrow => {
                        self.lexer.next();
                        let field = self.lexer.next().ok_or(ParsingError::WrongToken {
                            expected: Expected::Str("struct member"),
                            found: Token::EOF,
                            span: self.lexer.span().into(),
                        })??;
                        match field {
                            Token::Ident(ident) => {
                                lhs = Expression::ArrowDereference(Box::new(lhs), ident);
                                continue;
                            }
                            other => {
                                return Err(ParsingError::WrongToken {
                                    expected: Expected::Str("struct member"),
                                    found: other,
                                    span: self.lexer.span().into(),
                                })
                            }
                        }
                    }
                    t @ (Token::Asterisk
                    // TODO single equals assignment return
                    // TODO shift{left,right}
                    | Token::Minus
                    | Token::Plus
                    | Token::Slash
                    | Token::OpenParen
                    | Token::OpenSquare
                    | Token::DoubleEquals
                    | Token::GreaterThanEquals
                    | Token::LessThanEquals
                    | Token::GreaterThan
                    | Token::LessThan
                    | Token::NotEquals
                    | Token::Circumflex
                    | Token::Ampersand
                    | Token::Pipe
                    | Token::Increment
                    | Token::Decrement) => t,
                    Token::Semicolon
                    | Token::Equals
                    | Token::CompoundAdd
                    | Token::CompoundSub
                    | Token::CompoundMul
                    | Token::CompoundDiv => break,
                    Token::EOF => unreachable!(),
                    t => {
                        return Err(ParsingError::WrongToken {
                            expected: Expected::Str("operator"),
                            found: t,
                            span: self.lexer.span().into(),
                        })
                    }
                };
                if let Some((l_bp, ())) = postfix_binding_power(&op) {
                    if l_bp < min_bp {
                        break;
                    }
                    self.lexer.next();
                    match op {
                        Token::OpenParen => todo!(),
                        Token::OpenSquare => todo!(),
                        Token::Increment => lhs = Expression::PostIncrement(Box::new(lhs)),
                        Token::Decrement => lhs = Expression::PostDecrement(Box::new(lhs)),
                        _ => unreachable!(),
                    }
                    // lhs = S::Cons(op, vec![lhs]);
                    continue;
                }
                if let Some((lbp, rbp)) = infix_binding_power(&op) {
                    if lbp < min_bp {
                        break;
                    }
                    self.lexer.next();
                    let rhs = self.parse_expr_bp(rbp)?;
                    match op {
                        Token::DoubleEquals => {
                            lhs = Expression::Equals(Box::new(lhs), Box::new(rhs))
                        }
                        Token::Asterisk => lhs = Expression::Mul(Box::new(lhs), Box::new(rhs)),
                        Token::Minus => lhs = Expression::Sub(Box::new(lhs), Box::new(rhs)),
                        Token::Plus => lhs = Expression::Add(Box::new(lhs), Box::new(rhs)),
                        Token::Slash => lhs = Expression::Div(Box::new(lhs), Box::new(rhs)),
                        Token::GreaterThanEquals => {
                            lhs = Expression::GreaterThanOrEquals(Box::new(lhs), Box::new(rhs))
                        }

                        Token::LessThanEquals => {
                            lhs = Expression::LessThanOrEquals(Box::new(lhs), Box::new(rhs))
                        }
                        Token::GreaterThan => {
                            lhs = Expression::GreaterThan(Box::new(lhs), Box::new(rhs))
                        }
                        Token::LessThan => lhs = Expression::LessThan(Box::new(lhs), Box::new(rhs)),
                        Token::NotEquals => {
                            lhs = Expression::NotEquals(Box::new(lhs), Box::new(rhs))
                        }
                        Token::Circumflex => {
                            lhs = Expression::BitwiseXor(Box::new(lhs), Box::new(rhs))
                        }
                        Token::Ampersand => {
                            lhs = Expression::BitwiseAnd(Box::new(lhs), Box::new(rhs))
                        }
                        Token::Pipe => lhs = Expression::BitwiseOr(Box::new(lhs), Box::new(rhs)),
                        _ => unreachable!(),
                    }
                }
                break;
            }
            Ok(lhs)
        }
        pub fn parse_assign_to(&mut self) -> Result<Type, ParsingError> {
            todo!()
        }
    }
    fn prefix_binding_power(op: &Token) -> ((), u8) {
        match op {
            // '+' | '-' => ((), 5),
            _ => todo!(),
        }
    }
    fn infix_binding_power(op: &Token) -> Option<(u8, u8)> {
        match op {
            // '+' | '-' => (1, 2),
            // '*' | '/' => (3, 4),
            // '.' => (8, 7),
            _ => panic!("bad op: {op:?}"),
        }
    }
    fn postfix_binding_power(op: &Token) -> Option<(u8, ())> {
        let res = match op {
            // '!' => (7, ()),
            _ => return todo!(),
        };
        Some(res)
    }
}

mod ast {
    pub use super::span::*;
    #[derive(Debug, Clone)]
    pub enum Type {
        Void,
        Int,
        Struct(String),
        Ptr(Box<Type>),
        Fn(String),
    }

    #[derive(Clone, Debug)]
    pub struct Program {
        pub top_level: Vec<TopLevel>,
    }

    #[derive(Clone, Debug)]
    pub enum TopLevel {
        FunctionDecl(FunctionDecl),
        VariableDecl(VariableDecl),
        StructDecl(StructDecl),
    }

    #[derive(Clone, Debug)]
    pub struct StructDecl {
        pub name: String,
        pub members: Vec<NameAndType>,
    }

    #[derive(Clone, Debug)]
    pub struct NameAndType {
        pub name: String,
        pub typ: Type,
    }

    #[derive(Clone, Debug)]
    pub struct FunctionDecl {
        pub return_type: Type,
        pub name: String,
        pub parameters: Vec<NameAndType>,
        pub block: Block,
    }

    #[derive(Debug, Clone)]
    pub struct VariableDecl {
        pub typ: Type,
        pub name: String,
        pub assigned: Option<Expression>,
    }

    #[derive(Debug, Clone)]
    pub enum Statement {
        Expression(Expression),
        Declaration(VariableDecl),
        Block(Block),
        If(If),
        While(While),
        For(For),
        Parenthesized(Box<Statement>),
        // TODO LVALUE ALL
        Assignment {
            to: Box<Expression>,
            value: Box<Expression>,
        },
        CompoundAdd(Box<Expression>, Box<Expression>),
        CompoundSub(Box<Expression>, Box<Expression>),
        CompoundMul(Box<Expression>, Box<Expression>),
        CompoundDiv(Box<Expression>, Box<Expression>),
        CompoundBitwiseAnd(Box<Expression>, Box<Expression>),
        CompoundBitwiseOr(Box<Expression>, Box<Expression>),
        CompoundBitwiseXor(Box<Expression>, Box<Expression>),
        // TODO goto?
    }
    #[derive(Debug, Clone)]
    pub enum Expression {
        Equals(Box<Expression>, Box<Expression>),
        NotEquals(Box<Expression>, Box<Expression>),
        GreaterThanOrEquals(Box<Expression>, Box<Expression>),
        LessThanOrEquals(Box<Expression>, Box<Expression>),
        GreaterThan(Box<Expression>, Box<Expression>),
        LessThan(Box<Expression>, Box<Expression>),
        ArrayIndex(Box<Expression>, Box<Expression>),
        AsteriskDereference(Box<Expression>),
        ArrowDereference(Box<Expression>, String),
        IntLiteral(i32),
        Ident(String),
        Cast(Box<Expression>, Type),
        // Parenthesized(Box<Expression>),
        Add(Box<Expression>, Box<Expression>),
        Sub(Box<Expression>, Box<Expression>),
        Mul(Box<Expression>, Box<Expression>),
        Div(Box<Expression>, Box<Expression>),
        BitwiseAnd(Box<Expression>, Box<Expression>),
        BitwiseOr(Box<Expression>, Box<Expression>),
        BitwiseXor(Box<Expression>, Box<Expression>),
        BitwiseNot(Box<Expression>),
        ShiftLeft(Box<Expression>, Box<Expression>),
        ShiftRight(Box<Expression>, Box<Expression>),
        FunctionCall(String, Vec<Expression>),
        StructInstantiation(Vec<Expression>),
        NullPtr,
        UnaryNegation(Box<Expression>),
        UnaryPlus(Box<Expression>),
        PreIncrement(Box<Expression>),
        PreDecrement(Box<Expression>),
        PostIncrement(Box<Expression>),
        PostDecrement(Box<Expression>),
    }

    #[derive(Debug, Clone)]
    pub struct If {
        pub condition: Expression,
        pub block: Block,
    }
    #[derive(Debug, Clone)]
    pub struct While {
        pub condtion: Expression,
        pub block: Block,
    }
    #[derive(Debug, Clone)]
    pub struct For {
        pub init: Box<Statement>,
        pub condition: Expression,
        pub after: Box<Statement>,
        pub block: Block,
    }

    #[derive(Debug, Clone)]
    pub struct Block {
        pub statements: Vec<Statement>,
    }

    // #[derive(Debug, Clone)]
    // pub enum LValue {
    //     Ident(String),
    //     Dereference(Expression),
    //     ArrayIndex { ptr: String, index: Box<Expression> },
    //     StructureMember { struct_: String, member: String },
    //     ArrowDereference { struct_: String, member: String },
    //     Parenthesized(Box<LValue>),
    // }
}

// mod assembling {
//     #[derive(Clone, Debug)]
//     struct Assembler {
//         code: String,
//     }
// }

mod span {
    use std::ops::{Deref, DerefMut, Range};

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Span {
        pub start: usize,
        pub end: usize,
    }

    impl From<Range<usize>> for Span {
        fn from(value: Range<usize>) -> Self {
            Self {
                start: value.start,
                end: value.end,
            }
        }
    }

    impl Span {
        pub fn new(start: usize, end: usize) -> Self {
            Self { start, end }
        }
        pub fn into_inner(self) -> (usize, usize) {
            (self.start, self.end)
        }
    }
    #[derive(Debug, Clone, Copy)]
    pub struct Spanned<T> {
        pub inner: T,
        pub span: Span,
    }
    impl<T> Spanned<T> {
        pub fn into_inner(self) -> T {
            self.inner
        }
    }
    impl<T> Deref for Spanned<T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            &self.inner
        }
    }
    impl<T> DerefMut for Spanned<T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.inner
        }
    }
}
