mod testing_utils {
    pub trait Boxed: Sized {
        fn boxed(self) -> Box<Self> {
            Box::new(self)
        }
    }
    impl<T> Boxed for T {}
}
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
        #[token(r"\^=")]
        CompoundXor,
        #[token(r"&=")]
        CompoundAnd,
        #[token(r"\|=")]
        CompoundOr,
        #[token("fn")]
        Function,
        #[token("nullptr")]
        NullPtr,
        #[token("as")]
        AsCast,
        #[token(",")]
        Comma,
        #[token(r"\.")]
        Period,
        #[token(">>")]
        Shr,
        #[token("<<")]
        Shl,
        #[token("!")]
        LogicalNot,
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
            self.parse_expr_bp(0)
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
                Token::OpenParen => {
                    let expr = self.parse_expr_bp(0)?;
                    let closing = self.lexer.next().ok_or(ParsingError::WrongToken {
                        expected: Expected::Token(Token::CloseParen),
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
                | Token::LogicalNot
                | Token::Ampersand
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
                        Token::LogicalNot => Expression::LogicalNot(Box::new(inner)),
                        Token::Ampersand => Expression::AddrOf(Box::new(inner)),
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
                    Token::Period => {
                        self.lexer.next();
                        let field = self.lexer.next().ok_or(ParsingError::WrongToken {
                            expected: Expected::Str("struct member"),
                            found: Token::EOF,
                            span: self.lexer.span().into(),
                        })??;
                        match field {
                            Token::Ident(ident) => {
                                lhs = Expression::MemberAccess(Box::new(lhs), ident);
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
                    | Token::Decrement
                    | Token::Equals
                    | Token::Shl
                    | Token::Shr
                    | Token::CompoundAdd
                    | Token::CompoundSub
                    | Token::CompoundMul
                    | Token::CompoundDiv
                    | Token::CompoundAnd
                    | Token::CompoundOr
                    | Token::CompoundXor
                    | Token::AsCast) => t,
                    Token::CloseParen | Token::CloseSquare | Token::Semicolon => break,
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
                        Token::OpenParen => {
                            let exprs = self.parse_args()?;
                            let closing = self.lexer.next().ok_or(ParsingError::WrongToken {
                                expected: Expected::Token(Token::CloseParen),
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
                            lhs = Expression::FunctionCall(Box::new(lhs), exprs);
                        }
                        Token::OpenSquare => {
                            let expr = self.parse_expr_bp(0)?;
                            let closing = self.lexer.next().ok_or(ParsingError::WrongToken {
                                expected: Expected::Token(Token::CloseSquare),
                                found: Token::EOF,
                                span: self.lexer.span().into(),
                            })??;
                            match closing {
                                Token::CloseSquare => (),
                                other => {
                                    return Err(ParsingError::WrongToken {
                                        expected: Expected::Token(Token::CloseSquare),
                                        found: other,
                                        span: self.lexer.span().into(),
                                    })
                                }
                            }
                            lhs = Expression::ArrayIndex(Box::new(lhs), Box::new(expr));
                        }
                        Token::Increment => lhs = Expression::PostIncrement(Box::new(lhs)),
                        Token::Decrement => lhs = Expression::PostDecrement(Box::new(lhs)),
                        Token::AsCast => {
                            let typ = self.parse_type()?;
                            lhs = Expression::Cast(Box::new(lhs), typ);
                        }
                        _ => unreachable!(),
                    }
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
                        Token::Equals => lhs = Expression::Assignment(Box::new(lhs), Box::new(rhs)),
                        Token::Shl => lhs = Expression::ShiftLeft(Box::new(lhs), Box::new(rhs)),
                        Token::Shr => lhs = Expression::ShiftRight(Box::new(lhs), Box::new(rhs)),
                        Token::CompoundAdd => {
                            lhs = Expression::CompoundAdd(Box::new(lhs), Box::new(rhs))
                        }
                        Token::CompoundSub => {
                            lhs = Expression::CompoundSub(Box::new(lhs), Box::new(rhs))
                        }
                        Token::CompoundMul => {
                            lhs = Expression::CompoundMul(Box::new(lhs), Box::new(rhs))
                        }
                        Token::CompoundDiv => {
                            lhs = Expression::CompoundDiv(Box::new(lhs), Box::new(rhs))
                        }
                        Token::CompoundAnd => {
                            lhs = Expression::CompoundAnd(Box::new(lhs), Box::new(rhs))
                        }
                        Token::CompoundOr => {
                            lhs = Expression::CompoundOr(Box::new(lhs), Box::new(rhs))
                        }
                        Token::CompoundXor => {
                            lhs = Expression::CompoundXor(Box::new(lhs), Box::new(rhs))
                        }
                        _ => unreachable!(),
                    }
                    continue;
                }
                break;
            }
            Ok(lhs)
        }

        pub fn parse_args(&mut self) -> Result<Vec<Expression>, ParsingError> {
            match self.lexer.peek() {
                Some(Ok(t)) => {
                    if let Token::CloseParen = t {
                        // empty function call
                        return Ok(Vec::new());
                    }
                }
                Some(Err(e)) => return Err(e.into()),
                None => return Err(ParsingError::EOF),
            };
            let mut exprs = Vec::new();
            let first_expr = self.parse_expr()?;
            exprs.push(first_expr);
            loop {
                match self.lexer.peek() {
                    Some(Ok(t)) => {
                        if let Token::CloseParen = t {
                            return Ok(exprs);
                        } else if let Token::Comma = t {
                            // this is what we expect
                            self.lexer.next();
                        } else {
                            self.lexer.next();
                            return Err(ParsingError::WrongToken {
                                expected: Expected::Token(Token::CloseParen),
                                found: t,
                                span: self.lexer.span().into(),
                            });
                        }
                    }
                    Some(Err(e)) => return Err(e.into()),
                    None => return Err(ParsingError::EOF),
                };
                let expr = self.parse_expr()?;
                exprs.push(expr)
            }
        }

        pub fn parse_assign_to(&mut self) -> Result<Type, ParsingError> {
            todo!()
        }
    }
    fn prefix_binding_power(op: &Token) -> ((), u8) {
        match op {
            Token::Increment
            | Token::Decrement
            | Token::Minus
            | Token::Plus
            | Token::LogicalNot
            | Token::Tilde
            | Token::Asterisk => ((), 95),
            _ => todo!(),
        }
    }
    fn infix_binding_power(op: &Token) -> Option<(u8, u8)> {
        match op {
            Token::Asterisk | Token::Slash => Some((85, 86)),
            Token::Plus | Token::Minus => Some((80, 81)),
            Token::Shl | Token::Shr => Some((75, 76)),
            Token::GreaterThanEquals
            | Token::GreaterThan
            | Token::LessThanEquals
            | Token::LessThan => Some((70, 71)),
            Token::DoubleEquals | Token::NotEquals => Some((65, 66)),
            Token::Ampersand => Some((60, 61)),
            Token::Circumflex => Some((55, 56)),
            Token::Pipe => Some((50, 51)),
            Token::Equals
            | Token::CompoundAdd
            | Token::CompoundSub
            | Token::CompoundMul
            | Token::CompoundDiv
            | Token::CompoundXor
            | Token::CompoundAnd
            | Token::CompoundOr => Some((45, 46)),
            // '+' | '-' => (1, 2),
            // '*' | '/' => (3, 4),
            // '.' => (8, 7),
            _ => None,
        }
    }
    fn postfix_binding_power(op: &Token) -> Option<(u8, ())> {
        match op {
            Token::Increment | Token::Decrement | Token::OpenParen | Token::OpenSquare => {
                Some((101, ()))
            }
            Token::AsCast => Some((91, ())),
            _ => None,
        }
    }
    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn simple_parse() {
            let mut parser = Parser::new("Something * **x->y->something / ((5 as int**) * 5)");
            let parsed = parser.parse_expr().unwrap();
            dbg!(parsed);
            assert!(parser.lexer.next().is_none());
        }
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
        // TODO goto?
    }
    #[derive(Debug, Clone)]
    pub enum Expression {
        LogicalNot(Box<Expression>),
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
        FunctionCall(Box<Expression>, Vec<Expression>),
        StructInstantiation(Vec<Expression>),
        NullPtr,
        MemberAccess(Box<Expression>, String),
        UnaryNegation(Box<Expression>),
        UnaryPlus(Box<Expression>),
        PreIncrement(Box<Expression>),
        PreDecrement(Box<Expression>),
        PostIncrement(Box<Expression>),
        PostDecrement(Box<Expression>),
        Assignment(Box<Expression>, Box<Expression>),
        AddrOf(Box<Expression>),
        CompoundAdd(Box<Expression>, Box<Expression>),
        CompoundSub(Box<Expression>, Box<Expression>),
        CompoundMul(Box<Expression>, Box<Expression>),
        CompoundDiv(Box<Expression>, Box<Expression>),
        CompoundAnd(Box<Expression>, Box<Expression>),
        CompoundOr(Box<Expression>, Box<Expression>),
        CompoundXor(Box<Expression>, Box<Expression>),
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
