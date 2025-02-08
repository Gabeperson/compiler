#![allow(unused)]
#![allow(dead_code)]
mod span;
mod lexing {
    use crate::ast::{Span, Spanned};
    use regex::Regex;
    use std::sync::LazyLock;

    #[derive(Debug, Clone)]
    pub enum LexingError {
        UnknownToken { position: usize },
        Eof,
    }

    #[derive(Clone, Debug)]
    pub enum Token {
        Struct,
        Int,
        Void,
        Asterisk,
        Minus,
        Plus,
        Slash,
        OpenCurly,
        CloseCurly,
        OpenParen,
        CloseParen,
        OpenSquare,
        CloseSquare,
        DecLiteral(String),
        HexLiteral(String),
        Ident(String),
        Semicolon,
        Equals,
        DoubleEquals,
        GreaterThanEquals,
        LessThanEquals,
        GreaterThan,
        LessThan,
        NotEquals,
        Tilde,
        Circumflex,
        Ampersand,
        Pipe,
        If,
        While,
        For,
        Increment,
        Decrement,
        Arrow,
        CompoundAdd,
        CompoundSub,
        CompoundMul,
        CompoundDiv,
        CompoundXor,
        CompoundAnd,
        CompoundOr,
        Function,
        NullPtr,
        AsCast,
        Comma,
        Period,
        Shr,
        Shl,
        LogicalNot,
    }

    type LexConvertFunction = fn(&str) -> Token;

    static REGEX_DEFS: &[(&str, LexConvertFunction)] = &[
        (r"^\+\+", |_| Token::Increment),
        (r"^--", |_| Token::Decrement),
        (r"^->", |_| Token::Arrow),
        (r"^\+=", |_| Token::CompoundAdd),
        (r"^-=", |_| Token::CompoundSub),
        (r"^\*=", |_| Token::CompoundMul),
        (r"^>>", |_| Token::Shr),
        (r"^<<", |_| Token::Shl),
        (r"^>=", |_| Token::GreaterThanEquals),
        (r"^<=", |_| Token::LessThanEquals),
        (r"^!=", |_| Token::NotEquals),
        (r"^==", |_| Token::DoubleEquals),
        (r"^\/=", |_| Token::CompoundDiv),
        (r"^\^=", |_| Token::CompoundXor),
        (r"^&=", |_| Token::CompoundAnd),
        (r"^\|=", |_| Token::CompoundOr),
        (r"^\*", |_| Token::Asterisk),
        (r"^-", |_| Token::Minus),
        (r"^\+", |_| Token::Plus),
        (r"^\/", |_| Token::Slash),
        (r"^=", |_| Token::Equals),
        (r"^>", |_| Token::GreaterThan),
        (r"^<", |_| Token::LessThan),
        (r"^!", |_| Token::LogicalNot),
        (r"^~", |_| Token::Tilde),
        (r"^\^", |_| Token::Circumflex),
        (r"^&", |_| Token::Ampersand),
        (r"^\|", |_| Token::Pipe),
        (r"^\{", |_| Token::OpenCurly),
        (r"^\}", |_| Token::CloseCurly),
        (r"^\(", |_| Token::OpenParen),
        (r"^\)", |_| Token::CloseParen),
        (r"^\[", |_| Token::OpenSquare),
        (r"^\]", |_| Token::CloseSquare),
        (r"^struct\b", |_| Token::Struct),
        (r"^int\b", |_| Token::Int),
        (r"^void\b", |_| Token::Void),
        (r"^if\b", |_| Token::If),
        (r"^while\b", |_| Token::While),
        (r"^for\b", |_| Token::For),
        (r"^fn\b", |_| Token::Function),
        (r"^nullptr\b", |_| Token::NullPtr),
        (r"^as\b", |_| Token::AsCast),
        (r#"^[_a-zA-Z][_a-zA-Z0-9]*"#, |slice| {
            Token::Ident(slice.to_owned())
        }),
        (r#"^0x[0-9a-zA-Z]*"#, |slice| {
            Token::HexLiteral(slice[2..].to_owned())
        }),
        (r#"^[0-9a-zA-Z]+"#, |slice| {
            Token::DecLiteral(slice.to_owned())
        }),
        (r"^;", |_| Token::Semicolon),
        (r"^,", |_| Token::Comma),
        (r"^\.", |_| Token::Period),
    ];

    static RE_COMPILED_DEFS: LazyLock<&[(Regex, LexConvertFunction)]> = LazyLock::new(|| {
        let boxed: Box<[(Regex, LexConvertFunction)]> = REGEX_DEFS
            .iter()
            .map(|(re, mapper)| (Regex::new(re).unwrap(), *mapper))
            .collect();
        Box::leak(boxed)
    });

    static WHITESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s+").unwrap());

    #[derive(Debug)]
    pub struct Lexer<'a> {
        input: &'a str,
        position: usize,
        next: Option<Spanned<Token>>,
    }

    impl<'a> Lexer<'a> {
        pub fn new(s: &'a str) -> Self {
            Self {
                input: s,
                position: 0,
                next: None,
            }
        }
        pub fn next(&mut self) -> Result<Spanned<Token>, LexingError> {
            if let Some(r) = self.next.take() {
                let span = r.span();
                self.position += (span.end - span.start);
                return Ok(r);
            }
            let next = self.get_next()?;
            let span = next.span();
            self.position += (span.end - span.start);
            Ok(next)
        }
        pub fn peek(&mut self) -> Result<Spanned<Token>, LexingError> {
            if let Some(spanned) = &self.next {
                return Ok(spanned.clone());
            }
            self.get_next()
        }
        fn get_next(&mut self) -> Result<Spanned<Token>, LexingError> {
            if let Some(m) = WHITESPACE.find(&self.input[self.position..]) {
                self.position += m.len();
            }
            if self.input[self.position..].is_empty() {
                return Err(LexingError::Eof);
            }
            for (re, mapping_fn) in RE_COMPILED_DEFS.iter() {
                if let Some(m) = re.find(&self.input[self.position..]) {
                    let token = mapping_fn(m.as_str());
                    let span = Span::new(self.position, self.position + m.len());
                    return Ok(Spanned(token, span));
                }
            }
            Err(LexingError::UnknownToken {
                position: self.position,
            })
        }
    }
}

pub mod parsing {
    use lexer_helper::NextExpect;

    use super::ast::*;
    use super::lexing::*;
    #[derive(Debug)]
    pub struct Parser<'s> {
        lexer: Lexer<'s>,
    }

    #[derive(Debug, Clone)]
    pub enum ParsingError {
        Lexing(LexingError),
        Eof(Expected),
        WrongToken {
            expected: Expected,
            found: Spanned<Token>,
        },
        InvalidInteger(Spanned<Token>),
    }

    #[derive(Debug, Clone)]
    pub enum Expected {
        Str(&'static str),
        Token(Token),
    }

    impl From<LexingError> for ParsingError {
        fn from(value: LexingError) -> Self {
            ParsingError::Lexing(value)
        }
    }

    // #[derive(Debug)]
    // struct Checkpoint<'s> {
    //     lexer: Lexer<'s>,
    // }

    impl<'s> Parser<'s> {
        pub fn new(s: &'s str) -> Self {
            Self {
                lexer: Lexer::new(s),
            }
        }
        // fn checkpoint(&self) -> Checkpoint<'s> {
        //     Checkpoint {
        //         lexer: self.lexer.clone(),
        //     }
        // }
        // fn restore(&mut self, checkpoint: Checkpoint<'s>) {
        //     self.lexer = checkpoint.lexer;
        // }
    }

    mod lexer_helper {
        use super::*;
        pub trait NextExpect {
            fn next_expect(&mut self, exp: Token) -> Result<Spanned<Token>, ParsingError>;
            fn next_expect_number(&mut self) -> Result<Spanned<i32>, ParsingError>;
            fn next_expect_ident(
                &mut self,
                usage: &'static str,
            ) -> Result<Spanned<Token>, ParsingError>;
            fn next_expect_non_eof(
                &mut self,
                exp: Expected,
            ) -> Result<Spanned<Token>, ParsingError>;
        }
        impl NextExpect for Lexer<'_> {
            fn next_expect(&mut self, expected: Token) -> Result<Spanned<Token>, ParsingError> {
                let token = self.next_expect_non_eof(Expected::Token(expected.clone()))?;
                if std::mem::discriminant(&expected) != std::mem::discriminant(&token.0) {
                    return Err(ParsingError::WrongToken {
                        expected: Expected::Token(expected),
                        found: token,
                    });
                }
                Ok(token)
            }

            fn next_expect_number(&mut self) -> Result<Spanned<i32>, ParsingError> {
                let token = self.next_expect_non_eof(Expected::Str("number"))?;
                match token {
                    Spanned(Token::DecLiteral(num), span) => match num.parse() {
                        Ok(n) => Ok(Spanned(n, span)),
                        Err(_) => Err(ParsingError::InvalidInteger(Spanned(
                            Token::DecLiteral(num),
                            span,
                        ))),
                    },
                    Spanned(Token::HexLiteral(num), span) => match u32::from_str_radix(&num, 16) {
                        Ok(n) => Ok(Spanned(n as i32, span)),
                        Err(_) => Err(ParsingError::InvalidInteger(Spanned(
                            Token::HexLiteral(num),
                            span,
                        ))),
                    },
                    other => Err(ParsingError::WrongToken {
                        expected: Expected::Str("number"),
                        found: other,
                    }),
                }
            }

            fn next_expect_ident(
                &mut self,
                usage: &'static str,
            ) -> Result<Spanned<Token>, ParsingError> {
                let token = self.next_expect_non_eof(Expected::Str("ident"))?;
                match token {
                    t @ Spanned(Token::Ident(_), _) => Ok(t),
                    other => Err(ParsingError::WrongToken {
                        expected: Expected::Str(usage),
                        found: other,
                    }),
                }
            }

            fn next_expect_non_eof(
                &mut self,
                exp: Expected,
            ) -> Result<Spanned<Token>, ParsingError> {
                self.next().map_err(|e| match e {
                    e @ LexingError::UnknownToken { position } => ParsingError::Lexing(e),
                    LexingError::Eof => ParsingError::Eof(exp),
                })
            }
        }
    }

    impl Parser<'_> {
        pub fn parse_program(&mut self) -> Result<Program, ParsingError> {
            todo!()
        }
        pub fn parse_toplevel(&mut self) -> Result<TopLevelDecl, ParsingError> {
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
            let name = self.lexer.next_expect_ident("variable name")?;
            // handle arrays
            // handle assignment sign (=)
            // handle expression
            // handle semicolon
            todo!()
        }
        pub fn parse_type(&mut self) -> Result<Spanned<Type>, ParsingError> {
            let res = self.lexer.next_expect_non_eof(Expected::Str("type"))?;
            let base = match res {
                Spanned(Token::Struct, span) => {
                    let struct_name = self.lexer.next_expect_ident("struct name")?;
                    match struct_name {
                        Spanned(Token::Ident(ident), span) => Spanned(Type::Struct(ident), span),
                        _ => unreachable!(),
                    }
                }
                Spanned(Token::Int, span) => Spanned(Type::Int, span),
                Spanned(Token::Void, span) => Spanned(Type::Void, span),
                Spanned(Token::Function, span) => todo!("need punctuated support"),
                other => {
                    return Err(ParsingError::WrongToken {
                        expected: Expected::Str("type"),
                        found: other,
                    })
                }
            };
            let mut curr = base;
            let start_base = curr.span().start;
            while let Ok(Spanned(Token::Asterisk, span)) = self.lexer.peek() {
                self.lexer.next();
                curr = Spanned(Type::Ptr(Box::new(curr)), Span::new(start_base, span.end));
            }
            Ok(curr)
        }
        pub fn parse_expr(&mut self) -> Result<Spanned<Expression>, ParsingError> {
            self.parse_expr_bp(0)
        }
        fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Spanned<Expression>, ParsingError> {
            let lhs = self
                .lexer
                .next_expect_non_eof(Expected::Str("expression"))?;
            let mut lhs = match lhs {
                Spanned(Token::DecLiteral(n), span) => match n.parse() {
                    Ok(n) => Spanned(Expression::IntLiteral(n), span),
                    Err(_) => {
                        return Err(ParsingError::InvalidInteger(Spanned(
                            Token::DecLiteral(n),
                            span,
                        )))
                    }
                },
                Spanned(Token::HexLiteral(n), span) => match u32::from_str_radix(&n, 16) {
                    Ok(n) => Spanned(Expression::IntLiteral(n as i32), span),
                    Err(_) => {
                        return Err(ParsingError::InvalidInteger(Spanned(
                            Token::HexLiteral(n),
                            span,
                        )))
                    }
                },
                Spanned(Token::Ident(ident), span) => Spanned(Expression::Ident(ident), span),
                Spanned(Token::NullPtr, span) => Spanned(Expression::NullPtr, span),
                Spanned(Token::OpenParen, span) => {
                    let expr = self.parse_expr_bp(0)?;
                    let start = span.start;
                    let end_token = self.lexer.next_expect(Token::CloseParen)?;
                    let end = end_token.span().end;
                    Spanned(
                        Expression::Parenthesized(expr.map(Box::new)),
                        Span::new(start, end),
                    )
                }

                token @ Spanned(
                    Token::Asterisk
                    | Token::Minus
                    | Token::Plus
                    | Token::OpenSquare
                    | Token::Tilde
                    | Token::Increment
                    | Token::LogicalNot
                    | Token::Ampersand
                    | Token::Decrement,
                    span,
                ) => {
                    let ((), rbp) = prefix_binding_power(&token);
                    let inner = self.parse_expr_bp(rbp)?;
                    let inner_end = inner.span().end;
                    let Spanned(token, span) = token;
                    #[rustfmt::skip]
                    let ret = Spanned(
                        match token {
                            Token::Asterisk => Expression::AsteriskDereference(inner.map(Box::new).with_end(inner_end)),
                            Token::Minus => Expression::UnaryNegation(inner.map(Box::new).with_end(inner_end)),
                            Token::Plus => Expression::UnaryPlus(inner.map(Box::new).with_end(inner_end)),
                            Token::Tilde => Expression::BitwiseNot(inner.map(Box::new).with_end(inner_end)),
                            Token::Increment => Expression::PreIncrement(inner.map(Box::new).with_end(inner_end)),
                            Token::Decrement => Expression::PreDecrement(inner.map(Box::new).with_end(inner_end)),
                            Token::LogicalNot => Expression::LogicalNot(inner.map(Box::new).with_end(inner_end)),
                            Token::Ampersand => Expression::AddrOf(inner.map(Box::new).with_end(inner_end)),
                            _ => unreachable!(),
                        },
                        span,
                    );
                    ret
                }
                other => {
                    return Err(ParsingError::WrongToken {
                        expected: Expected::Str("expression"),
                        found: other,
                    })
                }
            };
            while let Ok(op) = self.lexer.peek() {
                let op = match op {
                    Spanned(Token::Arrow, span) => {
                        self.lexer.next();
                        let field = self.lexer.next_expect_ident("struct member")?;
                        let lhs_start = lhs.span().start;
                        match field {
                            Spanned(Token::Ident(ident), span) => {
                                lhs = Spanned(
                                    Expression::ArrowDereference(
                                        lhs.map(Box::new),
                                        Spanned(ident, span),
                                    ),
                                    Span::new(lhs_start, span.end),
                                );
                                continue;
                            }
                            _ => unreachable!(),
                        }
                    }
                    Spanned(Token::Period, span) => {
                        self.lexer.next();
                        let field = self.lexer.next_expect_ident("struct member")?;
                        let lhs_start = lhs.span().start;
                        match field {
                            Spanned(Token::Ident(ident), span) => {
                                lhs = Spanned(
                                    Expression::MemberAccess(
                                        lhs.map(Box::new),
                                        Spanned(ident, span),
                                    ),
                                    Span::new(lhs_start, span.end),
                                );
                                continue;
                            }
                            _ => unreachable!(),
                        }
                    }
                    t @ Spanned(
                        Token::Asterisk
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
                        | Token::AsCast,
                        _span,
                    ) => t,
                    Spanned(Token::CloseParen | Token::CloseSquare | Token::Semicolon, _span) => {
                        break
                    }
                    t => {
                        return Err(ParsingError::WrongToken {
                            expected: Expected::Str("operator or end of expression"),
                            found: t,
                        })
                    }
                };
                if let Some((l_bp, ())) = postfix_binding_power(&op) {
                    if l_bp < min_bp {
                        break;
                    }
                    self.lexer.next();
                    let lhs_start = lhs.span().start;
                    match op {
                        // Function call
                        Spanned(Token::OpenParen, span) => {
                            let exprs = self.parse_args()?;
                            let closing = self.lexer.next_expect(Token::CloseParen)?;
                            let closing_end = closing.span().end;
                            lhs = Spanned(
                                Expression::FunctionCall(
                                    lhs.map(Box::new),
                                    Spanned(exprs, Span::new(span.start, closing_end)),
                                ),
                                Span::new(lhs_start, closing_end),
                            );
                        }
                        // Array indexing
                        Spanned(Token::OpenSquare, span) => {
                            let expr = self.parse_expr_bp(0)?;
                            let closing = self.lexer.next_expect(Token::CloseSquare)?;
                            let closing_end = closing.span().end;
                            lhs = Spanned(
                                Expression::ArrayIndex(lhs.map(Box::new), expr.map(Box::new)),
                                Span::new(lhs_start, closing_end),
                            );
                        }
                        Spanned(Token::Increment, span) => {
                            lhs = Spanned(
                                Expression::PostIncrement(lhs.map(Box::new)),
                                span.with_start(lhs_start),
                            )
                        }
                        Spanned(Token::Decrement, span) => {
                            lhs = Spanned(
                                // Expression::PostDecrement(lhs.map(Box::new)),
                                // TESTING
                                Expression::PostIncrement(lhs.map(Box::new)),
                                span.with_start(lhs_start),
                            )
                        }
                        Spanned(Token::AsCast, _span) => {
                            let typ = self.parse_type()?;
                            let span_end = typ.span().end;
                            lhs = Spanned(
                                Expression::Cast(lhs.map(Box::new), typ),
                                Span::new(lhs_start, span_end),
                            );
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
                    let lhs_start = lhs.span().start;
                    let rhs = self.parse_expr_bp(rbp)?;
                    let rhs_end = rhs.span().end;
                    let expr_span = Span::new(lhs_start, rhs_end);
                    let Spanned(op_token, op_span) = op;
                    let infix_expr = match op_token {
                        Token::DoubleEquals => {
                            Expression::Equals((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::Asterisk => {
                            Expression::Mul((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::Minus => Expression::Sub((lhs).map(Box::new), (rhs).map(Box::new)),
                        Token::Plus => Expression::Add((lhs).map(Box::new), (rhs).map(Box::new)),
                        Token::Slash => Expression::Div((lhs).map(Box::new), (rhs).map(Box::new)),
                        Token::GreaterThanEquals => Expression::GreaterThanOrEquals(
                            (lhs).map(Box::new),
                            (rhs).map(Box::new),
                        ),

                        Token::LessThanEquals => {
                            Expression::LessThanOrEquals((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::GreaterThan => {
                            Expression::GreaterThan((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::LessThan => {
                            Expression::LessThan((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::NotEquals => {
                            Expression::NotEquals((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::Circumflex => {
                            Expression::BitwiseXor((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::Ampersand => {
                            Expression::BitwiseAnd((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::Pipe => {
                            Expression::BitwiseOr((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::Equals => {
                            Expression::Assignment((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::Shl => {
                            Expression::ShiftLeft((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::Shr => {
                            Expression::ShiftRight((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::CompoundAdd => {
                            Expression::CompoundAdd((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::CompoundSub => {
                            Expression::CompoundSub((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::CompoundMul => {
                            Expression::CompoundMul((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::CompoundDiv => {
                            Expression::CompoundDiv((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::CompoundAnd => {
                            Expression::CompoundAnd((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::CompoundOr => {
                            Expression::CompoundOr((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        Token::CompoundXor => {
                            Expression::CompoundXor((lhs).map(Box::new), (rhs).map(Box::new))
                        }
                        _ => unreachable!(),
                    };
                    lhs = Spanned(infix_expr, expr_span);
                    continue;
                }
                break;
            }
            Ok(lhs)
        }

        pub fn parse_args(&mut self) -> Result<Vec<Spanned<Expression>>, ParsingError> {
            let next = self.lexer.peek()?;
            if let Token::CloseParen = *next {
                // empty function call
                return Ok(Vec::new());
            }
            let mut exprs = Vec::new();
            let first_expr = self.parse_expr()?;
            exprs.push(first_expr);
            loop {
                let Spanned(token, span) = self.lexer.peek()?;
                if let Token::CloseParen = token {
                    // End of function arguments
                    return Ok(exprs);
                } else if let Token::Comma = token {
                    // comma, so there's more expressions.
                    self.lexer.next();
                } else {
                    // something else that's not an expression. abort!
                    self.lexer.next();
                    return Err(ParsingError::WrongToken {
                        expected: Expected::Token(Token::CloseParen),
                        found: Spanned(token, span),
                    });
                }
                // Since we ran into a comma, we parse. No trailing commas for now :(
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
            | Token::Ampersand
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

        macro_rules! expr_parse_pass {
            ($s:expr, $expected:expr $(,)?) => {
                let mut parser = Parser::new($s);
                let parsed = parser.parse_expr().unwrap();
                assert!(matches!(parser.lexer.next(), Err(LexingError::Eof { .. })));
                assert_eq!(format!("{parsed:?}"), $expected);
            };
        }

        macro_rules! expr_parse_fail {
            ($s:expr) => {
                let mut parser = Parser::new($s);
                assert!(parser.parse_expr().is_err());
            };
        }

        macro_rules! test_prefix {
            ($test_name:ident, $op:literal) => {
                #[test]
                fn $test_name() {
                    expr_parse_pass!(concat!($op, "x"), concat!("(", $op, "x)"));
                    expr_parse_pass!(concat!($op, "y"), concat!("(", $op, "y)"));
                    expr_parse_pass!(
                        concat!($op, "             100  \n"),
                        concat!("(", $op, "100)")
                    );
                }
            };
        }

        macro_rules! test_infix {
            ($test_name:ident, $op:literal) => {
                #[test]
                fn $test_name() {
                    expr_parse_pass!(concat!("x", $op, "x"), concat!("(x ", $op, " x)"),);
                    expr_parse_pass!(
                        concat!("  x     ", $op, "   x    "),
                        concat!("(x ", $op, " x)"),
                    );
                    expr_parse_pass!(concat!("x", $op, "5"), concat!("(x ", $op, " 5)"),);
                    expr_parse_pass!(concat!("50", $op, "5"), concat!("(50 ", $op, " 5)"),);
                }
            };
        }
        #[test]
        fn ints() {
            expr_parse_pass!("5", "5");
            expr_parse_pass!("15000", "15000");
            expr_parse_pass!("-2500500", "(-2500500)",);
            expr_parse_pass!("0x55125", "348453");
            expr_parse_pass!("2147483647", "2147483647");
            expr_parse_fail!("2147483648");
            expr_parse_fail!("");
            expr_parse_pass!("-0x1", "(-1)");
            expr_parse_pass!("0xffffffff", "-1");
        }

        #[test]
        fn idents() {
            expr_parse_pass!("_", "_");
            expr_parse_pass!("____________", "____________");
            expr_parse_pass!("_something", "_something");
            expr_parse_pass!("var_name", "var_name");
            expr_parse_pass!("var", "var");
            expr_parse_fail!("int");
            expr_parse_fail!("struct");
            expr_parse_fail!("void");
            expr_parse_fail!("fn");
            expr_parse_pass!(
                "some_really_long_variable_name_idk_this_is_probably_long_enough",
                "some_really_long_variable_name_idk_this_is_probably_long_enough",
            );
            expr_parse_pass!("u31", "u31");
            expr_parse_fail!("3ab");
        }

        test_prefix!(logicalnot, "!");
        test_prefix!(addrof, "&");
        test_prefix!(unarynegation, "-");
        test_prefix!(unaryplus, "+");
        test_prefix!(preincrement, "++");
        test_prefix!(predecrement, "--");
        test_prefix!(asteriskdereference, "*");
        test_prefix!(bitwisenot, "~");

        test_infix!(equals, "==");
        test_infix!(notequals, "!=");
        test_infix!(greaterthanorequals, ">=");
        test_infix!(lessthanorequals, "<=");
        test_infix!(greaterthan, ">");
        test_infix!(lessthan, "<");
        test_infix!(add, "+");
        test_infix!(sub, "-");
        test_infix!(mul, "*");
        test_infix!(div, "/");
        test_infix!(bitwiseand, "&");
        test_infix!(bitwiseor, "|");
        test_infix!(bitwisexor, "^");
        test_infix!(shiftleft, "<<");
        test_infix!(shiftright, ">>");
        test_infix!(assignment, "=");
        test_infix!(compoundadd, "+=");
        test_infix!(compoundsub, "-=");
        test_infix!(compoundmul, "*=");
        test_infix!(compounddiv, "/=");
        test_infix!(compoundand, "&=");
        test_infix!(compoundor, "|=");
        test_infix!(compoundxor, "^=");

        /*
        PostIncrement(Box<Expression>),
        PostDecrement(Box<Expression>),
        ArrowDereference(Box<Expression>, String),
        Cast(Box<Expression>, Type),
        // Parenthesized(Box<Expression>),
        FunctionCall(Box<Expression>, Vec<Expression>),
        StructInstantiation(Vec<Expression>),
        NullPtr,
        MemberAccess(Box<Expression>, String),
        ArrayIndex(Box<Expression>, Box<Expression>),
        */
    }
}

mod ast {
    pub use super::span::*;
    #[derive(PartialEq, Clone)]
    pub enum Type {
        Void,
        Int,
        Struct(String),
        Ptr(Box<Spanned<Type>>),
        Fn {
            args: Spanned<Vec<Type>>,
            ret: Box<Spanned<Type>>,
        },
    }

    #[derive(Clone)]
    pub struct Program {
        pub top_level: Vec<TopLevelDecl>,
    }

    #[derive(Clone)]
    pub enum TopLevelDecl {
        Function(FunctionDecl),
        Variable(VariableDecl),
        Struct(StructDecl),
    }

    #[derive(Clone)]
    pub struct StructDecl {
        pub name: String,
        pub members: Vec<NameAndType>,
    }

    #[derive(Clone)]
    pub struct NameAndType {
        pub name: String,
        pub typ: Type,
    }

    #[derive(Clone)]
    pub struct FunctionDecl {
        pub return_type: Type,
        pub name: String,
        pub parameters: Vec<NameAndType>,
        pub block: Block,
    }

    #[derive(Clone)]
    pub struct VariableDecl {
        pub typ: Type,
        pub name: String,
        pub assigned: Option<Expression>,
    }

    #[derive(Clone)]
    pub enum Statement {
        Expression(Expression),
        Declaration(VariableDecl),
        Block(Block),
        If(If),
        While(While),
        For(For),
        // Parenthesized(Box<Statement>),
        // TODO LVALUE ALL
        // TODO goto?
    }

    type SpannedBox<T> = Spanned<Box<T>>;

    #[derive(Clone, PartialEq)]
    pub enum Expression {
        LogicalNot(SpannedBox<Expression>),
        Equals(SpannedBox<Expression>, SpannedBox<Expression>),
        NotEquals(SpannedBox<Expression>, SpannedBox<Expression>),
        GreaterThanOrEquals(SpannedBox<Expression>, SpannedBox<Expression>),
        LessThanOrEquals(SpannedBox<Expression>, SpannedBox<Expression>),
        GreaterThan(SpannedBox<Expression>, SpannedBox<Expression>),
        LessThan(SpannedBox<Expression>, SpannedBox<Expression>),
        ArrayIndex(SpannedBox<Expression>, SpannedBox<Expression>),
        AsteriskDereference(SpannedBox<Expression>),
        ArrowDereference(SpannedBox<Expression>, Spanned<String>),
        IntLiteral(i32),
        Ident(String),
        Cast(SpannedBox<Expression>, Spanned<Type>),
        // Parenthesized(SpannedBox<Expression>),
        Add(SpannedBox<Expression>, SpannedBox<Expression>),
        Sub(SpannedBox<Expression>, SpannedBox<Expression>),
        Mul(SpannedBox<Expression>, SpannedBox<Expression>),
        Div(SpannedBox<Expression>, SpannedBox<Expression>),
        BitwiseAnd(SpannedBox<Expression>, SpannedBox<Expression>),
        BitwiseOr(SpannedBox<Expression>, SpannedBox<Expression>),
        BitwiseXor(SpannedBox<Expression>, SpannedBox<Expression>),
        BitwiseNot(SpannedBox<Expression>),
        ShiftLeft(SpannedBox<Expression>, SpannedBox<Expression>),
        ShiftRight(SpannedBox<Expression>, SpannedBox<Expression>),
        FunctionCall(SpannedBox<Expression>, Spanned<Vec<Spanned<Expression>>>),
        StructInstantiation(Spanned<Vec<Spanned<Expression>>>),
        NullPtr,
        MemberAccess(SpannedBox<Expression>, Spanned<String>),
        UnaryNegation(SpannedBox<Expression>),
        UnaryPlus(SpannedBox<Expression>),
        PreIncrement(SpannedBox<Expression>),
        PreDecrement(SpannedBox<Expression>),
        PostIncrement(SpannedBox<Expression>),
        PostDecrement(SpannedBox<Expression>),
        Assignment(SpannedBox<Expression>, SpannedBox<Expression>),
        AddrOf(SpannedBox<Expression>),
        CompoundAdd(SpannedBox<Expression>, SpannedBox<Expression>),
        CompoundSub(SpannedBox<Expression>, SpannedBox<Expression>),
        CompoundMul(SpannedBox<Expression>, SpannedBox<Expression>),
        CompoundDiv(SpannedBox<Expression>, SpannedBox<Expression>),
        CompoundAnd(SpannedBox<Expression>, SpannedBox<Expression>),
        CompoundOr(SpannedBox<Expression>, SpannedBox<Expression>),
        CompoundXor(SpannedBox<Expression>, SpannedBox<Expression>),
        Parenthesized(SpannedBox<Expression>),
    }

    #[derive(Clone)]
    pub struct If {
        pub condition: Expression,
        pub block: Block,
    }
    #[derive(Clone)]
    pub struct While {
        pub condition: Expression,
        pub block: Block,
    }
    #[derive(Clone)]
    pub struct For {
        pub init: Box<Statement>,
        pub condition: Expression,
        pub after: Box<Statement>,
        pub block: Block,
    }

    #[derive(Clone)]
    pub struct Block {
        pub statements: Vec<Statement>,
    }

    // #[derive(, Clone)]
    // pub enum LValue {
    //     Ident(String),
    //     Dereference(Expression),
    //     ArrayIndex { ptr: String, index: Box<Expression> },
    //     StructureMember { struct_: String, member: String },
    //     ArrowDereference { struct_: String, member: String },
    //     Parenthesized(Box<LValue>),
    // }
}

mod ast_sfmt {
    use super::ast::*;
    // we abuse LowerExp to print an S-expression.
    impl std::fmt::Debug for Type {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            match self {
                Type::Void => write!(f, "void"),
                Type::Int => write!(f, "int"),
                Type::Struct(struct_name) => write!(f, "struct {struct_name}"),
                Type::Ptr(spanned) => {
                    write!(f, "{spanned:?}*")
                }
                Type::Fn { args, ret } => {
                    let mut helper = f.debug_tuple("fn");
                    for typ in args.inner() {
                        helper.field(typ);
                    }
                    helper.finish()?;
                    write!(f, " -> {ret:?}")
                }
            }
        }
    }
    impl std::fmt::Debug for Program {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            for tl in &self.top_level {
                write!(f, "{tl:?}")?;
            }
            Ok(())
        }
    }
    impl std::fmt::Debug for TopLevelDecl {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            match self {
                TopLevelDecl::Function(function_decl) => write!(f, "{function_decl:?}"),
                TopLevelDecl::Variable(variable_decl) => write!(f, "{variable_decl:?}"),
                TopLevelDecl::Struct(struct_decl) => write!(f, "{struct_decl:?}"),
            }
        }
    }
    impl std::fmt::Debug for StructDecl {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            let mut helper = f.debug_struct(&self.name);
            for field in &self.members {
                helper.field(&field.name, &field.typ);
            }
            helper.finish()?;
            Ok(())
        }
    }
    impl std::fmt::Debug for NameAndType {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(f, "{} {:?}", self.name, self.typ)
        }
    }
    impl std::fmt::Debug for FunctionDecl {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(f, "{:?} ", self.return_type)?;
            let mut helper = f.debug_tuple(&self.name);
            for paramlist in &self.parameters {
                helper.field(paramlist);
            }
            helper.finish()?;
            write!(f, " ")?;
            write!(f, "{:?}", self.block)?;
            Ok(())
        }
    }
    impl std::fmt::Debug for VariableDecl {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            todo!()
        }
    }
    impl std::fmt::Debug for Statement {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            match self {
                Statement::Expression(expression) => write!(f, "{expression:?}"),
                Statement::Declaration(variable_decl) => write!(f, "{variable_decl:?}"),
                Statement::Block(block) => write!(f, "{block:?}"),
                Statement::If(if_) => write!(f, "{if_:?}"),
                Statement::While(while_) => write!(f, "{while_:?}"),
                Statement::For(for_) => write!(f, "{for_:?}"),
                // Statement::Parenthesized(statement) => write!(f, "[<{statement:?}>]"),
            }
        }
    }
    impl std::fmt::Debug for Expression {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            match self {
                Expression::LogicalNot(inner) => write!(f, "(!{inner:?})"),
                Expression::Equals(lhs, rhs) => write!(f, "({lhs:?} == {rhs:?})"),
                Expression::NotEquals(lhs, rhs) => write!(f, "({lhs:?} != {rhs:?})"),
                Expression::GreaterThanOrEquals(lhs, rhs) => write!(f, "({lhs:?} >= {rhs:?})"),
                Expression::LessThanOrEquals(lhs, rhs) => write!(f, "({lhs:?} <= {rhs:?})"),
                Expression::GreaterThan(lhs, rhs) => write!(f, "({lhs:?} > {rhs:?})"),
                Expression::LessThan(lhs, rhs) => write!(f, "({lhs:?} < {rhs:?})"),
                Expression::ArrayIndex(lhs, rhs) => write!(f, "({lhs:?}[{rhs:?}])"),
                Expression::AsteriskDereference(inner) => write!(f, "(*{inner:?})"),
                Expression::ArrowDereference(lhs, rhs) => write!(f, "({lhs:?}->{rhs:?})"),
                Expression::IntLiteral(n) => write!(f, "{n}"),
                Expression::Ident(n) => write!(f, "{n}"),
                Expression::Cast(lhs, rhs) => write!(f, "({lhs:?} as {rhs:?})"),
                Expression::Add(lhs, rhs) => write!(f, "({lhs:?} + {rhs:?})"),
                Expression::Sub(lhs, rhs) => write!(f, "({lhs:?} - {rhs:?})"),
                Expression::Mul(lhs, rhs) => write!(f, "({lhs:?} * {rhs:?})"),
                Expression::Div(lhs, rhs) => write!(f, "({lhs:?} / {rhs:?})"),
                Expression::BitwiseAnd(lhs, rhs) => write!(f, "({lhs:?} & {rhs:?})"),
                Expression::BitwiseOr(lhs, rhs) => write!(f, "({lhs:?} | {rhs:?})"),
                Expression::BitwiseXor(lhs, rhs) => write!(f, "({lhs:?} ^ {rhs:?})"),
                Expression::BitwiseNot(inner) => write!(f, "(~{inner:?})"),
                Expression::ShiftLeft(lhs, rhs) => write!(f, "({lhs:?} << {rhs:?})"),
                Expression::ShiftRight(lhs, rhs) => write!(f, "({lhs:?} >> {rhs:?})"),
                Expression::FunctionCall(func, args) => {
                    write!(f, "{func:?}")?;
                    let mut tuple = f.debug_tuple("");
                    for arg in args.inner() {
                        tuple.field(args);
                    }
                    tuple.finish()?;
                    Ok(())
                }
                Expression::StructInstantiation(spanned) => {
                    let mut s = f.debug_tuple("");
                    s.field(&"struct");
                    for val in spanned.inner() {
                        s.field(val);
                    }
                    s.finish()?;
                    Ok(())
                }
                Expression::NullPtr => write!(f, "nullptr"),
                Expression::MemberAccess(lhs, rhs) => write!(f, "({lhs:?}.{})", rhs.inner()),
                Expression::UnaryNegation(inner) => write!(f, "(-{inner:?})"),
                Expression::UnaryPlus(inner) => write!(f, "(+{inner:?})"),
                Expression::PreIncrement(inner) => write!(f, "(++{inner:?})"),
                Expression::PreDecrement(inner) => write!(f, "(--{inner:?})"),
                Expression::PostIncrement(inner) => write!(f, "({inner:?}++)"),
                Expression::PostDecrement(inner) => write!(f, "({inner:?}--)"),
                Expression::Assignment(lhs, rhs) => write!(f, "({lhs:?} = {rhs:?})"),
                Expression::AddrOf(inner) => write!(f, "(&{inner:?})"),
                Expression::CompoundAdd(lhs, rhs) => write!(f, "({lhs:?} += {rhs:?})"),
                Expression::CompoundSub(lhs, rhs) => write!(f, "({lhs:?} -= {rhs:?})"),
                Expression::CompoundMul(lhs, rhs) => write!(f, "({lhs:?} *= {rhs:?})"),
                Expression::CompoundDiv(lhs, rhs) => write!(f, "({lhs:?} /= {rhs:?})"),
                Expression::CompoundAnd(lhs, rhs) => write!(f, "({lhs:?} &= {rhs:?})"),
                Expression::CompoundOr(lhs, rhs) => write!(f, "({lhs:?} |= {rhs:?})"),
                Expression::CompoundXor(lhs, rhs) => write!(f, "({lhs:?} ^= {rhs:?})"),
                Expression::Parenthesized(inner) => write!(f, "({inner:?})"),
            }
        }
    }
    impl std::fmt::Debug for If {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(f, "if ({:?}) {:?}", self.condition, self.block)
        }
    }
    impl std::fmt::Debug for While {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(f, "while ({:?}) {:?}", self.condition, self.block)
        }
    }
    impl std::fmt::Debug for For {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(
                f,
                "for ({:?}; {:?}; {:?}) {:?}",
                self.init, self.condition, self.after, self.block
            )
        }
    }
    impl std::fmt::Debug for Block {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            let mut helper = f.debug_struct("");
            for (i, stmt) in self.statements.iter().enumerate() {
                helper.field(&i.to_string(), stmt);
            }
            helper.finish()?;
            Ok(())
        }
    }
}
