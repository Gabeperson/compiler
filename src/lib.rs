mod parsing_ {
    // use std::sync::LazyLock;
    // macro_rules! regex {
    //     ($name:ident, $lit:literal) => {
    //         static $name: LazyLock<regex::Regex> =
    //             LazyLock::new(|| regex::Regex::new($lit).unwrap());
    //     };
    // }

    // #[derive(Clone, Debug)]
    // enum Token {
    //     Struct,
    //     Int,
    //     Void,
    //     Asterisk,
    //     Minus,
    //     Plus,
    //     Slash,
    //     OpenCurly,
    //     CloseCurly,
    //     OpenParen,
    //     CloseParen,
    //     OpenSquare,
    //     CloseSquare,
    //     NumberLiteral(String),
    //     Ident(String),
    //     Semicolon,
    //     Equals,
    //     GreaterThanEquals,
    //     LessThanEquals,
    //     GreaterThan,
    //     LessThan,
    //     NotEquals,
    //     Tilde,
    //     Circumflex,
    //     Ampersand,
    //     Pipe,
    //     If,
    //     While,
    //     For,
    // }
    use super::span::*;
    // regex!(RE_STRUCT, "^struct");
    // regex!(RE_INT, "^int");
    // regex!(RE_VOID, "^void");
    // regex!(RE_ASTERISK, r"^\*");
    // regex!(RE_MINUS, "^-");
    // regex!(RE_PLUS, r"^\+");
    // regex!(RE_SLASH, "^/");
    // regex!(RE_OPENCURLY, r"^\{");
    // regex!(RE_CLOSECURLY, "^}");
    // regex!(RE_OPENPAREN, r"^\(");
    // regex!(RE_CLOSEPAREN, r"^\)");
    // regex!(RE_OPENSQUARE, r"^\[");
    // regex!(RE_CLOSESQUARE, "^]");
    // regex!(RE_NUMBERLITERAL, r"^0x[\da-fA-F]+|\d+");
    // regex!(RE_IDENT, r"^[_a-zA-Z][\d_a-zA-Z]*");
    // regex!(RE_ARROW, "^->");
    // regex!(RE_SEMICOLON, "^;");
    // regex!(RE_DOUBLE_EQUALS, "^==");
    // regex!(RE_EQUALS, "^=");
    // regex!(RE_GTE, "^>=");
    // regex!(RE_LTE, "^<=");
    // regex!(RE_GT, "^>");
    // regex!(RE_LT, "^<");
    // regex!(RE_NOT_EQUALS, "^!=");
    // regex!(RE_TILDE, "^~");
    // regex!(RE_CIRCUMFLEX, r"^\^");
    // regex!(RE_AMPERSAND, "^&");
    // regex!(RE_PIPE, r"^\|");
    // regex!(RE_INCR, r"\+\+");
    // regex!(RE_DECR, "--");
    // regex!(RE_RETURN, "return");
    // regex!(RE_BREAK, "break");
    // regex!(RE_CONTINUE, "continue");
    // #[derive(Clone, Debug)]
    // struct Lexer<'s> {
    //     input: &'s str,
    //     pos: usize,
    // }
    // #[derive(Clone, Debug)]
    // enum LexingError<'s> {
    //     UnknownCharacter(&'s str),
    // }
    // impl<'s> Lexer<'s> {
    //     fn new(s: &'s str) -> Self {
    //         Self { input: s, pos: 0 }
    //     }
    //     fn next(&mut self) -> Option<Result<Token, LexingError<'s>>> {
    //         let input = &self.input[self.pos..];
    //         if let Some(s) = RE_STRUCT.
    //         todo!()
    //     }
    // }
    // #[derive(Clone, Debug)]
    // struct Parser<'s> {
    //     input: &'s str,
    // }

    // struct Checkpoint<'s>(&'s str);

    // impl<'s> Parser<'s> {
    //     fn new(s: &'s str) -> Self {
    //         Self { input: s }
    //     }
    //     fn advance(&mut self, n: usize) {
    //         self.input = &self.input[n..]
    //     }
    //     fn checkpoint(&self) -> Checkpoint {
    //         Checkpoint(self.input)
    //     }
    //     fn restore(&'s mut self, checkpoint: Checkpoint<'s>) {
    //         self.input = checkpoint.0;
    //     }
    // }

    // #[derive(Clone, Debug)]
    // enum ParseError {
    //     UnknownToken(String),
    //     UnexpectedEndOfOutput,
    // }

    // type Result<T> = std::result::Result<T, ParseError>;

    // impl Parser<'_> {
    //     fn consume_whitespace(&mut self) {
    //         self.input = self.input.trim_start()
    //     }
    //     fn err_on_empty(&self) -> Result<()> {
    //         if self.input.is_empty() {
    //             return Err(ParseError::UnexpectedEndOfOutput);
    //         }
    //         Ok(())
    //     }
    //     pub fn parse_program(&mut self) -> Result<Program> {
    //         let mut top_level = Vec::new();
    //         self.consume_whitespace();
    //         while !self.input.is_empty() {
    //             top_level.push(self.parse_top_level()?);
    //             self.consume_whitespace();
    //         }
    //         Ok(Program { top_level })
    //     }

    //     fn parse_top_level(&mut self) -> Result<TopLevel> {
    //         // can't be eof because it is checked in parse_program

    //         todo!()
    //     }

    //     fn parse_type(&mut self) -> Result<Type> {
    //         if self.input.starts_with()
    //     }
    // }
    // use super::ast::*;
}

mod parsing {
    use super::ast::*;
    use winnow::{
        ascii::alphanumeric1,
        combinator::{alt, repeat},
        error::{ContextError, ParseError, StrContext},
        prelude::*,
        stream::AsChar,
        token::one_of,
    };

    type Error = ContextError<StrContext>;

    pub fn parse_program(s: &str) -> Result<Program, ParseError<&str, Error>> {
        let top_level: Vec<TopLevel> = repeat(0.., top_level).parse(s)?;
        Ok(Program { top_level })
    }

    pub fn top_level(s: &mut &str) -> PResult<TopLevel, Error> {
        todo!()
    }

    pub fn typ(s: &mut &str) -> PResult<Type, Error> {
        alt(());
        todo!()
    }

    pub fn ident(s: &mut &str) -> PResult<String, Error> {
        (
            alt(("_", one_of('a'..='Z').take())),
            repeat(0.., ("_", alphanumeric1)).map(|()| ()),
        )
            .take()
            .map(|s: &str| s.to_string())
            .parse_next(s)
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
        Assignment { from: AssignTo, to: Box<Expression> },
        If(If),
        While(While),
        For(For),
        Parenthesized(Box<Statement>),
        CompoundAdd(AssignTo, Box<Expression>),
        CompoundSub(AssignTo, Box<Expression>),
        CompoundMul(AssignTo, Box<Expression>),
        CompoundDiv(AssignTo, Box<Expression>),
        CompoundBitwiseAnd(AssignTo, Box<Expression>),
        CompoundBitwiseOr(AssignTo, Box<Expression>),
        CompoundBitwiseXor(AssignTo, Box<Expression>),
        Block(Block), // TODO
                      // goto?
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
        FunctionCall(String, Vec<Expression>),
        StructInstantiation(Vec<Expression>),
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

    #[derive(Debug, Clone)]
    pub enum AssignTo {
        Ident(String),
        Dereference(Expression),
        ArrayIndex { ptr: String, index: Box<Expression> },
        StructureMember { struct_: String, member: String },
        StructureMemberDereference { struct_: String, member: String },
        Parenthesized(Box<AssignTo>),
    }
}

// mod assembling {
//     #[derive(Clone, Debug)]
//     struct Assembler {
//         code: String,
//     }
// }

mod span {
    use std::ops::{Deref, DerefMut};

    #[derive(Debug, Clone, Copy)]
    pub struct Span {
        pub start: usize,
        pub end: usize,
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

// fn main() {
//     println!("Hello, world!");
// }
