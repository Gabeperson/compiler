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
    pub fn with_start(self, start: usize) -> Self {
        Self {
            start,
            end: self.end,
        }
    }
    pub fn with_end(self, end: usize) -> Self {
        Self {
            start: self.start,
            end,
        }
    }
}
#[derive(Clone, Copy)]
pub struct Spanned<T>(pub T, pub Span);
impl<T: std::fmt::Debug> std::fmt::Debug for Spanned<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.inner())
    }
}
impl<T> Spanned<T> {
    pub fn into_inner(self) -> T {
        self.0
    }
    pub fn map<U>(self, f: impl Fn(T) -> U) -> Spanned<U> {
        Spanned(f(self.0), self.1)
    }
    pub fn span(&self) -> Span {
        self.1
    }
    pub fn with_end(self, end: usize) -> Self {
        Self(self.0, Span::new(self.1.start, end))
    }
    pub fn with_start(self, start: usize) -> Self {
        Self(self.0, Span::new(start, self.1.end))
    }
    pub fn inner(&self) -> &T {
        &self.0
    }
}
impl<T: PartialEq> PartialEq for Spanned<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}
impl<T> Deref for Spanned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
// impl<T> DerefMut for Spanned<T> {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.0
//     }
// }
