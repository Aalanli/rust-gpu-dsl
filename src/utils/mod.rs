
pub enum Doc {
    Indent(Box<Doc>),
    Lines(Vec<Doc>),
    Token(String),
}

impl Doc {
    pub fn token(s: impl Into<String>) -> Self {
        Doc::Token(s.into())
    }

    pub fn indent(self) -> Self {
        Doc::Indent(Box::new(self))
    }

    pub fn append(self, other: Self) -> Self {
        match self {
            Doc::Lines(mut v) => {
                v.push(other);
                Doc::Lines(v)
            }
            _ => Doc::Lines(vec![self, other]),
        }
    }

    pub fn append_token(self, other: impl Into<String>) -> Self {
        self.append(Doc::token(other))
    }

    pub fn len(&self) -> usize {
        match self {
            Doc::Indent(d) => d.len(),
            Doc::Lines(v) => v.iter().map(|d| d.len()).sum(),
            Doc::Token(t) => t.len(),
        }
    }

    pub fn into_string(self, indent: usize) -> String {
        let mut s = String::new();

        fn into_string_helper(indent_level: usize, doc: Doc, s: &mut String, indent: usize) {
            match doc {
                Doc::Indent(d) => into_string_helper(indent_level + 1, *d, s, indent),
                Doc::Lines(v) => {
                    for d in v {
                        into_string_helper(indent_level, d, s, indent);
                    }
                }
                Doc::Token(t) => {
                    for _ in 0..indent_level * indent {
                        s.push(' ');
                    }
                    s.push_str(&t);
                    s.push('\n');
                }
            }
        }
        into_string_helper(0, self, &mut s, indent);
        s
    }
}