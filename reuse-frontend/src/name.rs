use std::fmt::{Display, Formatter};

type ID = u64;

pub struct Name {
    id: ID,
    raw: String,
}

impl Display for Name {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.raw, self.id)
    }
}

#[allow(dead_code)]
#[derive(Default)]
struct IDs(ID);

impl IDs {
    #[allow(dead_code)]
    fn fresh(&mut self, raw: &str) -> Name {
        self.0 += 1;
        Name {
            id: self.0,
            raw: raw.to_string(),
        }
    }
}
