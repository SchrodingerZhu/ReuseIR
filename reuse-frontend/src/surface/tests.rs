use chumsky::Parser;

use crate::print_parse_errors;
use crate::surface::Surface;

#[test]
fn it_parses_file() {
    const SRC: &str = "

def f0 (  ) -> None : None
def f1 (
) -> None : None

def f2 [
    T,
    U,
] (s: str) -> str :
\"hello, world\"

data
  Foo [T]:
    A
B(str)
    C(
        s  : str  ,
        b: bool  ,
    )

";
    Surface::default()
        .file()
        .parse(SRC)
        .into_result()
        .inspect_err(|es| print_parse_errors(es, SRC))
        .unwrap();
}
