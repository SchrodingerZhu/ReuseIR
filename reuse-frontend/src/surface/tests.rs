use chumsky::Parser;

use crate::surface::Surface;

#[test]
fn it_parses_decl() {
    Surface::default()
        .decl()
        .parse("def foo () -> None : None")
        .unwrap();
}
