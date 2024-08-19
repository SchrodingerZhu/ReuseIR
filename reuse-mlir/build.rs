fn main() {
    let dst = cmake::Config::new(".")
        .generator("Ninja")
        .build()
        .join("build")
        .join("lib");
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=MLIRReuseIR");
}
