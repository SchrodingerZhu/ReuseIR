fn main() {
    let dst = cmake::Config::new(".").target("ReuseIR").build();
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=ReuseIR");
}
