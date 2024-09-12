mod mlir {
    use std::{env, error::Error, fs, io, path::Path, process::Command, str};

    pub fn run() -> Result<(), Box<dyn Error>> {
        println!("cargo:rerun-if-changed=include/ReuseIR/CAPI.h");
        let libdir = llvm_config("--libdir")?;
        println!("cargo:rustc-link-search={}", llvm_config("--libdir")?);
        println!("cargo:rustc-link-lib=MLIR");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libdir);
        for name in fs::read_dir(libdir)?
            .map(|entry| {
                Ok(if let Some(name) = entry?.path().file_name() {
                    name.to_str().map(String::from)
                } else {
                    None
                })
            })
            .collect::<Result<Vec<_>, io::Error>>()?
            .into_iter()
            .flatten()
        {
            if name.starts_with("libMLIRCAPI") && name.ends_with(".a") {
                if let Some(name) = trim_library_name(&name) {
                    println!("cargo:rustc-link-lib=static={}", name);
                }
            }
        }

        for name in llvm_config("--libnames")?.trim().split(' ') {
            if let Some(name) = trim_library_name(name) {
                println!("cargo:rustc-link-lib={}", name);
            }
        }

        for flag in llvm_config("--system-libs")?.trim().split(' ') {
            let flag = flag.trim_start_matches("-l");

            if flag.starts_with('/') {
                // llvm-config returns absolute paths for dynamically linked libraries.
                let path = Path::new(flag);

                println!(
                    "cargo:rustc-link-search={}",
                    path.parent().unwrap().display()
                );
                println!(
                    "cargo:rustc-link-lib={}",
                    path.file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .split_once('.')
                        .unwrap()
                        .0
                        .trim_start_matches("lib")
                );
            } else {
                println!("cargo:rustc-link-lib={}", flag);
            }
        }

        if let Some(name) = get_system_libcpp() {
            println!("cargo:rustc-link-lib={}", name);
        }

        bindgen::builder()
            .header("include/ReuseIR/CAPI.h")
            .clang_arg(format!("-I{}", llvm_config("--includedir")?))
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .unwrap()
            .write_to_file(Path::new(&env::var("OUT_DIR")?).join("bindings.rs"))?;

        Ok(())
    }

    fn get_system_libcpp() -> Option<&'static str> {
        if cfg!(target_env = "msvc") {
            None
        } else if cfg!(target_os = "macos") {
            Some("c++")
        } else {
            Some("stdc++")
        }
    }

    fn llvm_config(argument: &str) -> Result<String, Box<dyn Error>> {
        let prefix = env::var("LLVM_DIR")
            .map(|path| Path::new(&path).join("bin"))
            .unwrap_or_default();
        let call = format!(
            "{} --link-static {}",
            prefix.join("llvm-config").display(),
            argument
        );

        Ok(str::from_utf8(
            &if cfg!(target_os = "windows") {
                Command::new("cmd").args(["/C", &call]).output()?
            } else {
                Command::new("sh").arg("-c").arg(&call).output()?
            }
            .stdout,
        )?
        .trim()
        .to_string())
    }

    fn trim_library_name(name: &str) -> Option<&str> {
        if let Some(name) = name.strip_prefix("lib") {
            name.strip_suffix(".a")
        } else {
            None
        }
    }
}

fn main() {
    let dst = cmake::Config::new(".")
        .generator("Ninja")
        .build()
        .join("build")
        .join("lib");
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=MLIRReuseIRCAPI");
    println!("cargo:rustc-link-lib=static=MLIRReuseIR");
    println!("cargo:rustc-link-lib=static=MLIRReuseIRPasses");
    println!("cargo:rustc-link-lib=static=MLIRReuseIRAnalysis");
    println!("cargo:rustc-link-lib=static=MLIRReuseIRInterfaces");
    mlir::run().unwrap();
}
