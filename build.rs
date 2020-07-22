use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rustc-link-lib=gdrapi");
    println!("cargo:rerun-if-changed=src/");

    let bindings = bindgen::Builder::default()
        .header("src/bindings.h")
        .generate_comments(true)
        .generate_inline_functions(true)
        .opaque_type("gdr")
        .whitelist_type("grd.*")
        .whitelist_function("gdr_.*")
        .whitelist_var("(gdr|GDR|MINIMUM_GDR|GPU)_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .rustfmt_bindings(true)
        .generate()
        .unwrap();

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .unwrap();
}
