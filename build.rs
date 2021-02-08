fn main() {
    println!("cargo:rustc-link-search=native=armffi/lib");
    println!("cargo:rustc-link-lib=static=armffi");
}