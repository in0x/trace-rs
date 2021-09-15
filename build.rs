#[cfg(target_arch = "aarch64")]
fn main() {
    // TODO: compile the ffi file in here";
    println!("cargo:rustc-link-search=native=armffi/lib");
    println!("cargo:rustc-link-lib=static=armffi");
}

#[cfg(target_arch = "x86_64")]
fn main() {
    
}