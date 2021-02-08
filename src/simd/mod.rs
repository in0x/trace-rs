#[cfg(target_feature = "sse3")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod arm;

#[cfg(target_feature = "sse3")]
pub use x86::*;

#[cfg(target_arch = "aarch64")]
pub use arm::*;