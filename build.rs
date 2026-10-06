// Records the compiler version for the platform key (`tensor::platform_key`), and links the GPU
// backends' system libraries when their feature is set: with `metal` (macOS) Metal.framework,
// Foundation and libobjc; with `cuda` the CUDA driver, NVRTC and cuBLAS from the toolkit at
// CUDA_PATH (or CUDA_HOME, else /usr/local/cuda).
fn main() {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let v = std::process::Command::new(rustc).arg("--version").output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).unwrap_or_default();
    let v = v.split_whitespace().take(2).collect::<Vec<_>>().join(" ");
    println!("cargo:rustc-env=TENSOR_BUILD_RUSTC={}", if v.is_empty() { "unknown".into() } else { v });
    if std::env::var_os("CARGO_FEATURE_METAL").is_some() && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=objc");
    }
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
        link_cuda();
    }
}

fn link_cuda() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    let root = std::env::var("CUDA_PATH").or_else(|_| std::env::var("CUDA_HOME")).unwrap_or_else(|_| "/usr/local/cuda".into());
    let root = std::path::PathBuf::from(root);
    let windows = std::env::var("CARGO_CFG_TARGET_OS").map(|o| o == "windows").unwrap_or(false);
    let dirs = if windows { vec![root.join("lib").join("x64")] } else { vec![root.join("lib64"), root.join("lib64").join("stubs")] };
    for d in dirs.iter().filter(|d| d.is_dir()) {
        println!("cargo:rustc-link-search=native={}", d.display());
    }
    for lib in ["cuda", "nvrtc", "cublas"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
