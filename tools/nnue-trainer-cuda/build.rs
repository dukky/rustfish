//! Compiles src/kernels.cu to PTX with nvcc and links the CUDA driver
//! (libcuda). nvcc: $NVCC, else $CUDA_HOME/bin/nvcc, else ~/cuda-12.8, else
//! PATH. Target: $NNUE_CUDA_ARCH (default compute_61, Pascal; the driver
//! JIT-compiles the PTX for newer GPUs).
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=NNUE_CUDA_ARCH");
    let home = std::env::var("HOME").unwrap_or_default();
    let nvcc = std::env::var("NVCC").ok().or_else(|| std::env::var("CUDA_HOME").ok().map(|c| format!("{c}/bin/nvcc"))).unwrap_or_else(|| {
        let local = format!("{home}/cuda-12.8/bin/nvcc");
        if std::path::Path::new(&local).exists() { local } else { "nvcc".into() }
    });
    let arch = std::env::var("NNUE_CUDA_ARCH").unwrap_or_else(|_| "compute_61".into());
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("kernels.ptx");
    let st = Command::new(&nvcc)
        .args(["-ptx", "-O3", &format!("-arch={arch}"), "-Wno-deprecated-gpu-targets", "src/kernels.cu", "-o"])
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("cannot run {nvcc}: {e} (set NVCC)"));
    assert!(st.success(), "nvcc failed");
    for dir in ["/usr/lib/wsl/lib", "/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/lib64/stubs"] {
        println!("cargo:rustc-link-search=native={dir}");
    }
    println!("cargo:rustc-link-lib=dylib=cuda");
}
