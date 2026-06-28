use std::env;
use time::{format_description, OffsetDateTime};

/// Locate cl.exe for nvcc on Windows when it's not on PATH. Searches the standard VS 2022
/// Community/BuildTools install for the newest MSVC toolchain's Hostx64/x64 cl.exe. Returns
/// Err on non-Windows or if nothing is found (nvcc then falls back to PATH).
fn discover_clbin() -> Result<String, std::env::VarError> {
    if env::var("CARGO_CFG_TARGET_OS").ok().as_deref() != Some("windows") {
        return Err(std::env::VarError::NotPresent);
    }
    let roots = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
    ];
    for root in roots {
        let tools = std::path::Path::new(root).join("VC\\Tools\\MSVC");
        let Ok(entries) = std::fs::read_dir(&tools) else { continue };
        // Newest toolchain version sorts last lexicographically.
        let mut versions: Vec<_> = entries.flatten().filter_map(|e| e.file_name().into_string().ok()).collect();
        versions.sort();
        if let Some(v) = versions.last() {
            let cl = tools.join(v).join("bin\\Hostx64\\x64\\cl.exe");
            if cl.exists() {
                return Ok(cl.to_string_lossy().into_owned());
            }
        }
    }
    Err(std::env::VarError::NotPresent)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let format = format_description::parse_borrowed::<2>("[year repr:last_two][month][day][hour][minute]")?;
    let dt = OffsetDateTime::now_utc().format(&format)?;
    println!("cargo:rustc-env=PACKAGE_COMPILE_TIME={}", dt);

    println!("cargo:rerun-if-changed=proto");
    println!("cargo:rerun-if-changed=src/keccakf1600_x86-64.s");
    tonic_build::configure()
        .build_server(false)
        // .type_attribute(".", "#[derive(Debug)]")
        .compile(
            &["proto/rpc.proto", "proto/p2p.proto", "proto/messages.proto"],
            &["proto"],
        )?;
    // PoM mining kernel → PTX (loaded at runtime into candle's CUDA context). nvcc 12.2 (PATH).
    println!("cargo:rerun-if-changed=cuda/pom_mine.cu");
    {
        let out_dir = env::var("OUT_DIR").unwrap();
        // Resolve nvcc explicitly: NVCC env var wins; else CUDA_PATH/bin/nvcc(.exe); else PATH.
        // This matters when multiple CUDA toolkits coexist (e.g. v11.8 + v12.8) — without it the
        // bare "nvcc" resolves via PATH and may pick an older toolkit whose headers reject a
        // newer MSVC (fatal error C1189), producing a confusing build failure. CUDA_PATH is the
        // standard env var the rest of the CUDA ecosystem honors.
        let nvcc = env::var("NVCC").ok().unwrap_or_else(|| {
            if let Some(base) = env::var_os("CUDA_PATH") {
                let mut p = std::path::PathBuf::from(base);
                p.push("bin");
                p.push(if cfg!(windows) { "nvcc.exe" } else { "nvcc" });
                if p.exists() { return p.to_string_lossy().into_owned(); }
            }
            "nvcc".to_string()
        });
        // Default to sm_61 (Tesla P40 / Pascal GP102) — this fork is the Pascal-tuned miner.
        // Override with SM_ARCH for other cards (e.g. sm_75 for Turing, sm_86 for Ampere).
        let sm = env::var("SM_ARCH").unwrap_or_else(|_| "61".to_string());
        let ptx = format!("{out_dir}/pom_mine.ptx");

        // On Windows nvcc needs MSVC's cl.exe as host compiler. Normally it's on PATH via
        // build_release.bat's vcvars64; if not (e.g. building from a bare shell), honor an
        // explicit MSVC_CLBIN env var or auto-discover the newest VS 2022 toolchain so the build
        // is robust regardless of how it's launched.
        // NOTE: -Xptxas=-v (verbose ptxas, for register/occupancy tuning) is intentionally
        // omitted — it crashes nvcc on deprecated targets (sm_61/75) with CUDA 12.8 + MSVC,
        // failing the build. The register/occupancy tuning is already done (21 regs, 100%
        // occupancy on sm_61); re-enable locally with SM_ARCH=89 if you need to re-measure.
        let mut cmd = std::process::Command::new(&nvcc);
        cmd.args(["-ptx", "-O3", "--use_fast_math",
                  &format!("-arch=sm_{sm}"), "cuda/pom_mine.cu", "-o", &ptx]);
        if let Ok(c) = env::var("MSVC_CLBIN").or_else(|_| discover_clbin()) {
            println!("cargo:warning=nvcc host compiler: {c}");
            cmd.arg("-ccbin").arg(c);
        }

        // -Xptxas=-v surfaces register usage + occupancy so the threads/block + maxrregcount
        // tuning in pom_gpu.rs can be measured instead of guessed.
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        let out = cmd.output().unwrap_or_else(|e| panic!("nvcc ({nvcc}) failed to run: {e}"));
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let stdout = String::from_utf8_lossy(&out.stdout);
            panic!("nvcc failed to compile cuda/pom_mine.cu (sm_{sm}):\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
        }
    }

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target_arch == "x86_64" && target_os != "windows" && target_os != "macos" {
        cc::Build::new().flag("-c").file("src/keccakf1600_x86-64.s").compile("libkeccak.a");
    }
    if target_arch == "x86_64" && target_os == "macos" {
        cc::Build::new().flag("-c").file("src/keccakf1600_x86-64-osx.s").compile("libkeccak.a");
    }
    Ok(())
}
