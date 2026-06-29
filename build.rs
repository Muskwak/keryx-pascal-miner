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
    // Regenerate the per-arch PTX selector (pom_ptx.rs) whenever this script changes.
    println!("cargo:rerun-if-changed=build.rs");
    tonic_build::configure()
        .build_server(false)
        // .type_attribute(".", "#[derive(Debug)]")
        .compile(
            &["proto/rpc.proto", "proto/p2p.proto", "proto/messages.proto"],
            &["proto"],
        )?;
    // PoM mining kernel → PTX (loaded at runtime into candle's CUDA context). Compile for all major NVIDIA architectures.
    println!("cargo:rerun-if-changed=cuda/pom_mine.cu");
    {
        let out_dir = env::var("OUT_DIR").unwrap();
        let nvcc = env::var("NVCC").ok().unwrap_or_else(|| {
            if let Some(base) = env::var_os("CUDA_PATH") {
                let mut p = std::path::PathBuf::from(base);
                p.push("bin");
                p.push(if cfg!(windows) { "nvcc.exe" } else { "nvcc" });
                if p.exists() { return p.to_string_lossy().into_owned(); }
            }
            "nvcc".to_string()
        });

        // Compile PTX for all major NVIDIA compute capabilities: Pascal (61), Volta (70), Turing (75), Ampere (80/86), Ada (89), Hopper (90)
        let archs = [("61", "PomMinerSm61"), ("70", "PomMinerSm70"), ("75", "PomMinerSm75"),
                     ("80", "PomMinerSm80"), ("86", "PomMinerSm86"), ("89", "PomMinerSm89"),
                     ("90", "PomMinerSm90")];

        let mut ptx_includes = Vec::new();

        for (arch, _name) in &archs {
            let ptx_path = format!("{out_dir}/pom_mine_sm{arch}.ptx");
            let mut cmd = std::process::Command::new(&nvcc);
            cmd.args(["-ptx", "-O3", "--use_fast_math",
                      &format!("-arch=sm_{arch}"), "cuda/pom_mine.cu", "-o", &ptx_path]);
            if let Ok(c) = env::var("MSVC_CLBIN").or_else(|_| discover_clbin()) {
                cmd.arg("-ccbin").arg(&c);
            }

            cmd.stdout(std::process::Stdio::piped());
            cmd.stderr(std::process::Stdio::piped());
            let out = cmd.output().unwrap_or_else(|e| panic!("nvcc ({nvcc}) failed to run: {e}"));
            if !out.status.success() {
                let stderr = String::from_utf8_lossy(&out.stderr);
                let stdout = String::from_utf8_lossy(&out.stdout);
                panic!("nvcc failed to compile cuda/pom_mine.cu (sm_{arch}):\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
            }

            // Generate include statement for this architecture
            ptx_includes.push(format!(
                "    (({major}, {minor}), include_str!(concat!(env!(\"OUT_DIR\"), \"/pom_mine_sm{arch}.ptx\"))),",
                major = arch.chars().next().unwrap(),
                minor = arch.chars().nth(1).unwrap(),
                arch = arch
            ));
        }

        // Generate a lookup function module with proper match arm syntax
        let mut match_arms = String::new();
        for (arch, _name) in &archs {
            let major = arch.chars().next().unwrap();
            let minor = arch.chars().nth(1).unwrap();
            match_arms.push_str(&format!(
                "        ({}, {}) => include_str!(concat!(env!(\"OUT_DIR\"), \"/pom_mine_sm{}.ptx\")),\n",
                major, minor, arch
            ));
        }

        let lookup_code = format!(
            "/// Auto-selected PTX based on GPU compute capability\npub fn get_pom_ptx(cc: (u32, u32)) -> &'static str {{\n    match cc {{\n{}        _ => include_str!(concat!(env!(\"OUT_DIR\"), \"/pom_mine_sm61.ptx\")),\n    }}\n}}",
            match_arms
        );

        std::fs::write(
            format!("{out_dir}/pom_ptx.rs"),
            lookup_code,
        ).unwrap_or_else(|e| panic!("Failed to write pom_ptx.rs: {e}"));
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
