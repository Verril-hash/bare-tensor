mod gpu;
mod loader;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use log::info;
use safetensors::SafeTensors;

#[derive(Parser)]
#[command(name = "bare-tensor")]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download a real, production AI model from Hugging Face and parse its structure.
    DownloadModel {
        /// Target URL (Defaults to ALBERT-base, as it is a compact 45MB model)
        #[arg(
            short,
            long,
            default_value = "https://huggingface.co/albert/albert-base-v2/resolve/main/model.safetensors"
        )]
        url: String,

        /// Local output path for the downloaded file.
        #[arg(short, long, default_value = "model.safetensors")]
        output: String,
    },
    /// Run Step 7 Prove-It Test: Execute Math on Real Hugging Face Tensors!
    RunReal {
        /// Path to the .safetensors model file.
        #[arg(short, long, default_value = "model.safetensors")]
        model: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize standard logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::DownloadModel { url, output } => {
            loader::download_model(&url, &output).await?;
            let mapped_file = loader::map_tensors(&output)?;
            loader::print_metadata(&mapped_file)?;
        }
        Commands::RunReal { model } => {
            run_real_model_inference(&model).await?;
        }
    }

    Ok(())
}

/// Orchestrates the process of loading the model, finding the tensor, generating dummy data, and running on the GPU.
async fn run_real_model_inference(model_path: &str) -> Result<()> {
    info!("Phase 1: Memory-Mapping SafeTensors...");
    let mapped_file = loader::map_tensors(model_path)?;
    let tensors = SafeTensors::deserialize(&mapped_file).context("Failed to deserialize safetensors metadata")?;

    // Search for a target weight tensor (Attention Query Weight)
    let target_tensor_name = tensors
        .names()
        .into_iter()
        .find(|name| name.contains("attention.query.weight") || name.contains("q_proj.weight"))
        .context("Could not find an attention query weight tensor in this model!")?;

    let real_tensor = tensors.tensor(target_tensor_name).context("Failed to read actual tensor")?;
    let shape_real = real_tensor.shape();

    info!("Searching for target attention layer... FOUND: {}", target_tensor_name);
    info!("Real Tensor Shape: {:?}", shape_real);

    // Safety check - we expect a 2D matrix like [768, 768] (which ALBERT base query weight is)
    if shape_real.len() != 2 {
        anyhow::bail!("Expected 2D target matrix for this test, found shape: {:?}", shape_real);
    }
    
    let real_rows = shape_real[0] as u32; // K for MatMul
    let real_cols = shape_real[1] as u32; // N for MatMul

    // 2. The Simulated Input
    info!("Phase 2: Generating simulated [1, {}] Text Embedding...", real_rows);
    let mut dummy_embedding: Vec<f32> = Vec::with_capacity(real_rows as usize);
    for i in 0..real_rows {
        // Fill with a dynamic but extremely small predictable pattern
        dummy_embedding.push((i as f32 % 10.0) * 0.01);
    }
    let dummy_bytes = bytemuck::cast_slice(&dummy_embedding);

    let m = 1u32;
    let k = real_rows;
    let n = real_cols;

    let bytes_a: &[u8] = dummy_bytes;      // Simulated Text Vector
    let bytes_b: &[u8] = real_tensor.data(); // Exact mapped pointer into weights

    // Run GPU Compute
    let final_data = gpu::execute_matmul(bytes_a, bytes_b, m, k, n).await?;

    info!("========================================");
    info!("REAL HUGGING FACE INFERENCE SUCCESS!");
    info!("TENSOR NAME: {}", target_tensor_name);
    info!("========================================");
    info!("Output Vector Dimension: [1, {}]", n);
    info!("First 10 Real Output Numbers:");
    info!("{:?}", &final_data[..10.min(n as usize)]);
    info!("========================================");

    Ok(())
}
