use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::{Mmap, MmapOptions};
use reqwest::Client;
use safetensors::SafeTensors;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Download a model from a URL and save it to the local path.
pub async fn download_model(url: &str, output_path: &str) -> Result<()> {
    println!("Initiating HTTP connection to Hugging Face...");

    let client = Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .context("Failed to send HTTP request")?;

    if !response.status().is_success() {
        anyhow::bail!("Failed to download. Status code: {}", response.status());
    }

    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
            .progress_chars("█▓▒░ "),
    );
    pb.set_message(format!("Downloading to {}", output_path));

    let mut file = File::create(output_path).context("Failed to create local model file")?;
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.context("Error while reading stream chunk")?;
        file.write_all(&chunk).context("Error while writing chunk to file")?;
        downloaded += chunk.len() as u64;
        pb.set_position(downloaded);
    }

    pb.finish_with_message(format!("Download successfully completed and saved as {}!", output_path));

    Ok(())
}

/// A thin wrapper to map a SafeTensors file into memory safely.
pub fn map_tensors(file_path: &str) -> Result<Mmap> {
    let path = Path::new(file_path);
    let file = File::open(path).context(format!("Failed to open model file: {}", file_path))?;
    let mapped_file = unsafe {
        MmapOptions::new()
            .map(&file)
            .context("Failed to memory-map the safetensors file")?
    };
    Ok(mapped_file)
}

/// Print the metadata (layers, dtypes, shapes) for debugging/verification.
pub fn print_metadata(mapped_file: &Mmap) -> Result<()> {
    println!("\n{:<65} | {:<10} | {}", "TENSOR NAME", "DTYPE", "SHAPE");
    println!("{}+{}+{}", "-".repeat(66), "-".repeat(12), "-".repeat(30));

    let tensors = SafeTensors::deserialize(mapped_file).context("Failed to deserialize safetensors metadata")?;
    let mut names: Vec<_> = tensors.names().into_iter().collect();
    names.sort();

    for name in names.iter().take(15) {
        let tensor = tensors.tensor(name).context("Failed to read tensor")?;
        println!("{:<65} | {:<10?} | {:?}", name, tensor.dtype(), tensor.shape());
    }

    println!("... (Skipping remaining layers)\n");
    println!("Metadata successfully ingested from disk!");

    Ok(())
}
