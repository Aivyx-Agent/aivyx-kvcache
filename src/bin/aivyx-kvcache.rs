use std::path::PathBuf;

use aivyx_kvcache::cli;
use clap::{Parser, Subcommand};

/// Inspect and prune the local aivyx-kvcache store.
#[derive(Parser)]
struct Cli {
    /// Path to the kvcache store directory (containing manifest.db and slots/).
    #[arg(long, env = "AIVYX_KVCACHE_DIR")]
    store_path: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List every cached slot.
    List,
    /// Show total bytes used against a budget.
    Stats {
        /// Budget in bytes to report usage against.
        #[arg(long)]
        max_bytes: u64,
    },
    /// Evict entries down to (at most) a target byte count.
    Prune {
        /// Evict until total bytes are at or under this value.
        #[arg(long)]
        to: u64,
        /// Report what would be evicted without deleting anything.
        #[arg(long)]
        dry_run: bool,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let output = match cli.command {
        Command::List => cli::list_slots(&cli.store_path).await?,
        Command::Stats { max_bytes } => cli::stats(&cli.store_path, max_bytes).await?,
        Command::Prune { to, dry_run } => cli::prune(&cli.store_path, to, dry_run).await?,
    };
    println!("{output}");
    Ok(())
}
