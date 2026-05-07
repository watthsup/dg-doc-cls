from __future__ import annotations

import click
from src.adapters.inbound.cli.process_command import execute_process_command
from src.adapters.inbound.cli.gen_report_command import execute_gen_report_command

@click.group()
def cli():
    """DocGuru Classification CLI."""
    pass

@cli.command(name="process")
@click.option("--file", help="Path to a single document file")
@click.option("--dir", "directory", help="Path to a directory containing documents")
@click.option("--out", "output_dir", default="./output", help="Output directory")
@click.option("--type", "graph_type", default="doc_cls", help="Graph type (default: doc_cls)")
@click.option("--concurrency", default=3, help="Max concurrent processes")
@click.option("--verbose", is_flag=True, help="Enable debug logging")
def process(file, directory, output_dir, graph_type, concurrency, verbose):
    """Process document(s) and classify pages."""
    if not file and not directory:
        raise click.UsageError("Either --file or --dir must be provided.")
    
    exit_code = execute_process_command(
        file=file,
        directory=directory,
        output_dir=output_dir,
        graph_type=graph_type,
        max_concurrency=concurrency,
        verbose=verbose
    )
    if exit_code != 0:
        raise click.ClickException("Process command failed.")

@cli.command(name="gen-report")
@click.option("--dir", "directory", required=True, help="Path to a directory containing documents")
@click.option("--out", "output_dir", default="./reports", help="Output directory for HTML reports")
@click.option("--type", "graph_type", default="doc_cls", help="Graph type (default: doc_cls)")
@click.option("--concurrency", default=3, help="Max concurrent processes")
@click.option("--verbose", is_flag=True, help="Enable debug logging")
def gen_report(directory, output_dir, graph_type, concurrency, verbose):
    """Generate interactive HTML reports for documents in a directory."""
    exit_code = execute_gen_report_command(
        directory=directory,
        output_dir=output_dir,
        graph_type=graph_type,
        concurrency=concurrency,
        verbose=verbose
    )
    if exit_code != 0:
        raise click.ClickException("Gen-report command failed.")

def build_cli():
    return cli
