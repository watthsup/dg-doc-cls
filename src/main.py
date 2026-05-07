from __future__ import annotations
import sys
import warnings
from src.adapters.inbound.cli.parser import build_cli

# Suppress noisy library-level deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

def main():
    cli = build_cli()
    try:
        cli()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
