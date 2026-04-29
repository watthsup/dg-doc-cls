import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
import click
import tqdm
import os

@click.command()
@click.option("--csv-path", type=click.Path(exists=True), required=True, help="Path to batch_results.csv")
@click.option("--input-dir", type=click.Path(exists=True), required=True, help="Directory containing original PDF/TIF files")
@click.option("--output-dir", type=click.Path(), required=True, help="Root directory to save PNG pages")
@click.option("--dpi", type=int, default=150, help="Resolution for PNG conversion")
def main(csv_path, input_dir, output_dir, dpi):
    """
    Convert PDF/TIF pages into PNGs and organize them for the Review UI.
    Structure: output_dir/{root_code}/{sub_code}/{filename}_p_{idx}.png
    """
    df = pd.read_csv(csv_path)
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    
    # Track opened files to avoid re-loading for every page
    opened_docs = {}

    click.echo(f"Processing {len(df)} pages from {csv_path}...")

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        file_name = str(row['file_name'])
        page_idx_1 = int(row['page_index'])  # 1-indexed from CSV
        root_code = str(row['root_code'])
        sub_code = str(row['sub_code'])
        
        # 1. Prepare output path
        target_dir = output_root / root_code / sub_code
        target_dir.mkdir(parents=True, exist_ok=True)
        
        stem = Path(file_name).stem
        target_path = target_dir / f"{stem}_p_{page_idx_1}.png"
        
        if target_path.exists():
            continue

        # 2. Find original file
        source_path = input_root / file_name
        if not source_path.exists():
            # Try recursive search if not in root
            found = list(input_root.rglob(file_name))
            if found:
                source_path = found[0]
            else:
                continue

        # 3. Extract Page
        try:
            ext = source_path.suffix.lower()
            
            if ext in ['.pdf']:
                # Handle PDF with PyMuPDF
                if str(source_path) not in opened_docs:
                    opened_docs[str(source_path)] = fitz.open(source_path)
                
                doc = opened_docs[str(source_path)]
                page = doc.load_page(page_idx_1 - 1) # 0-indexed
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                pix.save(str(target_path))
                
            elif ext in ['.tif', '.tiff']:
                # Handle TIFF with PIL (multipage support)
                img = Image.open(source_path)
                img.seek(page_idx_1 - 1)
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(target_path, "PNG")
                
        except Exception as e:
            click.echo(f"Error processing {file_name} p.{page_idx_1}: {e}")

    # Cleanup
    for doc in opened_docs.values():
        doc.close()

    click.echo(f"\n✅ Done! Images saved to: {output_root}")

if __name__ == "__main__":
    main()
