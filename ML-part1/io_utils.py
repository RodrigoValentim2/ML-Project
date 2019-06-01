from pathlib import Path

def prepare_outdir(outdir_path: Path):
    if outdir_path.exists():
        if outdir_path.is_file():
            print('ERROR: output path is a file')
    else:
        outdir_path.mkdir(parents=True)
