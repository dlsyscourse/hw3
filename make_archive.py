"""Package your HW3 code for mugrade's "Upload Code Archive" step.

Run from anywhere:  python3 make_archive.py
This writes hw3.tar.gz next to this file, containing your source code (python/ and src/)
but none of the build outputs (build/, *.so, __pycache__), which are what make the
archive too large to upload.
"""

import pathlib
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent
ARCHIVE = ROOT / "hw3.tar.gz"
INCLUDE = ["python", "src"]
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "build"}
SKIP_SUFFIXES = {".so", ".pyc", ".o", ".dylib"}


def wanted(path):
    rel = path.relative_to(ROOT)
    return (
        path.is_file()
        and not SKIP_DIRS.intersection(rel.parts)
        and path.suffix not in SKIP_SUFFIXES
        and not path.name.startswith("._")
        and path.name != ".DS_Store"
    )


def main():
    files = sorted(p for d in INCLUDE for p in (ROOT / d).rglob("*") if wanted(p))
    with tarfile.open(ARCHIVE, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    for path in files:
        print("  " + str(path.relative_to(ROOT)))
    print(f"Wrote {ARCHIVE.name}: {len(files)} files, {ARCHIVE.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
