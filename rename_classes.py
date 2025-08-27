from pathlib import Path
import shutil

ROOT = Path("data/kaggle_math")  # change if you keep classes under train/
MAP = {
    "!": "exclam",
    "+": "plus",
    "-": "minus",
    "=": "equals",
    "(": "lparen",
    ")": "rparen",
    "*": "times",
    "/": "divide",
    "^": "caret",
    "∫": "integral",
    "√": "sqrt",
    "×": "times_unicode",
    "÷": "divide_unicode",
}

def merge_or_rename(src: Path, dst: Path):
    if not src.exists():
        print(f"Skip: {src.name} not found")
        return
    if dst.exists():
        for p in src.iterdir():
            shutil.move(str(p), dst / p.name)
        src.rmdir()
        print(f"Merged {src.name} -> {dst.name}")
    else:
        src.rename(dst)
        print(f"Renamed {src.name} -> {dst.name}")

def main():
    for k, v in MAP.items():
        merge_or_rename(ROOT / k, ROOT / v)

if __name__ == "__main__":
    main()
