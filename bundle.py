import os
import sys


def bundle_folder(folder_path, output_file="bundle.txt"):
    folder_path = os.path.abspath(folder_path)

    with open(output_file, "w", encoding="utf-8") as out:
        for root, dirs, files in os.walk(folder_path):
            # Skip unwanted directories
            dirs[:] = [
                d for d in dirs
                if d not in {".git", "__pycache__", ".venv", "venv", "node_modules", "data", "log", "tmp_fedsvd", "archive", "docs"}
            ]

            for file in sorted(files):
                # Skip unwanted files
                if file in {"uv.lock", "package-lock.json"}:
                    continue

                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except (UnicodeDecodeError, PermissionError):
                    continue

                out.write(f"\n{'=' * 80}\n")
                out.write(f"FILE: {relative_path}\n")
                out.write(f"{'=' * 80}\n\n")
                out.write(content)
                out.write("\n")

    print(f"Bundled '{folder_path}' -> '{output_file}'")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bundle.py <folder> [output_file]")
        sys.exit(1)

    folder = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "bundle.txt"

    bundle_folder(folder, output)