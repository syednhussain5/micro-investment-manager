# scripts/create_release_zip.py
import os
import zipfile

def zip_project(output="microinvestment_release.zip", include_dirs=None):
    if include_dirs is None:
        include_dirs = ["src", "app.py", "data", "requirements.txt"]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for path in include_dirs:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for f in files:
                        full = os.path.join(root, f)
                        z.write(full)
            elif os.path.isfile(path):
                z.write(path)
    print(f"Created {output}")

if __name__ == "__main__":
    zip_project()
