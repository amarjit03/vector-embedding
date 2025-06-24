import os, zipfile

def unzip_codebase(zip_path: str, extract_to: str = "codebase") -> str:
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to

def get_python_files(folder: str) -> list[str]:
    py_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    py_files.append(f.read())
    return py_files
