from pathlib import Path

# Listę datasetów możesz rozbudować wg potrzeb
datasets = ["motion", "general"]
models = ["Rcnn", "Deeplab"]

# Tworzenie struktury Data/
for ds in datasets:
    (Path("Data") / ds / "Images").mkdir(parents=True, exist_ok=True)
    (Path("Data") / ds / "Groundtruth").mkdir(parents=True, exist_ok=True)
    # Gitkeep do pustych folderów
    (Path("Data") / ds / "Images" / ".gitkeep").touch()
    (Path("Data") / ds / "Groundtruth" / ".gitkeep").touch()

# Tworzenie struktury Results/
for ds in datasets:
    for model in models:
        (Path("Results") / ds / "Images" / model).mkdir(parents=True, exist_ok=True)
        (Path("Results") / ds / "Images" / model / ".gitkeep").touch()

print("Struktura folderów przygotowana! Dodano pliki .gitkeep do pustych katalogów.")
