from pathlib import Path

folder = Path("jsons")

for file in folder.iterdir():
    if not file.is_file():
        continue

    parts = file.stem.split("_", 1)

    if len(parts) < 2:
        continue

    number, name = parts

    if number.isdigit():
        new_name = f"{int(number):03d}_{name}{file.suffix}"
        file.rename(file.with_name(new_name))
        print(f"{file.name} -> {new_name}")