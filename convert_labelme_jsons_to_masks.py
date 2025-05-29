import os
import json
import base64
from labelme import utils
import numpy as np
import PIL.Image

def convert_json_to_mask(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    imageData = data.get('imageData')
    if imageData is None:
        imagePath = os.path.join(os.path.dirname(json_path), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = base64.b64encode(f.read()).decode('utf-8')

    img = utils.img_b64_to_arr(imageData)
    label_name_to_value = {'_background_': 0}
    for shape in data['shapes']:
        label_name = shape['label']
        if label_name in label_name_to_value:
            continue
        label_name_to_value[label_name] = len(label_name_to_value)

    lbl, _ = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

    base_name = os.path.splitext(os.path.basename(json_path))[0]
    out_mask_path = os.path.join(output_dir, base_name + "_label.png")
    PIL.Image.fromarray(lbl).save(out_mask_path)

    print(f"[OK] Zapisano maskę: {out_mask_path}")

def convert_folder(json_folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    if not json_files:
        print("[!] Nie znaleziono plików .json w folderze.")
        return

    for json_file in json_files:
        full_path = os.path.join(json_folder, json_file)
        convert_json_to_mask(full_path, output_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Konwertuj pliki LabelMe JSON na maski PNG.")
    parser.add_argument("input_dir", help="Folder z plikami .json (LabelMe)")
    parser.add_argument("output_dir", help="Folder docelowy dla masek PNG")

    args = parser.parse_args()

    convert_folder(args.input_dir, args.output_dir)

