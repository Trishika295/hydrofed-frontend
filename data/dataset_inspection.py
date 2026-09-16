import os
import re
import json
import csv
import hashlib
from PIL import Image

def get_image_hash(filepath):
    """Compute MD5 hash of an image file to check for duplicates."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def parse_patient_id(filename, class_name):
    """Extract patient ID from file name based on class characteristics."""
    if class_name == 'PNEUMONIA':
        m = re.match(r'^(person\d+)_', filename)
        if m:
            return m.group(1)
    else:
        m1 = re.match(r'^NORMAL2-IM-(\d+)', filename)
        m2 = re.match(r'^IM-(\d+)', filename)
        if m1:
            return 'normal_' + m1.group(1)
        elif m2:
            return 'normal_' + m2.group(2) if len(m2.groups()) > 1 else 'normal_' + m2.group(1)
    return None

def inspect_dataset(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report_json_path = os.path.join(output_dir, 'dataset_report.json')
    report_csv_path = os.path.join(output_dir, 'dataset_report.csv')

    stats = {
        'total_images': 0,
        'by_split': {},
        'by_class': {'NORMAL': 0, 'PNEUMONIA': 0, 'BACTERIA': 0, 'VIRUS': 0},
        'corrupted_files': [],
        'duplicates': [],
        'dimensions': {},
        'modes': {}
    }

    seen_hashes = {}
    records = []

    for split in ['train', 'val', 'test']:
        stats['by_split'][split] = {'NORMAL': 0, 'PNEUMONIA': 0}
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            continue

        for cls in ['NORMAL', 'PNEUMONIA']:
            cls_path = os.path.join(split_path, cls)
            if not os.path.exists(cls_path):
                continue

            for filename in os.listdir(cls_path):
                if not filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                    continue

                filepath = os.path.join(cls_path, filename)
                stats['total_images'] += 1
                stats['by_split'][split][cls] += 1
                stats['by_class'][cls] += 1

                # Parse specific type for pneumonia
                subclass = cls
                if cls == 'PNEUMONIA':
                    if 'bacteria' in filename.lower():
                        stats['by_class']['BACTERIA'] += 1
                        subclass = 'BACTERIA'
                    elif 'virus' in filename.lower():
                        stats['by_class']['VIRUS'] += 1
                        subclass = 'VIRUS'

                patient_id = parse_patient_id(filename, cls)

                # Integrity and duplication check
                is_corrupted = False
                width, height, mode = 0, 0, 'Unknown'
                try:
                    with Image.open(filepath) as img:
                        width, height = img.size
                        mode = img.mode
                        img.verify()
                except Exception as e:
                    stats['corrupted_files'].append({'path': filepath, 'error': str(e)})
                    is_corrupted = True

                if not is_corrupted:
                    try:
                        file_hash = get_image_hash(filepath)
                        if file_hash in seen_hashes:
                            stats['duplicates'].append({'original': seen_hashes[file_hash], 'duplicate': filepath})
                        else:
                            seen_hashes[file_hash] = filepath
                    except Exception as e:
                        pass

                    # Log dimensions and modes
                    dim_str = f"{width}x{height}"
                    stats['dimensions'][dim_str] = stats['dimensions'].get(dim_str, 0) + 1
                    stats['modes'][mode] = stats['modes'].get(mode, 0) + 1

                records.append({
                    'filename': filename,
                    'filepath': filepath,
                    'split': split,
                    'class': cls,
                    'subclass': subclass,
                    'patient_id': patient_id if patient_id else 'unknown',
                    'width': width,
                    'height': height,
                    'mode': mode,
                    'is_corrupted': int(is_corrupted)
                })

    # Save outputs
    with open(report_json_path, 'w') as f:
        json.dump(stats, f, indent=4)

    with open(report_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'filepath', 'split', 'class', 'subclass', 'patient_id', 'width', 'height', 'mode', 'is_corrupted'])
        writer.writeheader()
        writer.writerows(records)

    print(f"Inspection complete. Total images processed: {stats['total_images']}.")
    print(f"Report saved to: {report_json_path} and {report_csv_path}")
    return stats

if __name__ == '__main__':
    data_dir = os.path.join('archive (4)', 'chest_xray')
    output_dir = 'reports'
    inspect_dataset(data_dir, output_dir)
