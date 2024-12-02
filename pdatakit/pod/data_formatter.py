from enum import Enum
import shutil
import json
from pathlib import Path
import yaml
from typing import List, Dict, Optional
from pdatakit.pod.data_splitter import DataSplitter
from PIL import Image
import xml.etree.ElementTree as ET

class FormatType(Enum):
    YOLO_OLD = "yolo_old"
    YOLO_NEW = "yolo_new"
    COCO = "coco"
    VOC = "voc"

class DataFormatter:
    def __init__(self, data_splitter: DataSplitter, class_mapping: Optional[Dict[int, str]] = None):
        self.data_splitter = data_splitter
        self.train_files, self.val_files, self.test_files = data_splitter.get_splits()

        if not self.train_files or not self.val_files or not self.test_files:
            raise ValueError("One of the data splits is empty. Please check the DataSplitter implementation.")

        self.class_mapping = class_mapping or {i: cls for i, cls in enumerate(self.data_splitter.all_classes)}
        self.num_classes = len(self.class_mapping)
        self._validate_class_mapping()

    def _validate_class_mapping(self):
        """Validate the class_mapping"""
        if not isinstance(self.class_mapping, dict):
            raise ValueError("class_mapping must be a dictionary.")

        # Check the number of classes
        if self.num_classes != len(self.data_splitter.all_classes):
            raise ValueError(
                f"The number of classes in class_mapping ({self.num_classes}) "
                f"does not match the number of classes in the dataset ({len(self.data_splitter.all_classes)})"
            )

        seen_values = {}
        for key, value in self.class_mapping.items():
            # Check data types
            if not isinstance(key, int):
                raise ValueError(f"Keys in class_mapping must be integers. Received: {key} ({type(key)})")
            if not isinstance(value, str):
                raise ValueError(f"Values in class_mapping must be strings. Received: {value} ({type(value)})")

            # Check key range
            if key < 0 or key >= self.num_classes:
                raise ValueError(f"Class number {key} is invalid. It must be between 0 and {self.num_classes - 1}.")

            # Check for duplicate values
            if value in seen_values:
                prev_key = seen_values[value]
                raise ValueError(f"Class name '{value}' is duplicated for IDs {prev_key} and {key}")
            seen_values[value] = key

    def create_format(self, output_dir: str, format_type: FormatType):
        """Create dataset in specified format"""
        if format_type == FormatType.YOLO_OLD:
            self.create_yolo_old_format(output_dir)
        elif format_type == FormatType.YOLO_NEW:
            self.create_yolo_new_format(output_dir)
        elif format_type == FormatType.COCO:
            self.create_coco_format(output_dir)
        elif format_type == FormatType.VOC:
            self.create_voc_format(output_dir)
        else:
            print(f"Unsupported format type: {format_type}")

    def create_yolo_old_format(self, output_dir: str):
        """Create old YOLO format dataset (yolo_old)"""
        print("Creating YOLO Old format...")
        yolo_dir = Path(output_dir)
        images_dir = yolo_dir / "images"
        labels_dir = yolo_dir / "labels"

        # Create directory structure
        for split in ["train", "val", "test"]:
            (images_dir / split).mkdir(parents=True, exist_ok=True)
            (labels_dir / split).mkdir(parents=True, exist_ok=True)

        # Copy files
        for split, files in zip(["train", "val", "test"], [self.train_files, self.val_files, self.test_files]):
            self._copy_files(files, images_dir / split, labels_dir / split)

        # Create data.yaml
        self._create_yaml(yolo_dir, format_type="yolo_old")
        print("YOLO Old format created successfully.")

    def create_yolo_new_format(self, output_dir: str):
        """Create new YOLO format dataset (yolo_new) with a different directory structure"""
        print("Creating YOLO New format...")
        new_yolo_dir = Path(output_dir)

        # Create directory structure
        for split in ["train", "val", "test"]:
            (new_yolo_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (new_yolo_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        # Copy files
        for split, files in zip(["train", "val", "test"], [self.train_files, self.val_files, self.test_files]):
            self._copy_files(files, new_yolo_dir / split / "images", new_yolo_dir / split / "labels")

        # Create data.yaml
        self._create_yaml_new(new_yolo_dir)
        print("YOLO New format created successfully.")

    def create_coco_format(self, output_dir: str):
        """Create COCO format dataset"""
        print("Creating COCO format...")
        coco_dir = Path(output_dir)
        images_dir = coco_dir / "images"
        annotations_dir = coco_dir / "annotations"

        # Create directory structure
        for split in ["train", "val", "test"]:
            (images_dir / split).mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        for split, files in zip(["train", "val", "test"], [self.train_files, self.val_files, self.test_files]):
            self._copy_files(files, images_dir / split, None)

        # Create annotations for each split
        for split, files in zip(["train", "val", "test"], [self.train_files, self.val_files, self.test_files]):
            self._create_coco_annotations(files, annotations_dir, split)

        print("COCO format created successfully.")

    def create_voc_format(self, output_dir: str):
        """Create VOC format dataset"""
        print("Creating VOC format...")
        voc_dir = Path(output_dir)
        jpeg_dir = voc_dir / "JPEGImages"
        annotations_dir = voc_dir / "Annotations"
        imagesets_dir = voc_dir / "ImageSets" / "Main"

        # Create directory structure
        jpeg_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        imagesets_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dataset splits
        splits = {
            "train": self.train_files,
            "val": self.val_files,
            "test": self.test_files
        }

        # Process each split
        for split_name, files in splits.items():
            split_file = imagesets_dir / f"{split_name}.txt"
            with split_file.open('w') as f_split:
                for file in files:
                    image_path = Path(file)
                    if not image_path.exists():
                        print(f"Warning: Image file {image_path} does not exist. Skipping.")
                        continue

                    # Copy image to JPEGImages
                    dest_image_path = jpeg_dir / image_path.name
                    shutil.copy2(image_path, dest_image_path)

                    # Convert YOLO labels to VOC XML annotations
                    label_path = image_path.with_suffix('.txt')
                    if label_path.exists():
                        self._convert_yolo_to_voc(image_path, label_path, annotations_dir)
                    else:
                        print(f"Warning: Label file {label_path} does not exist for image {image_path}.")
                        # Optionally, create an empty annotation file
                        (annotations_dir / f"{image_path.stem}.xml").touch()

                    # Write image name (without extension) to the split file
                    f_split.write(f"{image_path.stem}\n")

        print("VOC format created successfully.")


    def _copy_files(self, files: List[str], images_dest: Path, labels_dest: Optional[Path]):
        """Copy image and label files to respective directories for YOLO formats"""
        for file in files:
            image_path = Path(file)
            if not image_path.exists():
                print(f"Warning: Image file {image_path} does not exist. Skipping.")
                continue

            # Copy image
            shutil.copy2(image_path, images_dest / image_path.name)

            if labels_dest:
                # Copy corresponding label
                label_path = image_path.with_suffix('.txt')
                if label_path.exists():
                    shutil.copy2(label_path, labels_dest / label_path.name)
                else:
                    print(f"Warning: Label file {label_path} does not exist for image {image_path}.")
                    # Optionally, create an empty label file
                    (labels_dest / label_path.name).touch()

    def _copy_files_voc(self, files: List[str], jpeg_dest: Path, annotations_dest: Path):
        """Copy image files and convert labels to VOC format"""
        for file in files:
            image_path = Path(file)
            if not image_path.exists():
                print(f"Warning: Image file {image_path} does not exist. Skipping.")
                continue

            # Copy image
            shutil.copy2(image_path, jpeg_dest / image_path.name)

            # Convert YOLO labels to VOC XML annotations
            label_path = image_path.with_suffix('.txt')
            if label_path.exists():
                self._convert_yolo_to_voc(image_path, label_path, annotations_dest)
            else:
                print(f"Warning: Label file {label_path} does not exist for image {image_path}.")
                # Optionally, create an empty annotation file
                (annotations_dest / f"{image_path.stem}.xml").touch()

    def _convert_yolo_to_voc(self, image_path: Path, label_path: Path, annotations_dir: Path):
        """Convert YOLO format annotations to VOC XML format"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Warning: Unable to read image {image_path}. Skipping annotation conversion.")
            return

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = image_path.parent.name
        ET.SubElement(annotation, "filename").text = image_path.name
        ET.SubElement(annotation, "path").text = str(image_path.resolve())

        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "Unknown"

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"

        ET.SubElement(annotation, "segmented").text = "0"

        try:
            with label_path.open('r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Warning: Unable to read label file {label_path}. Skipping.")
            return

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Warning: Invalid annotation in {label_path}: {line.strip()}")
                continue
            cls_id, x_center, y_center, w, h = parts
            try:
                cls_id = int(cls_id)
                x_center = float(x_center)
                y_center = float(y_center)
                w = float(w)
                h = float(h)
            except ValueError:
                print(f"Warning: Non-numeric values in {label_path}: {line.strip()}")
                continue

            if cls_id not in self.class_mapping:
                print(f"Warning: Class ID {cls_id} does not exist in class_mapping. Skipping.")
                continue

            # Convert normalized YOLO coordinates to VOC format
            x_center_abs = x_center * width
            y_center_abs = y_center * height
            w_abs = w * width
            h_abs = h * height

            xmin = int(max(x_center_abs - w_abs / 2, 0))
            ymin = int(max(y_center_abs - h_abs / 2, 0))
            xmax = int(min(x_center_abs + w_abs / 2, width))
            ymax = int(min(y_center_abs + h_abs / 2, height))

            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = self.class_mapping[cls_id]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)

        # Write XML to file
        xml_tree = ET.ElementTree(annotation)
        xml_output_path = annotations_dir / f"{image_path.stem}.xml"
        xml_output_path.parent.mkdir(parents=True, exist_ok=True)
        xml_tree.write(xml_output_path, encoding='utf-8', xml_declaration=True)

    def _create_yaml(self, output_dir: Path, format_type: str, use_absolute_paths: bool = False):
        """Create data.yaml file for YOLO formats"""
        if format_type == "yolo_old":
            if use_absolute_paths:
                train_path = str((output_dir / "images" / "train").resolve())
                val_path = str((output_dir / "images" / "val").resolve())
                test_path = str((output_dir / "images" / "test").resolve())
            else:
                train_path = str(Path("../images/train"))
                val_path = str(Path("../images/val"))
                test_path = str(Path("../images/test"))

            yaml_content = {
                'train': train_path,
                'val': val_path,
                'test': test_path,
                'nc': self.num_classes,
                'names': [self.class_mapping[i] for i in range(self.num_classes)]
            }
        else:
            print(f"Unsupported format type for YAML creation: {format_type}")
            return

        yaml_file = output_dir / "data.yaml"
        with yaml_file.open('w') as f:
            yaml.dump(yaml_content, f)
        print(f"Created data.yaml at {yaml_file}")

    def _create_yaml_new(self, output_dir: Path, use_absolute_paths: bool = False):
        """Create data.yaml file for yolo_new format"""
        if use_absolute_paths:
            train_path = str((output_dir / "train" / "images").resolve())
            val_path = str((output_dir / "val" / "images").resolve())
            test_path = str((output_dir / "test" / "images").resolve())
        else:
            train_path = str(Path("../train/images"))
            val_path = str(Path("../val/images"))
            test_path = str(Path("../test/images"))

        yaml_content = {
            'train': train_path,
            'val': val_path,
            'test': test_path,
            'nc': self.num_classes,
            'names': [self.class_mapping[i] for i in range(self.num_classes)]
        }

        yaml_file = output_dir / "data.yaml"
        with yaml_file.open('w') as f:
            yaml.dump(yaml_content, f)
        print(f"Created data.yaml at {yaml_file}")

    def _create_coco_annotations(self, files: List[str], annotations_dir: Path, split: str):
        """Create COCO format annotations with proper validation and structure"""
        print(f"Creating COCO annotations for {split}...")

        # Define COCO categories
        categories = [
            {"id": idx, "name": self.class_mapping[idx]}
            for idx in range(self.num_classes)
        ]

        # Initialize COCO annotation structure
        coco_annotations = {
            "images": [],
            "annotations": [],
            "categories": categories
        }

        image_id_counter = 1
        annotation_id_counter = 1

        for file in files:
            image_path = Path(file)
            if not image_path.exists():
                print(f"Warning: Image file {image_path} does not exist. Skipping.")
                continue

            try:
                with Image.open(image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Warning: Unable to read image {image_path}. Skipping.")
                continue

            # Add image metadata to COCO structure
            coco_annotations["images"].append({
                "id": image_id_counter,
                "file_name": image_path.name,
                "height": height,
                "width": width
            })

            # Convert YOLO annotations to COCO
            label_path = image_path.with_suffix('.txt')
            if label_path.exists():
                try:
                    with label_path.open('r') as f:
                        lines = f.readlines()
                except Exception as e:
                    print(f"Warning: Unable to read label file {label_path}. Skipping annotations for this image.")
                    image_id_counter += 1
                    continue

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Warning: Invalid annotation in {label_path}: {line.strip()}")
                        continue
                    cls_id, x_center, y_center, w, h = parts
                    try:
                        cls_id = int(cls_id)
                        x_center = float(x_center)
                        y_center = float(y_center)
                        w = float(w)
                        h = float(h)
                    except ValueError:
                        print(f"Warning: Non-numeric values in {label_path}: {line.strip()}")
                        continue

                    if cls_id not in self.class_mapping:
                        print(f"Warning: Class ID {cls_id} does not exist in class_mapping. Skipping.")
                        continue

                    # Convert YOLO bbox to COCO bbox
                    x = (x_center - w / 2) * width
                    y = (y_center - h / 2) * height
                    w_abs = w * width
                    h_abs = h * height

                    # Ensure bbox boundaries are within image
                    x = max(0, x)
                    y = max(0, y)
                    w_abs = min(w_abs, width - x)
                    h_abs = min(h_abs, height - y)

                    coco_annotations["annotations"].append({
                        "id": annotation_id_counter,
                        "image_id": image_id_counter,
                        "category_id": cls_id,
                        "bbox": [x, y, w_abs, h_abs],
                        "area": w_abs * h_abs,
                        "iscrowd": 0
                    })
                    annotation_id_counter += 1
            else:
                print(f"Warning: Label file {label_path} does not exist for image {image_path}.")

            image_id_counter += 1

        # Save COCO annotations
        annotation_file = annotations_dir / f"instances_{split}.json"
        try:
            with annotation_file.open('w') as f:
                json.dump(coco_annotations, f, indent=4)
            print(f"Saved COCO annotations to {annotation_file}")
        except Exception as e:
            print(f"Error saving COCO annotations to {annotation_file}: {e}")

    def yolo_to_coco(self, source_dir: str, target_dir: str):
        """Convert YOLO format to COCO format"""
        print("Converting YOLO format to COCO format...")
        # Implement reading from source_dir and writing to target_dir
        # This requires re-initializing DataSplitter with source_dir
        source_splitter = DataSplitter(data_root=source_dir)
        train_files, val_files, test_files = source_splitter.get_splits()

        # Update class mapping if needed
        class_mapping = {i: cls for i, cls in enumerate(source_splitter.all_classes)}

        # Initialize a new DataFormatter with source_splitter
        temp_formatter = DataFormatter(source_splitter, class_mapping=class_mapping)
        temp_formatter.create_coco_format(target_dir)
        print("Conversion from YOLO to COCO completed.")

    def coco_to_yolo(self, source_dir: str, target_dir: str, yolo_version: FormatType = FormatType.YOLO_OLD):
        """Convert COCO format to YOLO format"""
        print("Converting COCO format to YOLO format...")
        # Implement reading from source_dir and writing to target_dir
        # This would require parsing COCO annotations and creating YOLO label files

        source_coco_dir = Path(source_dir)
        images_dir = source_coco_dir / "images"
        annotations_dir = source_coco_dir / "annotations"

        # Load COCO annotations
        annotation_files = list(annotations_dir.glob("instances_*.json"))
        if not annotation_files:
            print(f"No annotation files found in {annotations_dir}.")
            return

        for annotation_file in annotation_files:
            try:
                with annotation_file.open('r') as f:
                    coco_data = json.load(f)
            except Exception as e:
                print(f"Error reading {annotation_file}: {e}")
                continue

            images_info = {img['id']: img for img in coco_data.get("images", [])}
            annotations = coco_data.get("annotations", [])

            # Organize annotations by image_id
            annotations_by_image = {}
            for ann in annotations:
                img_id = ann["image_id"]
                annotations_by_image.setdefault(img_id, []).append(ann)

            for img_id, img in images_info.items():
                file_name = img["file_name"]
                image_path = images_dir / file_name
                if not image_path.exists():
                    print(f"Warning: Image file {image_path} does not exist. Skipping.")
                    continue

                label_path = Path(target_dir) / "labels" / image_path.with_suffix('.txt').name
                label_path.parent.mkdir(parents=True, exist_ok=True)

                img_annotations = annotations_by_image.get(img_id, [])
                try:
                    with label_path.open('w') as label_file:
                        for ann in img_annotations:
                            cls_id = ann["category_id"] - 1  # YOLO category_id starts from 0
                            bbox = ann["bbox"]  # [x, y, width, height]
                            x_center = (bbox[0] + bbox[2] / 2) / img["width"]
                            y_center = (bbox[1] + bbox[3] / 2) / img["height"]
                            w = bbox[2] / img["width"]
                            h = bbox[3] / img["height"]
                            label_file.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
                                             )
                except Exception as e:
                    print(f"Error writing to {label_path}: {e}")

            print(f"Converted annotations from {annotation_file} to YOLO format in {target_dir}.")

        # Copy images to YOLO format directory
        if yolo_version == FormatType.YOLO_OLD:
            for split in ["train", "val", "test"]:
                split_images_dir = Path(target_dir) / "images" / split
                split_images_dir.mkdir(parents=True, exist_ok=True)
                # Implement logic to split images accordingly based on your dataset
                # This could involve moving or copying images to respective splits
                # Example placeholder:
                # shutil.copy2(source_image, split_images_dir / source_image.name)
        elif yolo_version == FormatType.YOLO_NEW:
            for split in ["train", "val", "test"]:
                split_images_dir = Path(target_dir) / split / "images"
                split_labels_dir = Path(target_dir) / split / "labels"
                split_images_dir.mkdir(parents=True, exist_ok=True)
                split_labels_dir.mkdir(parents=True, exist_ok=True)
                # Implement logic to split images and labels accordingly based on your dataset
                # Example placeholder:
                # shutil.copy2(source_image, split_images_dir / source_image.name)
                # shutil.copy2(source_label, split_labels_dir / source_label.name)

        print("Conversion from COCO to YOLO completed.")