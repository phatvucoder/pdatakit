# data_manager.py

import os
from typing import Dict, Optional
from pdatakit.pod.data_loader import DataLoader
from pdatakit.pod.data_splitter import DataSplitter
from pdatakit.pod.label_processor import LabelProcessor
from pdatakit.pod.data_formatter import DataFormatter, FormatType
import matplotlib.pyplot as plt

class DataProcessor:
    """
    A unified interface to manage data loading, label processing, data splitting,
    and data formatting for machine learning workflows.
    """

    def __init__(
        self,
        data_root: str = './',
        image_extension: str = 'jpg',
        label_extension: str = 'txt',
        train_size: float = 0.7,
        val_size: float = 0.2,
        test_size: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the DataManager with data paths and splitting configurations.

        Args:
            data_root (str): Root directory containing the dataset.
            image_extension (str): Extension of image files (e.g., 'jpg', 'png').
            label_extension (str): Extension of label files (e.g., 'txt').
            train_size (float): Proportion of data to be used for training.
            val_size (float): Proportion of data to be used for validation.
            test_size (float): Proportion of data to be used for testing.
            random_state (int): Seed for reproducibility.
        """
        self.data_root = data_root
        self.image_extension = image_extension
        self.label_extension = label_extension
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

        # Initialize DataLoader
        self.loader = DataLoader(
            data_root=self.data_root,
            image_extension=self.image_extension,
            label_extension=self.label_extension
        )
        self.images, self.labels = self.loader.get_files()

        # Initialize placeholders for other components
        self.splitter = None
        self.processor = None
        self.formatter = None

    def process_labels(
        self, 
        from_label: Optional[str] = None, 
        to_label: Optional[str] = None, 
        reverse: bool = False
    ):
        """
        Process label files to replace specific labels.

        Args:
            from_label (str, optional): The label to be replaced.
            to_label (str, optional): The label to replace with.
            reverse (bool): If True, swap from_label and to_label.
        """
        if from_label is None and to_label is None:
            print("No label replacement specified. Skipping label processing.")
            return

        if from_label is None or to_label is None:
            raise ValueError("Both from_label and to_label must be provided for label replacement.")

        self.processor = LabelProcessor(labels=self.labels, data_root=self.data_root, label_extension=self.label_extension)
        self.processor.change_label(from_label, to_label, reverse)

        # After label processing, reload labels in case of changes
        self.images, self.labels = self.loader.get_files()

    def split_data(self):
        """
        Split the dataset into training, validation, and testing sets.
        """
        self.splitter = DataSplitter(
            data_root=self.data_root,
            image_extension=self.image_extension,
            label_extension=self.label_extension,
            images=self.images,
            labels=self.labels,
            train_size=self.train_size,
            val_size=self.val_size,
            test_size=self.test_size,
            random_state=self.random_state
        )
        self.train_files, self.val_files, self.test_files = self.splitter.get_splits()
        print("Data splitting completed.")
        print(f"Training samples: {len(self.train_files)}")
        print(f"Validation samples: {len(self.val_files)}")
        print(f"Testing samples: {len(self.test_files)}")

    def visualize_distributions(self):
        """
        Visualize the class distributions across training, validation, and testing sets.
        """
        if self.splitter is None:
            raise RuntimeError("Data has not been split yet. Call split_data() first.")

        self.splitter.visualize_distribution()

    def format_data(
        self, 
        output_dir: str, 
        format_type: FormatType = FormatType.YOLO_NEW, 
        class_mapping: Optional[Dict[int, str]] = None
    ):
        """
        Format the dataset into the specified format.

        Args:
            output_dir (str): Directory where the formatted dataset will be saved.
            format_type (FormatType): Desired format type (YOLO, COCO, VOC).
            class_mapping (dict, optional): Mapping from class indices to class names.
        """
        if self.splitter is None:
            raise RuntimeError("Data has not been split yet. Call split_data() first.")

        self.formatter = DataFormatter(
            data_splitter=self.splitter,
            class_mapping=class_mapping
        )
        self.formatter.create_format(output_dir, format_type)
        print(f"Data formatted into {format_type.value} format and saved to {output_dir}.")

    def save_splits(self, output_dir: str = './splits'):
        """
        Save the train, validation, and test splits to text files.

        Args:
            output_dir (str): Directory where split files will be saved.
        """
        if self.splitter is None:
            raise RuntimeError("Data has not been split yet. Call split_data() first.")

        self.splitter.save_splits(output_dir=output_dir)
        print(f"Data splits saved to {output_dir}.")

    def get_class_distribution(self):
        """
        Retrieve the class distribution across different splits.

        Returns:
            dict: A dictionary containing class distributions for train, val, and test sets.
        """
        if self.splitter is None:
            raise RuntimeError("Data has not been split yet. Call split_data() first.")

        train_dist, val_dist, test_dist = self.splitter.get_class_distribution()
        class_names = self.splitter.all_classes
        distribution = {
            'train': dict(zip(class_names, train_dist)),
            'validation': dict(zip(class_names, val_dist)),
            'test': dict(zip(class_names, test_dist))
        }
        return distribution

    def summary(self):
        """
        Print a summary of the data management process, including counts and class distributions.
        """
        print("===== Data Manager Summary =====")
        print(f"Data Root: {self.data_root}")
        print(f"Image Extension: {self.image_extension}")
        print(f"Label Extension: {self.label_extension}")
        print(f"Total Images: {len(self.images)}")
        print(f"Total Labels: {len(self.labels)}")

        if self.splitter:
            print(f"Training Samples: {len(self.train_files)}")
            print(f"Validation Samples: {len(self.val_files)}")
            print(f"Testing Samples: {len(self.test_files)}")
            distribution = self.get_class_distribution()
            print("Class Distribution in Training Set:")
            for cls, count in distribution['train'].items():
                print(f"  {cls}: {count}")
            print("Class Distribution in Validation Set:")
            for cls, count in distribution['validation'].items():
                print(f"  {cls}: {count}")
            print("Class Distribution in Testing Set:")
            for cls, count in distribution['test'].items():
                print(f"  {cls}: {count}")
        else:
            print("Data has not been split yet.")
        print("=================================")

