import os
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from collections import defaultdict
from pdatakit.pod.data_loader import DataLoader
import matplotlib.pyplot as plt

class DataSplitter:
    def __init__(
        self,
        data_root: str = './',
        image_extension: str = 'jpg',
        label_extension: str = 'txt',
        images: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        train_size: float = 0.7,
        val_size: float = 0.2,
        test_size: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize DataSplitter. If images and labels are not provided, 
        automatically use DataLoader to fetch them.
        
        Args:
            data_root: Root directory containing the data.
            image_extension: Image file extension.
            label_extension: Label file extension.
            images: List of image files.
            labels: List of label files.
            train_size: Training data ratio.
            val_size: Validation data ratio.
            test_size: Test data ratio.
            random_state: Seed for data splitting.
        """
        if images is None or labels is None:
            self.loader = DataLoader(data_root, image_extension, label_extension)
            self.images, self.labels = self.loader.get_files()
        else:
            self.images = images
            self.labels = labels

        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError("Sum of train_size, val_size, and test_size must be 1.0")

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

        # Process labels
        self.image_labels = self._process_labels()
        self.all_classes = sorted(set(cls for labels in self.image_labels for cls in labels))
        self.mlb = MultiLabelBinarizer(classes=self.all_classes)
        self.Y = self.mlb.fit_transform(self.image_labels)

        # Split data
        self.train_files, self.val_files, self.test_files = self._split_data()

    def _process_labels(self) -> List[List[str]]:
        """
        Process label files to get class list for each image.
        
        Returns:
            List of class lists for each image.
        """
        image_labels = []
        for label_file in self.labels:
            classes_in_image = set()
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = parts[0]
                            classes_in_image.add(class_id)
            image_labels.append(list(classes_in_image))
        return image_labels

    def _split_data(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Split data into train, validation, and test sets.
        
        Returns:
            Tuple of (train_files, val_files, test_files)
        """
        # First split into train and temp (val + test)
        msss1 = MultilabelStratifiedShuffleSplit(
            n_splits=1, 
            test_size=(self.val_size + self.test_size), 
            random_state=self.random_state
        )
        train_idx, temp_idx = next(msss1.split(self.images, self.Y))
        
        # Calculate test ratio in temp
        test_ratio = self.test_size / (self.val_size + self.test_size)
        
        # Split temp into validation and test
        msss2 = MultilabelStratifiedShuffleSplit(
            n_splits=1, 
            test_size=test_ratio, 
            random_state=self.random_state
        )
        val_idx, test_idx = next(msss2.split(
            np.array(self.images)[temp_idx], 
            self.Y[temp_idx]
        ))
        
        # Get file lists
        train_files = [self.images[i] for i in train_idx]
        val_files = [self.images[temp_idx[i]] for i in val_idx]
        test_files = [self.images[temp_idx[i]] for i in test_idx]
        
        return train_files, val_files, test_files

    def get_splits(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Return train, validation and test sets.
        
        Returns:
            Tuple of (train_files, val_files, test_files)
        """
        return self.train_files, self.val_files, self.test_files

    def get_class_distribution(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Get class distribution in each dataset.
        
        Returns:
            Tuple of (train_distribution, val_distribution, test_distribution)
        """
        train_dist = self._get_class_distribution(self.train_files)
        val_dist = self._get_class_distribution(self.val_files)
        test_dist = self._get_class_distribution(self.test_files)
        return train_dist, val_dist, test_dist

    def _get_class_distribution(self, file_list: List[str]) -> List[int]:
        """
        Calculate class distribution in a specific file list.
        
        Args:
            file_list: List of image files.
            
        Returns:
            List of counts for each class.
        """
        class_count = defaultdict(int)
        for file in file_list:
            label_path = file.replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = parts[0]
                            class_count[class_id] += 1
        return [class_count.get(cls, 0) for cls in self.all_classes]

    def save_splits(self, output_dir: str = './splits'):
        """
        Save train, validation and test sets to txt files.
        
        Args:
            output_dir: Directory to save split files.
        """
        os.makedirs(output_dir, exist_ok=True)
        self._save_split(self.train_files, 'train', output_dir)
        self._save_split(self.val_files, 'val', output_dir)
        self._save_split(self.test_files, 'test', output_dir)

    def _save_split(self, file_list: List[str], split_name: str, output_dir: str):
        """
        Save a dataset split to txt file.
        
        Args:
            file_list: List of image files.
            split_name: Name of the split (train, val, test).
            output_dir: Directory to save the file.
        """
        split_path = os.path.join(output_dir, f'{split_name}.txt')
        with open(split_path, 'w') as f:
            for file in file_list:
                f.write(os.path.abspath(file) + '\n')
        print(f'Saved {split_name} split to {split_path}')
    
    def visualize_distribution(self):
        """
        Trực quan hóa phân bố các lớp trong các tập train, validation và test.
        """
        train_dist, val_dist, test_dist = self.get_class_distribution()
        all_classes = self.all_classes

        x = np.arange(len(all_classes))
        width = 0.25

        plt.figure(figsize=(20, 10))
        plt.bar(x - width, train_dist, width, label='Train')
        plt.bar(x, val_dist, width, label='Validation')
        plt.bar(x + width, test_dist, width, label='Test')
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.title('Class Distribution in Train, Validation, and Test Sets')
        plt.xticks(x, all_classes, rotation='vertical')
        plt.legend()
        plt.tight_layout()
        plt.show()