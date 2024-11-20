import os
from typing import List, Tuple

class DataLoader:
    def __init__(self, data_root: str = './', image_extension: str = 'jpg', label_extension: str = 'txt'):
        if not os.path.isdir(data_root):
            raise ValueError(f"Invalid directory: {data_root}")
        
        self.data_root = data_root
        self.image_extension = image_extension.lower()
        self.label_extension = label_extension.lower()

    def _list_files(self, extension: str) -> List[str]:
        """
        List all files in the directory and subdirectories with the given extension.
        """
        files = []
        for root, _, filenames in os.walk(self.data_root):
            for filename in filenames:
                if filename.lower().endswith(f'.{extension}'):
                    files.append(os.path.join(root, filename))
        
        if not files:
            raise ValueError(f"No files found with extension '.{extension}' in {self.data_root}")

        return sorted(files)

    def get_images(self) -> List[str]:
        """
        Get all image files with the specified extension.
        """
        return self._list_files(self.image_extension)

    def get_labels(self) -> List[str]:
        """
        Get all label files with the specified extension.
        """
        return self._list_files(self.label_extension)

    def get_files(self) -> Tuple[List[str], List[str]]:
        """
        Get both image and label files, ensuring matching counts.
        """
        images = self.get_images()
        labels = self.get_labels()

        if len(images) != len(labels):
            raise ValueError(
                f"Count mismatch: found {len(images)} images and {len(labels)} labels in {self.data_root}."
            )

        return images, labels