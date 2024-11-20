import os
from typing import List, Optional, Union
from data_loader import DataLoader

class LabelProcessor:
    """
    A class to process and modify label files within a dataset.

    Attributes:
        labels (List[str]): List of label file paths.
    """

    def __init__(
        self, 
        labels: Optional[List[str]] = None, 
        data_root: str = './', 
        label_extension: str = 'txt'
    ):
        """
        Initialize the LabelProcessor with a list of label files.

        Args:
            labels (List[str]): List of paths to label files. If not provided, 
                                use DataLoader to automatically fetch them.
            data_root (str): Root directory containing the data.
            label_extension (str): Label file extension (default is 'txt').
        """
        if labels is None:
            # Use DataLoader to fetch label files if not provided
            loader = DataLoader(data_root, label_extension=label_extension)
            _, self.labels = loader.get_files()
        else:
            self.labels = labels

        if not self.labels:
            raise ValueError("The list of label files is empty.")

    def change_label(self, from_label: Union[str, int], to_label: Union[str, int], reverse: bool = False):
        """
        Replace all occurrences of from_label with to_label in all label files.

        Args:
            from_label (str): The label to be replaced.
            to_label (str): The label to replace with.
        """
        # First check for None
        if from_label is None:
            raise ValueError("from_label cannot be None.")
        if to_label is None:
            raise ValueError("to_label cannot be None.")
            
        # Convert to strings
        from_label = str(from_label)
        to_label = str(to_label)
        
        # Then check for empty strings
        if from_label.strip() == "":
            raise ValueError("from_label cannot be empty.")
        if to_label.strip() == "":
            raise ValueError("to_label cannot be empty.")
        
        if reverse:
            from_label, to_label = to_label, from_label

        print(f"Replacing '{from_label}' with '{to_label}' in label files...")

        total_files_processed = 0
        total_files_modified = 0

        for label_file in self.labels:
            if not os.path.exists(label_file):
                continue

            try:
                with open(label_file, 'r') as file:
                    lines = file.readlines()

                modified = False
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    current_label = parts[0]
                    if current_label == from_label:
                        parts[0] = to_label
                        modified = True
                    new_line = ' '.join(parts) + '\n'
                    new_lines.append(new_line)

                if modified:
                    with open(label_file, 'w') as file:
                        file.writelines(new_lines)
                    total_files_modified += 1

                total_files_processed += 1

            except Exception as e:
                print(f"Error processing file {label_file}: {e}")

        print(f"Label replacement completed. Processed {total_files_processed} files, modified {total_files_modified} files.")