"""Dataset management for AutoTuning system."""

import json
import yaml
import csv
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib
import random


@dataclass
class DatasetItem:
    """Individual test case within a dataset."""
    
    id: str
    input_data: Dict[str, Any]
    expected_output: Optional[str] = None
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)
    evaluation_method: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetItem":
        """Create from dictionary."""
        return cls(**data)


class Dataset:
    """Collection of test cases for evaluation."""
    
    def __init__(
        self,
        name: str,
        items: List[DatasetItem],
        version: str = "1.0",
        description: Optional[str] = None,
        author: Optional[str] = None,
        tags: List[str] = None,
        evaluation_metrics: List[str] = None,
        evaluation_methods: Dict[str, str] = None,
        train_split: float = 0.7,
        validation_split: float = 0.2,
        test_split: float = 0.1
    ):
        self.name = name
        self.items = items
        self.version = version
        self.description = description
        self.author = author
        self.tags = tags or []
        self.evaluation_metrics = evaluation_metrics or []
        self.evaluation_methods = evaluation_methods or {}
        
        # Validate splits
        if abs((train_split + validation_split + test_split) - 1.0) > 1e-6:
            raise ValueError("Data splits must sum to 1.0")
        
        self.train_split = train_split
        self.validation_split = validation_split
        self.test_split = test_split
        
        # Create splits
        self._create_data_splits()
    
    def _create_data_splits(self) -> None:
        """Create train/validation/test splits."""
        shuffled_items = self.items.copy()
        random.shuffle(shuffled_items)
        
        total_items = len(shuffled_items)
        train_end = int(total_items * self.train_split)
        validation_end = train_end + int(total_items * self.validation_split)
        
        self.train_items = shuffled_items[:train_end]
        self.validation_items = shuffled_items[train_end:validation_end]
        self.test_items = shuffled_items[validation_end:]
    
    def get_items(self, split: str = "all") -> List[DatasetItem]:
        """Return items for specified split."""
        if split == "all":
            return self.items
        elif split == "train":
            return self.train_items
        elif split == "validation":
            return self.validation_items
        elif split == "test":
            return self.test_items
        else:
            raise ValueError(f"Unknown split: {split}. Use 'all', 'train', 'validation', or 'test'")
    
    def get_item_by_id(self, item_id: str) -> Optional[DatasetItem]:
        """Retrieve specific item by ID."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None
    
    def filter_items(self, criteria: Dict[str, Any], split: str = "all") -> List[DatasetItem]:
        """Filter items based on criteria."""
        items = self.get_items(split)
        filtered_items = []
        
        for item in items:
            match = True
            for key, value in criteria.items():
                if key == "tags":
                    # Check if any of the required tags are present
                    item_tags = item.metadata.get("tags", [])
                    if not any(tag in item_tags for tag in value):
                        match = False
                        break
                elif key == "category":
                    if item.metadata.get("category") != value:
                        match = False
                        break
                elif key == "difficulty":
                    if item.metadata.get("difficulty") != value:
                        match = False
                        break
                elif key == "evaluation_method":
                    if item.evaluation_method != value:
                        match = False
                        break
            
            if match:
                filtered_items.append(item)
        
        return filtered_items
    
    def add_item(self, item: DatasetItem) -> None:
        """Add new item to dataset."""
        # Check for duplicate IDs
        if self.get_item_by_id(item.id):
            raise ValueError(f"Item with ID '{item.id}' already exists")
        
        self.items.append(item)
        self._create_data_splits()  # Recreate splits
    
    def remove_item(self, item_id: str) -> bool:
        """Remove item from dataset."""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                del self.items[i]
                self._create_data_splits()  # Recreate splits
                return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_items": len(self.items),
            "train_items": len(self.train_items),
            "validation_items": len(self.validation_items),
            "test_items": len(self.test_items),
            "evaluation_methods": {},
            "categories": {},
            "difficulties": {},
            "tags": {}
        }
        
        for item in self.items:
            # Evaluation methods
            method = item.evaluation_method or "default"
            stats["evaluation_methods"][method] = stats["evaluation_methods"].get(method, 0) + 1
            
            # Categories
            category = item.metadata.get("category", "unknown")
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
            
            # Difficulties
            difficulty = item.metadata.get("difficulty", "unknown")
            stats["difficulties"][difficulty] = stats["difficulties"].get(difficulty, 0) + 1
            
            # Tags
            item_tags = item.metadata.get("tags", [])
            for tag in item_tags:
                stats["tags"][tag] = stats["tags"].get(tag, 0) + 1
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "tags": self.tags,
            "evaluation_metrics": self.evaluation_metrics,
            "evaluation_methods": self.evaluation_methods,
            "train_split": self.train_split,
            "validation_split": self.validation_split,
            "test_split": self.test_split,
            "items": [item.to_dict() for item in self.items]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dataset":
        """Create dataset from dictionary."""
        items_data = data.pop("items", [])
        items = [DatasetItem.from_dict(item_data) for item_data in items_data]
        return cls(items=items, **data)
    
    def save_json(self, filepath: Union[str, Path]) -> None:
        """Save dataset to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def save_yaml(self, filepath: Union[str, Path]) -> None:
        """Save dataset to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def save_csv(self, filepath: Union[str, Path]) -> None:
        """Save dataset to CSV file (simplified format)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "id", "input_data", "expected_output", 
                "evaluation_method", "category", "difficulty"
            ])
            
            # Data rows
            for item in self.items:
                writer.writerow([
                    item.id,
                    item.input_data,
                    item.expected_output or "",
                    item.evaluation_method or "",
                    item.metadata.get("category", ""),
                    item.metadata.get("difficulty", "")
                ])
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "Dataset":
        """Load dataset from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> "Dataset":
        """Load dataset from YAML file."""
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_csv(
        cls, 
        filepath: Union[str, Path],
        name: str,
        version: str = "1.0",
        **kwargs
    ) -> "Dataset":
        """Load dataset from CSV file (simplified format)."""
        filepath = Path(filepath)
        items = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = DatasetItem(
                    id=row["id"],
                    input_message=row["input_message"],
                    expected_output=row.get("expected_output") or None,
                    evaluation_method=row.get("evaluation_method") or None,
                    metadata={
                        "category": row.get("category", ""),
                        "difficulty": row.get("difficulty", "")
                    }
                )
                items.append(item)
        
        return cls(name=name, version=version, items=items, **kwargs)
    
    def get_checksum(self) -> str:
        """Calculate MD5 checksum of dataset content."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def validate(self) -> List[str]:
        """Validate dataset and return list of issues."""
        issues = []
        
        # Check for duplicate IDs
        seen_ids = set()
        for item in self.items:
            if item.id in seen_ids:
                issues.append(f"Duplicate item ID: {item.id}")
            seen_ids.add(item.id)
        
        # Check for empty required fields
        for item in self.items:
            if not item.input_message.strip():
                issues.append(f"Empty input_message for item: {item.id}")
        
        # Check evaluation methods consistency
        for item in self.items:
            if item.evaluation_method and item.evaluation_method not in self.evaluation_methods:
                issues.append(f"Unknown evaluation method '{item.evaluation_method}' for item: {item.id}")
        
        return issues