#!/usr/bin/env python3
"""
Dataset loader for prompt hacking detection examples.
Supports loading from JSONL files and creating test datasets.
"""

import json
from pathlib import Path
from typing import List, Optional

from northau.autotuning import Dataset, DatasetItem


def load_dataset_from_jsonl(jsonl_path: str, dataset_name: str = "prompt_hacking") -> Dataset:
    """
    Load dataset from JSONL file.
    
    Expected JSONL format:
    {"id": "item_id", "input": "prompt text", "expected": true/false, "type": "hack/legit"}
    """
    items = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                item = DatasetItem(
                    id=data.get("id", f"item_{line_num}"),
                    input_data={"message": data["message"]},
                    expected_output=json.dumps({"is_prompt_hack": data["expected"]}),
                    evaluation_criteria={"correct_classification": True},
                    metadata={
                        "expected": data["expected"],
                        "type": data.get("type", "unknown"),
                        "line_number": line_num
                    }
                )
                items.append(item)
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping invalid line {line_num}: {e}")
                continue
    
    return Dataset(
        name=dataset_name,
        items=items,
        version="1.0",
        description=f"Dataset loaded from {jsonl_path}",
        author="Auto-generated",
        tags=["security", "classification"],
        evaluation_metrics=["accuracy"],
        evaluation_methods={"default": "classification"}
    )
