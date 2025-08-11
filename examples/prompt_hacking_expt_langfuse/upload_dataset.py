import os
import json
from langfuse import get_client

langfuse_client = get_client()

langfuse_client.create_dataset(
    name="prompt_hacking_dataset",
    # optional description
    description="Prompt hacking dataset",
    # optional metadata
    metadata={
        "author": "Northau",
        "date": "2025-08-10",
        "type": "evaluation"
    }
)

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "sample_dataset.jsonl")

with open(dataset_path, "r") as f:
    for line in f:
        data = json.loads(line)
        langfuse_client.create_dataset_item(
            dataset_name="prompt_hacking_dataset",
            # any python object or value, optional
            input={
                "message": data["message"]
            },
            # any python object or value, optional
            expected_output={
                "is_prompt_hack": data['expected'],
            },
        )
