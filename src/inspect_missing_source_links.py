
import json
import re
from pathlib import Path

input_file = Path("data/extracted/documents.jsonl")
target_ids = {"WBC-0012", "WBC-0017", "WBC-0018", "WBC-0016"}

with input_file.open(encoding="utf-8") as file:
    for line in file:
        row = json.loads(line)
        text = row.get("text", "")

        match = re.search(
            r"\bSource ID\s+(WBC-[A-Za-z0-9-]+)",
            text,
        )

        if match and match.group(1) in target_ids:
            print("\n" + "=" * 70)
            print("SOURCE ID:", match.group(1))
            print("FILE:", row.get("file_name", ""))
            print(text[:2500])
