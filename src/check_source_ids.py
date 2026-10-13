
import json
import re
from collections import Counter
from pathlib import Path

input_file = Path("data/extracted/documents.jsonl")

rows = [
    json.loads(line)
    for line in input_file.open(encoding="utf-8")
]

source_ids = []
missing_ids = []

for row in rows:
    text = row.get("text", "")
    match = re.search(r"Source ID\s+(WBC-[A-Za-z0-9-]+)", text)

    if match:
        source_ids.append(
            (match.group(1), row.get("file_name", ""))
        )
    else:
        missing_ids.append(row.get("file_name", ""))

counts = Counter(source_id for source_id, _ in source_ids)

print("Total cards:", len(rows))
print("Cards with source IDs:", len(source_ids))
print("Cards missing source IDs:", len(missing_ids))

print("\nRepeated source IDs:")
found_duplicates = False

for source_id, count in counts.items():
    if count > 1:
        found_duplicates = True
        files = [
            filename
            for sid, filename in source_ids
            if sid == source_id
        ]
        print(f"{source_id}: {files}")

if not found_duplicates:
    print("None")

print("\nCards missing source IDs:")
for filename in missing_ids:
    print(filename)
