# Data contract

This document describes the expected dataset contract for downstream retrieval and evaluation work.

## Record types

1. Source document
   - `document_type` is a metadata card or full paper.
   - `source_id` is retained even when a paper and card share the same ID.
   - `review_status` must reflect whether the document was verified or is only a metadata summary.

2. Observation
   - Captures artisan process notes, tool use, material handling, or qualitative descriptions.
   - Must retain original wording and status labels when the source is partial or informal.

3. Question / coverage gap
   - Represents an unanswered or partially answered research item.
   - Must not be promoted to scientific fact without an evidence link and explicit evidence status.

4. Evidence-backed explanation
   - Includes a citation path, URL, and a clear evidence level when known.
   - The source should be evaluated for direct, indirect, or weak relevance before use in retrieval.

## Minimal output schema

```json
{
  "record_id": "WBC-0051-0001",
  "source_id": "WBC-0051",
  "title": "The Influence of Hand Tool Design on Hand Grip Strength: A Review",
  "text": "...",
  "document_type": "Full research paper PDF",
  "source_url": "https://...",
  "local_path": "data/raw/full_papers/...pdf",
  "source_collection": "TRIPURA_BAMBOO_CARVING",
  "craft": "Bamboo",
  "region": "TRIPURA_BAMBOO_CARVING",
  "material": "",
  "process": "Rough shaping and secondary shaping (Battal use)",
  "primary_theme": "7. Posture, Grip and Body Mechanics",
  "evidence_level": "E3",
  "evidence_status": "reviewed metadata / verified source summary",
  "review_status": "Text extracted; scientific claims not independently verified",
  "chunk_index": 1,
  "chunk_count": 3
}
```

## Safe interpretation

- Metadata-card sources are useful for provenance and candidate evidence collection.
- Full research papers are the strongest evidence objects.
- Workbook rows are not necessarily evidence; they may represent unresolved questions, observations, or coverage gaps.
- `Not yet sourced` labels must be preserved, not converted into factual claims.
