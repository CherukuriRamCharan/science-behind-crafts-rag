"""Build a unified catalog for craft metadata cards and full research papers.

The script reads existing inventories and registries. It does not modify source
PDFs, DOCX files, JSONL data, or the workbooks. Repeated source IDs are kept as
separate document records when they refer to different local files.
"""

import json
import re
from pathlib import Path

import pandas as pd
from docx import Document

ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_FILE = ROOT / "data" / "extracted" / "documents.jsonl"
SOURCE_INVENTORY = ROOT / "data" / "metadata" / "source_inventory.csv"
TOOL_INVENTORY = ROOT / "data" / "metadata" / "tool_data_inventory.csv"
PAPER_REGISTRY = ROOT / "data" / "metadata" / "full_paper_registry.csv"
CATALOG_FILE = ROOT / "data" / "metadata" / "source_catalog.csv"

COLUMNS = [
    "source_id",
    "title",
    "authors",
    "year",
    "doi",
    "source_url",
    "drive_path",
    "local_path",
    "document_type",
    "craft",
    "region",
    "material",
    "process",
    "review_status",
]


def clean(value):
    """Convert pandas values to clean strings, keeping missing values blank."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalise_path(value):
    """Normalise path separators for matching CSV paths to JSONL paths."""
    return str(value).replace("\\", "/").strip().lower()


def get_section(text, start_pattern, end_pattern):
    """Extract text between two labels in a metadata card."""
    pattern = start_pattern + r"\s*(.*?)\s+" + end_pattern
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def extract_metadata(text):
    """Extract fields that are explicitly present in a metadata card."""
    source_match = re.search(
        r"\bSource ID\s+(WBC-[A-Za-z0-9-]+)",
        text,
        flags=re.IGNORECASE,
    )

    doi_match = re.search(r"\b10\.\d{4,9}/[^\s,;<>]+", text)
    doi = doi_match.group(0).rstrip(".,)") if doi_match else ""

    link_match = re.search(
        r"\bLink(?:\s*/\s*File)?\s*:?\s*(.*?)\s+Craft\b",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    link_text = (
        re.sub(r"\s+", " ", link_match.group(1)).strip()
        if link_match
        else ""
    )

    url_match = re.search(r"https?://[^\s]+", link_text)

    if url_match:
        source_url = url_match.group(0).rstrip(".,)")
    else:
        pmid_match = re.search(
            r"\bPMID\s*:\s*(\d+)",
            link_text,
            flags=re.IGNORECASE,
        )

        if pmid_match:
            source_url = (
                f"https://pubmed.ncbi.nlm.nih.gov/{pmid_match.group(1)}/"
            )
        elif doi:
            source_url = f"https://doi.org/{doi}"
        else:
            source_url = ""


    return {
        "source_id": source_match.group(1) if source_match else "",
        "title": get_section(text, r"\bTitle\b", r"Source type"),
        "doi": doi,
        "source_url": source_url,
        "source_type": get_section(
            text, r"Source\s+type", r"Link\s*/\s*File"
        ),
        "craft": get_section(
            text,
            r"Craft\s+applicability",
            r"Primary\s+Why[-\s]*theme",
        ),
        "process": get_section(
            text,
            r"Process\s+stage",
            r"Questions\s+answered\s*/\s*generated",
        ),
    }


def make_row(**values):
    """Create a catalog row with all expected columns."""
    row = {column: "" for column in COLUMNS}
    row.update(values)
    return row


def load_extracted_documents():
    """Index the JSONL extraction by normalised relative path."""
    documents = {}
    with DOCUMENTS_FILE.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            key = normalise_path(item.get("relative_path", ""))
            documents[key] = item
    return documents


def read_docx_text(path):
    """Extract paragraphs and table cells from a DOCX metadata card."""
    doc = Document(str(path))
    parts = [paragraph.text for paragraph in doc.paragraphs]

    for table in doc.tables:
        for row in table.rows:
            parts.append(" ".join(cell.text for cell in row.cells))

    return "\n".join(parts)


def resolve_docx_path(raw_path):
    """Resolve inventory paths, including paths relative to the Tool Data folder.

    The inventory stores paths like ``Tool Data\\...\\WBC-2019.docx`` while
    the files in this repository are under ``data/raw/teammate_data/``.
    """
    raw_path = clean(raw_path)
    if not raw_path:
        return None

    path = Path(raw_path)
    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        # First support paths already relative to the repository root.
        candidates.append(ROOT / path)
        # Then support paths relative to the shared teammate-data folder.
        candidates.append(ROOT / "data" / "raw" / "teammate_data" / path)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    # Last-resort lookup by filename within teammate_data. Only accept a unique
    # match; guessing between multiple files would risk linking the wrong source.
    filename = path.name
    if filename:
        search_root = ROOT / "data" / "raw" / "teammate_data"
        matches = list(search_root.rglob(filename)) if search_root.exists() else []
        if len(matches) == 1 and matches[0].is_file():
            return matches[0].resolve()

    return None


def relative_to_root(path):
    """Return a repository-relative path where possible."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def validate_catalog(records):
    """Check the generated catalog for required fields and missing IDs."""
    required_columns = set(COLUMNS)
    for record in records:
        missing = sorted(required_columns - record.keys())
        if missing:
            raise ValueError(f"Catalog row missing required columns: {missing}")
        if record.get("source_id", "") and "WBC-" not in record["source_id"]:
            raise ValueError(f"Unexpected source ID format: {record['source_id']!r}")
    return records


def build_catalog():
    """Build the unified source catalog and return the DataFrame."""
    records = []

    # 1. Original collection: 66 PDF metadata cards.
    inventory = pd.read_csv(SOURCE_INVENTORY)
    documents = load_extracted_documents()

    for _, item in inventory.iterrows():
        relative_path = clean(item.get("relative_path", ""))
        document = documents.get(normalise_path(relative_path), {})
        text = document.get("text", "")
        fields = extract_metadata(text)

        # Fall back to the filename for the ID only if the card text lacks one.
        filename = clean(item.get("file_name", ""))
        filename_id = re.search(r"\b(WBC-[A-Za-z0-9-]+)\b", filename)
        source_id = fields["source_id"] or (
            filename_id.group(1) if filename_id else ""
        )
        title = fields["title"] or Path(filename or relative_path).stem

        records.append(
            make_row(
                source_id=source_id,
                title=title,
                doi=fields["doi"],
                source_url=fields["source_url"],
                local_path=relative_path.replace("\\", "/"),
                document_type="PDF metadata card",
                craft=fields["craft"],
                region=clean(item.get("state_folder", "")),
                process=fields["process"],
                review_status=(
                    "Text extracted; scientific claims not independently verified"
                ),
            )
        )

    # 2. Tool Data: 32 DOCX metadata cards.
    tool_inventory = pd.read_csv(TOOL_INVENTORY)
    docx_read_successes = 0
    docx_read_failures = 0

    for _, item in tool_inventory.iterrows():
        raw_path = clean(item.get("original_docx_path", ""))
        resolved_path = resolve_docx_path(raw_path)
        text = ""

        if resolved_path is None:
            docx_read_failures += 1
            print(f"Warning: DOCX path not found: {raw_path}")
        else:
            try:
                text = read_docx_text(resolved_path)
                docx_read_successes += 1
            except Exception as exc:
                docx_read_failures += 1
                print(f"Warning: could not read DOCX {resolved_path}: {exc}")

        fields = extract_metadata(text)
        fallback_id = clean(item.get("source_id", ""))
        fallback_title = clean(item.get("title", ""))
        fallback_link = clean(item.get("link", ""))

        records.append(
            make_row(
                source_id=fields["source_id"] or fallback_id,
                title=fields["title"] or fallback_title,
                doi=fields["doi"],
                source_url=fields["source_url"] or fallback_link,
                local_path=(
                    relative_to_root(resolved_path)
                    if resolved_path is not None
                    else raw_path.replace("\\", "/")
                ),
                document_type="DOCX metadata card",
                craft=fields["craft"],
                process=fields["process"],
                review_status=(
                    "Metadata card; scientific claims not independently verified; "
                    "see tool_data_link_check.csv"
                ),
            )
        )

    # 3. Full research papers: use the existing registry.
    papers = pd.read_csv(PAPER_REGISTRY)
    for _, item in papers.iterrows():
        records.append(
            make_row(
                source_id=clean(item.get("source_id", "")),
                title=clean(item.get("paper_title", "")),
                doi=clean(item.get("doi", "")),
                source_url=clean(item.get("source_url", "")),
                local_path=clean(item.get("local_pdf_path", "")).replace("\\", "/"),
                document_type="Full research paper PDF",
                craft=clean(item.get("craft_applicability", "")),
                process=clean(item.get("process_stage", "")),
                review_status=clean(item.get("extraction_status", "")),
            )
        )

    catalog = pd.DataFrame(records, columns=COLUMNS)
    validate_catalog(catalog.to_dict(orient="records"))
    CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(CATALOG_FILE, index=False, encoding="utf-8")

    print("Source catalog generated.")
    print("Total document records:", len(catalog))
    print(
        "PDF metadata cards:",
        (catalog["document_type"] == "PDF metadata card").sum(),
    )
    print(
        "DOCX metadata cards:",
        (catalog["document_type"] == "DOCX metadata card").sum(),
    )
    print(
        "Full research papers:",
        (catalog["document_type"] == "Full research paper PDF").sum(),
    )
    print("DOCX files read successfully:", docx_read_successes)
    print("DOCX files with read/path errors:", docx_read_failures)
    print("Records without source IDs:", (catalog["source_id"] == "").sum())
    print("Saved to:", CATALOG_FILE)
    return catalog


def main():
    build_catalog()


if __name__ == "__main__":
    main()
