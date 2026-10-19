"""Load seed records and Field/Entry cards or raw source files."""
from __future__ import annotations
import logging
import os
import re
from pathlib import Path
from typing import Any
from .common import clean_text, normalize_record, read_json

LABELS = {
 "Source ID":"source_id", "Title":"title", "Source type":"source_type", "Link / File":"link",
 "Craft applicability":"craft_applicability", "Primary Why-theme":"why_theme",
 "Secondary Why-theme":"secondary_theme", "Process stage":"process_stage",
 "Questions answered / generated":"questions_answered", "Evidence level":"evidence_level",
 "Grade":"grade", "Relevance note":"relevance_note", "Species/material":"species_material",
 "Tool/treatment":"tool_treatment", "Pages/timestamps":"pages_timestamps", "Caution/limitation":"caution"
}
THEMES={1:"1. Material selection",2:"2. Harvesting, procurement and preparation",3:"3. Water, moisture, heating and conditioning",
 4:"4. Cutting, splitting and initial shaping",5:"5. Fine carving, piercing and decorative work",6:"6. Tools and tool-material interaction",
 7:"7. Posture, grip and body mechanics",8:"8. Duration, timing and sequence",9:"9. Surface treatment, finishing and preservation",
 10:"10. Failure, repair and quality",11:"11. Product function, use and performance",12:"12. Sustainability, mechanisation and craft integrity"}

def _cluster_name(path: Path, root: Path, raw_source: bool=False) -> str:
    rel=path.relative_to(root)
    parts=rel.parts
    name=parts[-3] if raw_source and len(parts)>2 else (parts[-2] if len(parts)>1 else path.parent.name)
    name=re.sub(r"^\d+[_-]*","",name).replace("_"," ").replace("-"," ")
    return clean_text(name)

def _looks_like_card(text: str) -> bool:
    folded=text.casefold()
    return all(label in folded for label in ("source id","title","evidence level")) and ("field" in folded and "entry" in folded)

def _registry_title_matches(stem: str, title: str) -> bool:
    def words(value):
        return {w for w in re.findall(r"[a-z0-9]+",value.casefold()) if len(w)>2 and w not in {"the","and","for","with","from","its","use","wood","bamboo"}}
    file_words=words(re.sub(r"WBC[-_]?\d+","",stem,flags=re.I).replace("_"," "))
    title_words=words(str(title or ""))
    if not file_words or not title_words: return False
    return len(file_words & title_words)/min(len(file_words),len(title_words))>=0.65

def _merge_duplicate_cards(cards):
    merged={}; merged_copies=[]
    for row in cards:
        title=str(row.get("metadata",{}).get("title", ""))
        title_key=" ".join(re.findall(r"[a-z0-9]+",title.casefold()))
        key=(row["source_id"],title_key)
        if key not in merged:
            item=dict(row); item["metadata"]=dict(row.get("metadata",{}))
            item["metadata"]["source_paths"]=[item["metadata"].get("source_path","")] if item["metadata"].get("source_path") else []
            merged[key]=item; continue
        prior=merged[key]; paths=prior["metadata"].setdefault("source_paths",[])
        source_path=row.get("metadata",{}).get("source_path","")
        if source_path and source_path not in paths: paths.append(source_path)
        if row.get("text_chunk") and row["text_chunk"] not in prior["text_chunk"]:
            prior["text_chunk"]=clean_text(prior["text_chunk"]+" "+row["text_chunk"])
        for field in ("questions_answered",):
            prior[field]=list(dict.fromkeys(prior.get(field,[])+row.get(field,[])))
        for field in ("why_theme","secondary_theme"):
            values=list(dict.fromkeys(x for x in (prior.get(field),row.get(field)) if x))
            prior[field]="; ".join(values)
        clusters=list(dict.fromkeys(x for x in (prior.get("craft_cluster"),row.get("craft_cluster")) if x))
        prior["craft_cluster"]="; ".join(clusters)
        merged_copies.append({"source_id":row["source_id"],"title":title,"duplicate_path":source_path})
    return list(merged.values()),merged_copies

def _parse_card(text: str, path: Path, root: Path | None = None) -> dict[str, Any]:
    lines = [clean_text(x) for x in text.splitlines() if clean_text(x)]
    fields: dict[str, str] = {}; i = 0
    while i < len(lines):
        line = lines[i].strip().rstrip(":")
        if line in LABELS and i + 1 < len(lines):
            fields[LABELS[line]] = lines[i + 1]; i += 2
        else: i += 1
    # PDF extraction can flatten table cells and wrap labels across line endings.
    # Match a punctuation/whitespace-free view while slicing values from original text.
    normalized=[]; offsets=[]
    for offset,char in enumerate(text):
        if char.isalnum(): normalized.append(char.casefold()); offsets.append(offset)
    flat="".join(normalized); found=[]
    for label,key in LABELS.items():
        needle="".join(ch.casefold() for ch in label if ch.isalnum())
        at=flat.find(needle)
        if at>=0: found.append((at,offsets[at],offsets[at+len(needle)-1]+1,key))
    found.sort()
    for j,(_,_,value_start,key) in enumerate(found):
        next_start=found[j+1][1] if j+1<len(found) else len(text)
        value=clean_text(text[value_start:next_start]).strip(" :\t\n-")
        if value and (key not in fields or len(value)>len(fields[key])): fields[key]=value
    sid = fields.get("source_id") or path.stem
    questions = [x.strip() for x in re.split(r"[?;]", fields.get("questions_answered", "")) if x.strip()]
    body = ". ".join(x for x in (fields.get("title"),
        ("Questions answered: " + fields.get("questions_answered", "")) if fields.get("questions_answered") else "",
        ("Why theme: " + fields.get("why_theme", "")) if fields.get("why_theme") else "",
        ("Process stage: " + fields.get("process_stage", "")) if fields.get("process_stage") else "",
        fields.get("relevance_note")) if x)
    return normalize_record({
      "source_id":sid,"title":fields.get("title",path.stem),"source_type":fields.get("source_type","Field/Entry annotation card"),
      "craft_cluster":fields.get("craft_cluster") or (_cluster_name(path,root,raw_source=path.suffix.casefold()==".pdf") if root else path.parent.name),"craft_applicability":fields.get("craft_applicability","Both"),
      "why_theme":fields.get("why_theme",""),"secondary_theme":fields.get("secondary_theme",""),"process_stage":fields.get("process_stage",""),
      "evidence_level":fields.get("evidence_level","E5"),"grade":fields.get("grade","C"),"questions_answered":questions,
      "text_chunk":body,"species_material":fields.get("species_material",""),"tool_treatment":fields.get("tool_treatment",""),
      "link":fields.get("link",""),"caution":fields.get("caution",fields.get("relevance_note","Field/Entry card does not state a separate caution.")),"pages_timestamps":fields.get("pages_timestamps",""),"annotation_path":path.relative_to(root).as_posix() if root else path.name,
      "craft_applicability":fields.get("craft_applicability","Both")})

def _extract(path: Path) -> str:
    ext = path.suffix.casefold()
    if ext in (".txt", ".html", ".htm"):
        data = path.read_text(encoding="utf-8", errors="ignore")
        if ext != ".txt":
            try:
                from bs4 import BeautifulSoup
                data = BeautifulSoup(data, "html.parser").get_text(" ")
            except ImportError: data = re.sub(r"<[^>]+>", " ", data)
        return data
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        except Exception as exc:
            logging.warning("PDF extraction failed for %s: %s",path,exc); return ""
    if ext == ".docx":
        try:
            from docx import Document
            doc = Document(str(path)); parts = [p.text for p in doc.paragraphs]
            for table in doc.tables:
                for row in table.rows:
                    parts.append("\n".join(cell.text for cell in row.cells))
            return "\n".join(parts)
        except Exception as exc:
            logging.warning("DOCX extraction failed for %s: %s",path,exc); return ""
    return ""

def load_seed(path: str | Path) -> list[dict[str, Any]]:
    data = read_json(path)
    return [normalize_record(row) for row in data]

def load_gap_analysis(path: str | Path):
    """Read the provided source register and its hand-maintained why-question map."""
    from openpyxl import load_workbook
    book=load_workbook(path,read_only=True,data_only=True)
    registry={}; questions=[]
    for sheet in book.worksheets:
        rows=sheet.iter_rows(values_only=True)
        headers=[clean_text(str(v or "")).casefold() for v in next(rows,())]
        positions={h:i for i,h in enumerate(headers)}
        if {"id","source / title","evidence"}.issubset(positions):
            for values in rows:
                sid=clean_text(str(values[positions["id"]] or "")).upper()
                if not sid: continue
                theme_raw=values[positions.get("primary theme",-1)] if "primary theme" in positions else ""
                themes=[]
                for t in re.findall(r"\d+",str(theme_raw or "")):
                    if int(t) in THEMES: themes.append(THEMES[int(t)])
                registry[sid]={"title":values[positions.get("source / title",-1)] or sid,
                    "source_type":values[positions.get("source type",-1)] or "",
                    "craft_applicability":values[positions.get("craft application",-1)] or "Both",
                    "why_theme":themes[0] if themes else "",
                    "secondary_theme":themes[1] if len(themes)>1 else "",
                    "evidence_level":values[positions.get("evidence",-1)] or "E5",
                    "link":values[positions.get("url",-1)] or "","grade":"C",
                    "caution":"Metadata from the supplied source register; grade not listed."}
        elif {"why question","sources currently answering"}.issubset(positions):
            for values in rows:
                question=clean_text(str(values[positions["why question"]] or ""))
                raw=clean_text(str(values[positions["sources currently answering"]] or ""))
                ids=re.findall(r"WBC[-_]\d+",raw,flags=re.I)
                if question and ids: questions.append({"query":question,"relevant_source_ids":[x.replace("_","-").upper() for x in ids]})
    book.close()
    return registry,questions

def load_real_sources(root: str | Path, registry_path: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(root).expanduser().resolve()
    if not root.exists(): raise FileNotFoundError(f"Real data directory not found: {root}")
    paths=[p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.casefold() in {".docx", ".pdf", ".html", ".htm", ".txt"}]
    cards=[]; card_paths=set()
    for path in paths:
        if path.suffix.casefold() not in {".docx",".pdf"}: continue
        text=_extract(path)
        if _looks_like_card(text):
            row=_parse_card(text,path,root); row["metadata"]={**row.get("metadata",{}),"source_path":path.relative_to(root).as_posix()}
            cards.append(row); card_paths.add(path)
    registry={}; workbook_questions=[]
    if registry_path and Path(registry_path).exists(): registry,workbook_questions=load_gap_analysis(registry_path)
    sources=[]; consumed=set(); registry_matches=[]; registry_conflicts=[]; empty_files=[]
    for path in paths:
        if path in card_paths: continue
        if path.suffix.casefold()==".docx": continue
        text=_extract(path)
        if not text.strip(): empty_files.append(str(path)); continue
        stem=path.stem; match=re.search(r"(?:WBC[-_]?\d+)",stem,re.I)
        sid=match.group(0).replace("_","-").upper() if match else f"FILE-{len(sources)+1:04d}"
        reg=registry.get(sid,{})
        title_match=bool(reg and _registry_title_matches(stem,str(reg.get("title",""))))
        if reg and title_match: registry_matches.append(sid)
        elif reg: registry_conflicts.append({"source_id":sid,"file_title":stem,"registry_title":str(reg.get("title",""))})
        if not title_match: reg={}
        row=normalize_record({"source_id":sid,"title":reg.get("title",stem),"source_type":reg.get("source_type","raw source"),
             "craft_cluster":_cluster_name(path,root,raw_source=True),"craft_applicability":reg.get("craft_applicability","Both"),
             "why_theme":reg.get("why_theme",""),"secondary_theme":reg.get("secondary_theme",""),
             "evidence_level":reg.get("evidence_level","E5"),"grade":reg.get("grade","C"),"text_chunk":text,
             "link":reg.get("link",""),"caution":reg.get("caution","Raw source; metadata requires annotation.")})
        row["metadata"]={**row.get("metadata",{}),"source_path":path.relative_to(root).as_posix(),"registry_title_matched":title_match}; sources.append(row)
    cards,merged_copies=_merge_duplicate_cards(cards)
    sources.extend(cards)
    questions={}
    for row in cards:
        for question in row.get("questions_answered",[]):
            q=clean_text(str(question)).rstrip("?")
            if q: questions.setdefault(q,{"query":q,"relevant_source_ids":[]})["relevant_source_ids"].append(row["source_id"])
    matched_ids=set(registry_matches)
    compatible_gap_questions=0
    for item in workbook_questions:
        relevant=[sid for sid in item["relevant_source_ids"] if sid in matched_ids]
        if relevant:
            q=item["query"]
            questions.setdefault(q,{"query":q,"relevant_source_ids":[]})["relevant_source_ids"].extend(relevant)
            compatible_gap_questions+=1
    real_eval=[{"query":v["query"],"relevant_source_ids":sorted(set(v["relevant_source_ids"]))} for v in questions.values()]
    from collections import Counter
    id_counts=Counter(row["source_id"] for row in cards)
    try: manifest_root=os.path.relpath(root,Path.cwd())
    except ValueError: manifest_root="configured real-data directory"
    manifest={"root":manifest_root,"files_scanned":len(paths),"field_entry_card_files":len(card_paths),"field_entry_cards":len(cards),
      "raw_text_documents":len(sources)-len(cards),"indexed_records":len(sources),"empty_or_unreadable_files":empty_files,
      "duplicate_copies_merged":merged_copies,
      "duplicate_card_source_ids":{sid:count for sid,count in id_counts.items() if count>1},
      "registry_title_matches":sorted(set(registry_matches)),"registry_id_title_conflicts":registry_conflicts,
      "gap_workbook_evaluation_queries_compatible":compatible_gap_questions,
      "real_eval_queries":len(real_eval)}
    return sources,real_eval,manifest

def ingest(
    config: dict,
    include_real: bool = False,
    include_seed: bool | None = None,
) -> list[dict[str, Any]]:
    data_config = config["data"]
    if data_config.get("prepared_root"):
        if include_seed:
            raise ValueError(
                "Synthetic seed records are not part of the prepared-corpus configuration. "
                "Use a standalone prototype config to opt into a seed corpus."
            )
        from .prepared_data import load_prepared_corpus

        records, _ = load_prepared_corpus(config)
        return records

    use_seed = True if include_seed is None else include_seed
    rows = load_seed(data_config["seed_file"]) if use_seed else []
    if include_real or config["data"].get("include_real_by_default"):
        real, _, _ = load_real_sources(
            data_config["real_data_dir"],
            data_config.get("source_registry_file"),
        )
        rows.extend(real)
    return rows
