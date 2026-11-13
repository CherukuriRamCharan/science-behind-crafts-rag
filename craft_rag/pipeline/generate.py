"""Evidence-only answer construction and optional local Transformers backend."""
from __future__ import annotations

import logging
import json
import math
import re

from .common import STOP_WORDS, is_evidence_eligible, tokens


def _is_evidence(passage):
    return is_evidence_eligible(passage)


def _resolve_chunk_id(passage: dict) -> str:
    """Resolve a stable chunk identifier from a passage dictionary."""
    if not isinstance(passage, dict):
        return ""
    chunk_id = passage.get("chunk_id") or passage.get("record_id")
    if chunk_id:
        return str(chunk_id)
    metadata = passage.get("metadata")
    if isinstance(metadata, dict):
        if metadata.get("chunk_id"):
            return str(metadata["chunk_id"])
        if metadata.get("record_id"):
            return str(metadata["record_id"])
        prov = metadata.get("provenance")
        if isinstance(prov, dict):
            if prov.get("chunk_id"):
                return str(prov["chunk_id"])
            if prov.get("record_id"):
                return str(prov["record_id"])
            if prov.get("document_key"):
                return str(prov["document_key"])
    return ""


def _clean_display_sentence(sentence: str) -> str:
    """Remove page labels that belong to card layout, not source prose."""
    return re.sub(r"^\s*\[Page\s+\d+\]\s*", "", sentence, flags=re.IGNORECASE).strip()


def _citation_support(answer_sentence: str, passage: dict) -> tuple[float, str]:
    """Return content-token overlap and the best source sentence supporting a claim."""
    answer_tokens = [token for token in tokens(answer_sentence) if token not in STOP_WORDS]
    if not answer_tokens:
        return 0.0, ""
    best_ratio, best_span = 0.0, ""
    text = str(passage.get("text_chunk", ""))
    for source_sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", text):
        source_sentence = _clean_display_sentence(source_sentence)
        source_tokens = [token for token in tokens(source_sentence) if token not in STOP_WORDS]
        if not source_tokens:
            continue
        ratio = len(set(answer_tokens) & set(source_tokens)) / len(set(answer_tokens))
        contiguous = any(
            answer_tokens[index:index + 6] == source_tokens[start:start + 6]
            for index in range(max(0, len(answer_tokens) - 5))
            for start in range(max(0, len(source_tokens) - 5))
        ) if len(answer_tokens) >= 6 else False
        if ratio > best_ratio:
            best_ratio, best_span = ratio, source_sentence
        if contiguous:
            return max(ratio, 0.5), source_sentence
    return best_ratio, best_span if best_ratio >= 0.5 else ""


def _answer_sentences(answer: str) -> list[str]:
    answer = re.sub(r"\s*\[[^\]]+\s*\|\s*E[1-5]\]", "", answer)
    return [_clean_display_sentence(part) for part in re.split(r"(?<=[.!?])\s+", answer.strip()) if part.strip()]


def _normalize_sentence(sentence: str) -> str:
    return " ".join(sentence.casefold().split()).strip(" .!?\t")


class GroundedGenerator:
    def __init__(self, config):
        self.cfg = config["generator"]
        self.model = None
        self.tokenizer = None
        self.fallback_reason = None
        backend = self.cfg.get("backend", "extractive")
        if backend not in {"extractive", "auto", "transformers"}:
            raise ValueError(f"Unsupported generator backend: {backend}")
        if backend in {"transformers", "auto"}:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM

                local_only = backend == "auto" or not self.cfg.get("allow_download", False)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.cfg["model_name"],
                    local_files_only=local_only,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.cfg["model_name"],
                    device_map="auto",
                    torch_dtype="auto",
                    local_files_only=local_only,
                )
            except ImportError as exc:
                if backend == "transformers":
                    raise ImportError(
                        "The Transformers generator backend is unavailable. "
                        "Install craft_rag/requirements-llm.txt or select generator.backend: extractive."
                    ) from exc
                logging.warning(
                    "Transformers is not installed; using extractive evidence summaries instead."
                )
                self.fallback_reason = "Transformers dependencies are unavailable; using extractive evidence instead."
            except (OSError, RuntimeError) as exc:
                if backend == "transformers":
                    raise RuntimeError(
                        f"Could not load generator model {self.cfg['model_name']!r}. "
                        "Check the local model cache or explicitly set generator.allow_download: true."
                    ) from exc
                logging.warning(
                    "Generator model %s is unavailable locally (%s); using extractive evidence summaries.",
                    self.cfg["model_name"],
                    exc,
                )
                self.fallback_reason = f"Generator model {self.cfg['model_name']!r} is unavailable; using extractive evidence instead."

    def answer(self, question: str, passages: list[dict]) -> str:
        """Backward-compatible string answer generator."""
        structured = self.generate_structured(question, passages)
        return structured["answer"]

    def generate_structured(self, question: str, passages: list[dict]) -> dict:
        """Generate a grounded answer with passage-level structured citations.

        Returns a dictionary:
            {
                "question": str,
                "answer": str,
                "abstained": bool,
                "citations": [
                    {
                        "source_id": str,
                        "chunk_id": str,
                        "evidence_level": str,
                        "quote": str,
                    }
                ],
                "retrieved_chunk_ids": list[str],
            }
        """
        # Collect all retrieved chunk IDs preserving order
        retrieved_chunk_ids: list[str] = []
        for p in passages:
            cid = _resolve_chunk_id(p)
            if cid and cid not in retrieved_chunk_ids:
                retrieved_chunk_ids.append(cid)

        evidence = [passage for passage in passages if _is_evidence(passage)]
        if len(evidence) < self.cfg.get("minimum_results", 1):
            return {
                "question": question,
                "answer": "Insufficient evidence in the retrieved corpus to answer this question.",
                "abstained": True,
                "answer_mode": "abstained",
                "fallback_reason": self.fallback_reason,
                "confidence": 0.0,
                "citations": [],
                "retrieved_chunk_ids": retrieved_chunk_ids,
            }

        ranked_sentences = self._rank_sentences(question, evidence)
        if not ranked_sentences:
            return {
                "question": question,
                "answer": "Insufficient evidence in the retrieved corpus to answer this question.",
                "abstained": True,
                "answer_mode": "abstained",
                "fallback_reason": self.fallback_reason,
                "confidence": 0.0,
                "citations": [],
                "retrieved_chunk_ids": retrieved_chunk_ids,
            }

        if self.model is not None:
            context = "\n\n".join(
                f"[{passage['source_id']} | {passage.get('evidence_level', 'E5')}] "
                f"{passage.get('text_chunk', '')} Caution: {passage.get('caution', '')}"
                for passage in evidence
            )
            system = (
                "Answer only from the supplied source passages. Return only strict JSON with exactly "
                "the keys answer (string), citations (array of objects with sentence and source_id), "
                "and confidence (number from 0 to 1). Give a citation object for every factual sentence. "
                "If sources do not support the answer, use an empty answer and empty citations. Preserve "
                "cautions and never present placeholder data as fact."
            )
            user = f"Source passages:\n{context}\n\nQuestion: {question}"
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = f"{system}\n\n{user}\n\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.get("max_new_tokens", 220),
                do_sample=False,
            )
            model_answer = self.tokenizer.decode(
                output[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()
            try:
                payload = json.loads(model_answer)
                if not isinstance(payload, dict) or set(payload) != {"answer", "citations", "confidence"}:
                    raise ValueError("JSON does not match the required answer/citations/confidence schema")
                answer = payload["answer"]
                raw_citations = payload["citations"]
                confidence = payload["confidence"]
                if (not isinstance(answer, str) or not answer.strip()
                        or not isinstance(raw_citations, list)
                        or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                        or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                    raise ValueError("JSON does not match the required answer/citations/confidence schema")
                sentences = _answer_sentences(answer)
                mapped_citations = []
                rendered_sentences = []
                for sentence in sentences:
                    citation = next((item for item in raw_citations
                                     if isinstance(item, dict)
                                     and _normalize_sentence(str(item.get("sentence", ""))) == _normalize_sentence(sentence)), None)
                    passage = next((item for item in evidence
                                    if citation and item.get("source_id") == citation.get("source_id")), None)
                    if passage is None:
                        raise ValueError("A generated sentence has no citation to a retrieved passage")
                    support, quote = _citation_support(sentence, passage)
                    if support < 0.5 or not quote:
                        raise ValueError("A generated sentence is unsupported by its cited passage")
                    source_id = str(passage.get("source_id") or "UNKNOWN")
                    evidence_level = str(passage.get("evidence_level") or "E5")
                    mapped_citations.append({
                        "source_id": source_id,
                        "chunk_id": _resolve_chunk_id(passage),
                        "evidence_level": evidence_level,
                        "quote": quote,
                    })
                    rendered_sentences.append(f"{sentence} [{source_id} | {evidence_level}]")
                if not sentences or len(mapped_citations) != len(sentences):
                    raise ValueError("Generated answer has no citation for every sentence")
                return {
                    "question": question,
                    "answer": " ".join(rendered_sentences),
                    "abstained": False,
                    "answer_mode": "generated",
                    "fallback_reason": None,
                    "confidence": float(confidence),
                    "citations": mapped_citations,
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                }
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                reason = "malformed JSON" if isinstance(exc, json.JSONDecodeError) else str(exc)
                self.fallback_reason = f"Transformers output rejected ({reason}); an extractive answer was used instead."

        # Extractive path
        answer_text, citations = self._extractive_structured(evidence, question, ranked_sentences)
        if not citations or answer_text.startswith("Insufficient evidence"):
            return {
                "question": question,
                "answer": "Insufficient evidence in the retrieved corpus to answer this question.",
                "abstained": True,
                "answer_mode": "abstained",
                "fallback_reason": self.fallback_reason,
                "confidence": 0.0,
                "citations": [],
                "retrieved_chunk_ids": retrieved_chunk_ids,
            }
        return {
            "question": question,
            "answer": answer_text,
            "abstained": False,
            "answer_mode": "fallback" if self.fallback_reason else "extractive",
            "fallback_reason": self.fallback_reason,
            "confidence": 1.0,
            "citations": citations,
            "retrieved_chunk_ids": retrieved_chunk_ids,
        }

    @staticmethod
    def _rank_sentences(question, passages):
        query_terms = set(tokens(question))
        if not query_terms:
            return []
        content_terms = query_terms - {"bamboo", "wood", "carv"}
        required_matches = min(2, len(content_terms))
        ranked = []
        for passage in passages:
            metadata = passage.get("metadata", {})
            text_fields = [
                (passage.get("text_chunk", ""), False),
                (str(metadata.get("relevance_note", "")), True),
            ]
            for text, annotation in text_fields:
                for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", text):
                    sentence = _clean_display_sentence(sentence.strip(" \t-"))
                    if not sentence or re.match(
                        r"^(title|source id|source type|link|questions answered|why theme|"
                        r"process stage|evidence level|craft applicability)\s*:",
                        sentence,
                        re.IGNORECASE,
                    ):
                        continue
                    sentence_terms = set(tokens(sentence))
                    overlap = query_terms & sentence_terms
                    content_overlap = overlap & content_terms
                    coverage = len(content_overlap) / max(len(content_terms), 1)
                    if len(content_overlap) < required_matches or coverage < 0.5:
                        continue
                    specificity = len(content_overlap) / math.sqrt(max(len(sentence_terms), 1))
                    score = coverage * specificity
                    if annotation:
                        score *= 0.85
                    ranked.append((score, passage, sentence, annotation))
        ranked.sort(key=lambda item: -item[0])
        return ranked

    @staticmethod
    def _extractive_structured(passages, question="", ranked_sentences=None):
        ranked_sentences = ranked_sentences or GroundedGenerator._rank_sentences(question, passages)
        selected = []
        citations = []
        seen_sentences = set()
        used_source_kinds = set()
        for _, passage, sentence, annotation in ranked_sentences:
            normalized = " ".join(sentence.casefold().split())
            source_id = str(passage.get("source_id") or "UNKNOWN")
            source_kind = (source_id, annotation)
            if normalized in seen_sentences or source_kind in used_source_kinds:
                continue
            sentence_clean = sentence[:420].rstrip()
            display_sentence = (
                f"Repository annotation (not source text): {sentence_clean}"
                if annotation
                else sentence_clean
            )
            evidence_level = str(passage.get("evidence_level") or "E5")
            chunk_id = _resolve_chunk_id(passage)
            selected.append(
                f"{display_sentence} [{source_id} | {evidence_level}]"
            )
            citations.append({
                "source_id": source_id,
                "chunk_id": chunk_id,
                "evidence_level": evidence_level,
                "quote": display_sentence,
            })
            seen_sentences.add(normalized)
            used_source_kinds.add(source_kind)
            if len(selected) >= 3:
                break
        if not selected:
            return "Insufficient evidence in the retrieved corpus to answer this question.", []
        return " ".join(selected), citations

    @staticmethod
    def _extractive(passages, question="", ranked_sentences=None):
        answer_text, _ = GroundedGenerator._extractive_structured(passages, question, ranked_sentences)
        return answer_text
