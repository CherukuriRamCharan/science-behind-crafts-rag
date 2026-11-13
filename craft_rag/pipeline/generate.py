"""Evidence-only answer construction and optional local Transformers backend."""
from __future__ import annotations

import logging
import math
import re

from .common import is_evidence_eligible, tokens


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
                "Answer only from the supplied source passages. Cite factual statements inline as "
                "[Source ID | Evidence Level]. If sources do not support an answer, say "
                "'Insufficient evidence.' Preserve cautions and never present placeholder data as fact."
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
            valid = {passage["source_id"] for passage in evidence}
            cited = {citation.strip() for citation in re.findall(r"\[([^\]|]+)\s*\|\s*E[1-5]\]", model_answer)}
            if model_answer and cited and cited.issubset(valid) and not model_answer.startswith("Insufficient evidence"):
                # Model succeeded with valid citations
                model_citations = []
                for sid in cited:
                    for passage in evidence:
                        if passage.get("source_id") == sid:
                            model_citations.append({
                                "source_id": sid,
                                "chunk_id": _resolve_chunk_id(passage),
                                "evidence_level": str(passage.get("evidence_level") or "E5"),
                                "quote": str(passage.get("text_chunk", ""))[:200],
                            })
                            break
                return {
                    "question": question,
                    "answer": model_answer,
                    "abstained": False,
                    "answer_mode": "generated",
                    "fallback_reason": None,
                    "citations": model_citations,
                    "retrieved_chunk_ids": retrieved_chunk_ids,
                }

            self.fallback_reason = (
                "The configured Transformers model returned no fully valid citations; "
                "an extractive answer was used instead."
            )

        # Extractive path
        answer_text, citations = self._extractive_structured(evidence, question, ranked_sentences)
        if not citations or answer_text.startswith("Insufficient evidence"):
            return {
                "question": question,
                "answer": "Insufficient evidence in the retrieved corpus to answer this question.",
                "abstained": True,
                "answer_mode": "abstained",
                "fallback_reason": self.fallback_reason,
                "citations": [],
                "retrieved_chunk_ids": retrieved_chunk_ids,
            }
        return {
            "question": question,
            "answer": answer_text,
            "abstained": False,
            "answer_mode": "fallback" if self.fallback_reason else "extractive",
            "fallback_reason": self.fallback_reason,
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
                    sentence = sentence.strip(" \t-")
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
