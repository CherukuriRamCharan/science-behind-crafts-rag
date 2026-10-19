"""Evidence-only answer construction and optional local Transformers backend."""
from __future__ import annotations

import logging
import math
import re

from .common import is_evidence_eligible, tokens


def _is_evidence(passage):
    return is_evidence_eligible(passage)


class GroundedGenerator:
    def __init__(self, config):
        self.cfg = config["generator"]
        self.model = None
        self.tokenizer = None
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

    def answer(self, question, passages):
        evidence = [passage for passage in passages if _is_evidence(passage)]
        if len(evidence) < self.cfg.get("minimum_results", 1):
            return "Insufficient evidence in the retrieved corpus to answer this question."
        ranked_sentences = self._rank_sentences(question, evidence)
        if not ranked_sentences:
            return "Insufficient evidence in the retrieved corpus to answer this question."
        context = "\n\n".join(
            f"[{passage['source_id']} | {passage.get('evidence_level', 'E5')}] "
            f"{passage.get('text_chunk', '')} Caution: {passage.get('caution', '')}"
            for passage in evidence
        )
        if self.model is not None:
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
            answer = self.tokenizer.decode(
                output[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()
            valid = {passage["source_id"] for passage in evidence}
            cited = set(re.findall(r"\[([^\]|]+)\s*\|\s*E[1-5]\]", answer))
            if not answer or not cited or not cited.issubset(valid):
                return self._extractive(evidence, question)
            return answer

        return self._extractive(evidence, question, ranked_sentences)

    @staticmethod
    def _rank_sentences(question, passages):
        query_terms = set(tokens(question))
        if not query_terms:
            return []
        content_terms = query_terms - {"bamboo"}
        required_matches = min(2, len(content_terms))
        ranked = []
        for passage in passages:
            metadata = passage.get("metadata", {})
            record_type = metadata.get("record_type", passage.get("record_type", ""))
            if record_type == "source_document" and metadata.get("relevance_note"):
                text = str(metadata["relevance_note"])
            else:
                text = passage.get("text_chunk", "")
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
                if len(content_overlap) < required_matches or coverage < 0.6:
                    continue
                score = len(content_overlap) / math.sqrt(max(len(sentence_terms), 1))
                ranked.append((score, passage, sentence))
        ranked.sort(key=lambda item: -item[0])
        return ranked

    @staticmethod
    def _extractive(passages, question="", ranked_sentences=None):
        ranked_sentences = ranked_sentences or GroundedGenerator._rank_sentences(question, passages)
        selected = []
        seen_sentences = set()
        used_sources = set()
        for _, passage, sentence in ranked_sentences:
            normalized = " ".join(sentence.casefold().split())
            if normalized in seen_sentences or passage["source_id"] in used_sources:
                continue
            sentence = sentence[:420].rstrip()
            selected.append(
                f"{sentence} [{passage['source_id']} | {passage.get('evidence_level', 'E5')}]"
            )
            seen_sentences.add(normalized)
            used_sources.add(passage["source_id"])
            if len(selected) >= 3:
                break
        if not selected:
            return "Insufficient evidence in the retrieved corpus to answer this question."
        return " ".join(selected)
