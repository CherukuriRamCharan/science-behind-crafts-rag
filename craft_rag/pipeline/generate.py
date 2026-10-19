"""Evidence-only answer construction and optional local Transformers backend."""
from __future__ import annotations

import logging
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
        stop = {
            "why", "what", "when", "where", "which", "who", "how", "does", "do",
            "is", "are", "a", "an", "the", "to", "of", "for", "in", "on", "with",
            "and", "or", "might", "may", "can",
        }
        query_tokens = {token for token in tokens(question) if token not in stop}
        content = set()
        for passage in evidence:
            metadata = passage.get("metadata", {})
            content.update(tokens(passage.get("text_chunk", "")))
            content.update(tokens(" ".join([
                str(passage.get("why_theme", "")),
                str(metadata.get("title", "")),
                str(passage.get("questions_answered", [])),
            ])))
        if not query_tokens or not (query_tokens & content) or len(query_tokens & content) / len(query_tokens) < 0.08:
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
                return self._extractive(evidence)
            return answer

        return self._extractive(evidence)

    @staticmethod
    def _extractive(passages):
        selected = []
        for passage in passages[:3]:
            text = passage.get("text_chunk", "").strip()
            if text:
                selected.append(
                    f"{text} [{passage['source_id']} | {passage.get('evidence_level', 'E5')}]"
                )
        if not selected:
            return "Insufficient evidence in the retrieved corpus to answer this question."
        return "Retrieved evidence: " + " ".join(selected)
