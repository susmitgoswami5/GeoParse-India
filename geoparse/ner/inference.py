"""
NER inference engine for Indian address parsing.

Loads a trained token classification model and performs entity extraction
from unstructured Indian address text. Aggregates sub-tokens into
entity spans with confidence scores.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from geoparse.ner.label_schema import ID2LABEL, LABEL2ID, ENTITY_TYPES, get_entity_type, is_begin_label


class AddressNERParser:
    """
    Named Entity Recognition parser for Indian addresses.

    Loads a fine-tuned transformer model and extracts hierarchical address
    entities (house number, building, street, landmark, locality, city,
    state, pincode) with confidence scores.
    """

    def __init__(
        self,
        model_path: str = "models/address_ner/best",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to the trained model directory.
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def parse(self, address_text: str) -> Dict[str, Any]:
        """
        Parse an address string and extract named entities.

        Args:
            address_text: Raw unstructured Indian address text.

        Returns:
            Dict with:
                - 'entities': Dict mapping entity type to extracted text
                - 'spans': List of entity spans with confidence scores
                - 'tokens': Tokenized input with per-token predictions
        """
        # Tokenize
        inputs = self.tokenizer(
            address_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            return_offsets_mapping=True,
        )

        offset_mapping = inputs.pop("offset_mapping")[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

        # Extract entities from predictions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        spans = self._extract_spans(
            tokens, predictions, probs, offset_mapping, address_text
        )

        # Build entity dict (best match per type)
        entities = {}
        for span in spans:
            etype = span["entity_type"]
            if etype not in entities or span["confidence"] > entities[etype]["confidence"]:
                entities[etype] = {
                    "text": span["text"],
                    "confidence": span["confidence"],
                }

        # Build per-token prediction list for visualization
        token_predictions = []
        for i, (tok, pred_id) in enumerate(zip(tokens, predictions)):
            if tok in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            label = ID2LABEL.get(pred_id.item(), "O")
            conf = probs[i][pred_id].item()
            token_predictions.append({
                "token": tok,
                "label": label,
                "confidence": round(conf, 4),
            })

        return {
            "entities": entities,
            "spans": spans,
            "token_predictions": token_predictions,
            "input_text": address_text,
        }

    def _extract_spans(
        self,
        tokens: List[str],
        predictions: torch.Tensor,
        probs: torch.Tensor,
        offset_mapping: torch.Tensor,
        original_text: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract entity spans from token-level predictions.

        Aggregates sub-word tokens into complete entity spans and computes
        span-level confidence as the average of token confidences.
        """
        spans: List[Dict[str, Any]] = []
        current_entity: Optional[Dict] = None

        for i, (token, pred_id) in enumerate(zip(tokens, predictions)):
            # Skip special tokens
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                if current_entity:
                    spans.append(self._finalize_span(current_entity, original_text))
                    current_entity = None
                continue

            label = ID2LABEL.get(pred_id.item(), "O")
            confidence = probs[i][pred_id].item()
            start, end = offset_mapping[i].tolist()

            if label == "O":
                if current_entity:
                    spans.append(self._finalize_span(current_entity, original_text))
                    current_entity = None
            elif is_begin_label(label):
                if current_entity:
                    spans.append(self._finalize_span(current_entity, original_text))
                current_entity = {
                    "entity_type": get_entity_type(label),
                    "start": int(start),
                    "end": int(end),
                    "confidences": [confidence],
                    "tokens": [token],
                }
            else:
                # Inside tag
                if current_entity and get_entity_type(label) == current_entity["entity_type"]:
                    current_entity["end"] = int(end)
                    current_entity["confidences"].append(confidence)
                    current_entity["tokens"].append(token)
                else:
                    # Mismatched I- tag, start new entity
                    if current_entity:
                        spans.append(self._finalize_span(current_entity, original_text))
                    current_entity = {
                        "entity_type": get_entity_type(label),
                        "start": int(start),
                        "end": int(end),
                        "confidences": [confidence],
                        "tokens": [token],
                    }

        if current_entity:
            spans.append(self._finalize_span(current_entity, original_text))

        return spans

    def _finalize_span(self, entity: Dict, original_text: str) -> Dict[str, Any]:
        """Finalize an entity span with text extraction and confidence."""
        text = original_text[entity["start"]:entity["end"]].strip()
        # If text extraction from offsets failed, reconstruct from tokens
        if not text:
            text = self.tokenizer.convert_tokens_to_string(entity["tokens"])

        confidence = sum(entity["confidences"]) / len(entity["confidences"])

        return {
            "entity_type": entity["entity_type"],
            "text": text,
            "start": entity["start"],
            "end": entity["end"],
            "confidence": round(confidence, 4),
        }

    def parse_batch(self, addresses: List[str]) -> List[Dict[str, Any]]:
        """Parse a batch of address strings."""
        return [self.parse(addr) for addr in addresses]
