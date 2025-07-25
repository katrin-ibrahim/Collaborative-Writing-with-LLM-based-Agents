# src/evaluation/metrics/entity_metrics.py
import logging
from flair.data import Sentence
from flair.models import SequenceTagger
from typing import Set

logger = logging.getLogger(__name__)


class EntityMetrics:

    def debug_entity_extraction(self, text: str, label: str = "") -> Set[str]:
        """
        Debug version of extract_entities with detailed logging.
        """
        print(f"\n🔍 DEBUG: Extracting entities from {label}")
        print(f"Text preview: {text[:200]}...")
        if not hasattr(self, "ner_tagger") or self.ner_tagger is None:
            print("❌ FLAIR not available!")
            return set()
        entities = set()
        try:
            from flair.data import Sentence

            sentence = Sentence(text)
            self.ner_tagger.predict(sentence)
            print(f"FLAIR found {len(sentence.get_spans('ner'))} total entities:")
            for entity in sentence.get_spans("ner"):
                print(f"  - '{entity.text}' ({entity.tag}, {entity.score:.3f})")
                if entity.score > 0.5:
                    entity_text = entity.text.lower().strip()
                    if len(entity_text) > 1:
                        entities.add(entity_text)
                        print(f"    ✅ Added: '{entity_text}'")
                    else:
                        print(f"    ❌ Skipped (too short): '{entity_text}'")
                else:
                    print(f"    ❌ Skipped (low confidence): {entity.score:.3f}")
            print(f"Final entity set ({len(entities)}): {sorted(list(entities))}")
        except Exception as e:
            print(f"❌ FLAIR extraction failed: {e}")
            raise
        return entities

    """
    Entity extraction using simple heuristics.
    Focuses on patterns that work well without heavy NLP dependencies.
    """

    def __init__(self):
        self.ner_tagger = SequenceTagger.load("ner")
        logger.info("Loaded FLAIR NER model for entity extraction")

    def extract_entities(self, text: str) -> Set[str]:
        """
        Extract entities using FLAIR NER (STORM-compliant) or use pre-computed entities.
        """
        # Check for pre-computed entities (for threading support)
        if hasattr(self, "_precomputed_entities"):
            # This is a hack for threading - return appropriate pre-computed entities
            # based on which text matches generated vs reference content
            if hasattr(self, "_current_text_type"):
                return self._precomputed_entities.get(self._current_text_type, set())

        logger.debug(f"Extracting entities from text: {text[:200]}...")
        entities = set()

        entities.update(self._extract_entities_flair(text))

        logger.debug(f"Total entities extracted: {len(entities)} - {entities}")
        return entities

    def _extract_entities_flair(self, text: str) -> Set[str]:
        """Extract entities using FLAIR NER (STORM implementation)."""
        entities = set()

        try:
            # Create FLAIR sentence
            sentence = Sentence(text)

            # Predict NER tags
            self.ner_tagger.predict(sentence)

            # Extract entities with confidence > 0.5 (STORM threshold)
            for entity in sentence.get_spans("ner"):
                if entity.score > 0.5:  # STORM confidence threshold
                    entity_text = entity.text.lower().strip()
                    if len(entity_text) > 1:
                        entities.add(entity_text)
                        logger.debug(
                            f"FLAIR entity: {entity_text} ({entity.tag}, {entity.score:.3f})"
                        )

        except Exception as e:
            logger.warning(f"FLAIR NER failed: {e}. returning empty set.")

        return entities

    def calculate_overall_entity_recall(self, generated: str, reference: str) -> float:
        """
        Calculate Article Entity Recall (AER) using STORM-compliant entity extraction.

        AER = |NE(G_article) ∩ NE(P_article)| / |NE(G_article)|
        """
        logger.debug("=== Article Entity Recall Calculation ===")

        try:
            gen_entities = self.extract_entities(generated)
            ref_entities = self.extract_entities(reference)

            logger.debug(f"Generated entities ({len(gen_entities)}): {gen_entities}")
            logger.debug(f"Reference entities ({len(ref_entities)}): {ref_entities}")

            if not ref_entities:
                logger.debug("No reference entities found, returning 1.0")
                return 1.0

            # Calculate overlap using STORM formula
            overlap = len(ref_entities.intersection(gen_entities))
            common_entities = ref_entities.intersection(gen_entities)
            logger.debug(f"Common entities ({overlap}): {common_entities}")

            recall = overlap / len(ref_entities)
            logger.debug(
                f"Article Entity Recall: {overlap}/{len(ref_entities)} = {recall}"
            )
            return recall

        except Exception as e:
            logger.error(f"Entity recall calculation failed: {e}")
            # Since you removed regex fallback, return 0.0 on FLAIR failure
            return 0.0
