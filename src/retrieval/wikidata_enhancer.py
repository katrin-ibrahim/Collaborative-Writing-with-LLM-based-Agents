"""
Wikidata Enhancement Wrapper for Retrieval Managers
Adds entity discovery to any base retrieval method.
"""

import json
import logging
import requests
from typing import Dict, List

logger = logging.getLogger(__name__)


class WikidataEnhancer:
    """
    Wrapper that adds Wikidata entity discovery to any base retrieval method.
    Can be combined with TxtAI, BM25, or any other RM.
    """

    def __init__(self, base_rm, cache_file: str = "wikidata_cache.json"):
        self.base_rm = base_rm
        self.cache_file = cache_file
        self.entity_cache = self._load_cache()

    def search(
        self, query: str, topic: str = None, max_results: int = 10
    ) -> List[Dict]:
        """Enhanced search with Wikidata entities."""
        # 1. Get entities from Wikidata
        entities = []
        if topic:
            entities = self._get_wikidata_entities(topic)
            logger.debug(f"Wikidata entities for '{topic}': {entities}")

        # 2. Create enhanced query
        enhanced_query = query
        if entities:
            # Add top 5 entities to avoid query bloat
            top_entities = entities[:5]
            enhanced_query = f"{query} {' '.join(top_entities)}"

        logger.debug(f"Enhanced query: {enhanced_query}")

        # 3. Use base RM with enhanced query
        return self.base_rm.search(enhanced_query, max_results=max_results)

    def _get_wikidata_entities(self, topic: str) -> List[str]:
        """Get related entities from Wikidata with caching."""
        if topic in self.entity_cache:
            return self.entity_cache[topic]

        try:
            entities = self._query_wikidata(topic)
            self.entity_cache[topic] = entities
            self._save_cache()
            return entities
        except Exception as e:
            logger.warning(f"Wikidata query failed for '{topic}': {e}")
            return []

    def _query_wikidata(self, topic: str) -> List[str]:
        """Query Wikidata SPARQL endpoint."""
        endpoint = "https://query.wikidata.org/sparql"

        sparql_query = f"""
        SELECT DISTINCT ?entityLabel WHERE {{
          ?item rdfs:label "{topic}"@en .
          {{
            # Get items of same type
            ?item wdt:P31 ?type .
            ?related wdt:P31 ?type .
            ?related rdfs:label ?entityLabel .
            FILTER(LANG(?entityLabel) = "en")
          }} UNION {{
            # Get directly related entities
            ?item ?prop ?related .
            ?related rdfs:label ?entityLabel .
            FILTER(LANG(?entityLabel) = "en")
          }} UNION {{
            # Get broader/narrower concepts
            ?item wdt:P279 ?broader .
            ?broader rdfs:label ?entityLabel .
            FILTER(LANG(?entityLabel) = "en")
          }}
        }}
        LIMIT 15
        """

        headers = {
            "User-Agent": "EntityRM/1.0 (Research)",
            "Accept": "application/sparql-results+json",
        }

        response = requests.post(
            endpoint,
            data={"query": sparql_query, "format": "json"},
            headers=headers,
            timeout=10,
        )

        if response.status_code != 200:
            return []

        data = response.json()
        entities = []

        for result in data.get("results", {}).get("bindings", []):
            entity_label = result.get("entityLabel", {}).get("value", "")
            if entity_label and len(entity_label) < 50:
                entities.append(entity_label)

        # Remove duplicates and original topic
        entities = list(set(entities))
        if topic in entities:
            entities.remove(topic)

        return entities[:10]

    def _load_cache(self) -> Dict[str, List[str]]:
        """Load entity cache from JSON file."""
        try:
            with open(self.cache_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

    def _save_cache(self):
        """Save entity cache to JSON file."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.entity_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save Wikidata cache: {e}")
