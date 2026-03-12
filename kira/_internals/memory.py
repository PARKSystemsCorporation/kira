from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional


STOPWORDS = {
    "a", "an", "the", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "will", "would", "shall", "should",
    "may", "might", "must", "can", "could", "and", "but", "if", "or",
    "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "also", "now",
    "lol", "lmao", "haha", "hehe", "hmm", "idk", "tbh", "imo", "btw", "ok", "okay",
    "yeah", "yes", "nope", "hey", "hi", "hello", "yo", "sup", "like", "just", "get",
    "got", "would", "could", "should", "really", "thing", "things", "way", "even",
}

POS_CATEGORIES = {
    "NN": "noun",
    "NNS": "noun",
    "NNP": "noun",
    "NNPS": "noun",
    "VB": "verb",
    "VBD": "verb",
    "VBG": "verb",
    "VBN": "verb",
    "VBP": "verb",
    "VBZ": "verb",
    "JJ": "adj",
    "JJR": "adj",
    "JJS": "adj",
    "RB": "adv",
    "RBR": "adv",
    "RBS": "adv",
    "CD": "number",
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def _get_simple_pos(tag: str) -> str:
    return POS_CATEGORIES.get(tag, "noun")


def _tokenize_message(text: str) -> List[Dict[str, Any]]:
    cleaned = text.lower()
    cleaned = re.sub(r"[^\w\s'-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return []

    words = cleaned.split(" ")
    tokens: List[Dict[str, Any]] = []
    content_index = 0

    for word in words:
        if word in STOPWORDS:
            continue
        if len(word) < 3:
            continue
        if re.fullmatch(r"\d+", word):
            continue

        pos = "NN"
        simple_pos = _get_simple_pos(pos)
        tokens.append(
            {
                "word": word,
                "pos": pos,
                "simplePOS": simple_pos,
                "index": content_index,
            }
        )
        content_index += 1

    return tokens


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def _score_message_distance(current_index: int, last_seen_index: Optional[int]) -> float:
    if last_seen_index is None:
        return 2.5
    delta = current_index - last_seen_index
    if delta == 0:
        return 2.5
    if delta == 1:
        return 2.0
    if delta <= 3:
        return 1.0
    if delta <= 6:
        return 0.5
    return 0.0


def _score_category_relation(pos1: str, pos2: str) -> Dict[str, Any]:
    if pos1 == "noun" and pos2 == "noun":
        return {"score": 2.5, "relation": "noun+noun"}
    if (pos1 == "adj" and pos2 == "noun") or (pos1 == "noun" and pos2 == "adj"):
        return {"score": 2.0, "relation": "adj+noun"}
    if (pos1 == "verb" and pos2 == "noun") or (pos1 == "noun" and pos2 == "verb"):
        return {"score": 1.5, "relation": "verb+noun"}
    if pos1 == pos2:
        return {"score": 1.0, "relation": f"{pos1}+{pos2}"}
    return {"score": 0.5, "relation": f"{pos1}+{pos2}"}


def _score_sentence(same_sentence: bool) -> float:
    return 2.5 if same_sentence else 0.0


def _score_temporal(distance: int, same_sentence: bool) -> float:
    if not same_sentence:
        return 0.0
    if distance == 0:
        return 2.5
    if distance == 1:
        return 2.0
    if distance <= 3:
        return 1.0
    return 0.5


def _calculate_initial_score(pos1: str, pos2: str, word_distance: int) -> Dict[str, Any]:
    category = _score_category_relation(pos1, pos2)
    sentence_score = _score_sentence(True)
    temporal_score = _score_temporal(word_distance, True)
    msg_dist_score = 2.5

    raw_score = msg_dist_score + category["score"] + sentence_score + temporal_score
    initial_score = (raw_score / 10.0) * 0.1

    return {
        "score": min(initial_score, 1.0),
        "categoryRel": category["relation"],
    }


def _generate_pattern_key(word1: str, word2: str) -> str:
    return "_".join(sorted([word1.lower(), word2.lower()]))


def _get_tier_for_score(score: float, thresholds: Dict[str, float]) -> str:
    if score >= thresholds["MEDIUM_MAX"]:
        return "long"
    if score >= thresholds["SHORT_MAX"]:
        return "medium"
    return "short"


def _table_for_tier(tier: str) -> str:
    return f"chat_{tier}"


class MemoryOrchestrator:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._verbose = bool(self._config.get("verbose", False))
        self._db_blocked = False
        self._db_retry_at = 0.0
        self._db_block_duration = 30.0

        self._is_processing = False
        self._pending_queue: List[Dict[str, Any]] = []
        self._max_queue = 100

        self._phrase_tick = 0
        self._decay_tick = 0
        self._phrase_interval = 5
        self._decay_interval = 10

        self._thresholds = {
            "SHORT_MAX": 0.25,
            "MEDIUM_MAX": 0.65,
            "DECAY_MIN": 0.05,
        }

        self._decay_config = {
            "short": {"interval": 75, "rate": 0.10},
            "medium": {"interval": 200, "rate": 0.05},
            "long": {"interval": 1000, "rate": 0.01},
        }

        self._lock = threading.Lock()

        memory_path = os.path.expanduser(self._config.get("memory_path", "~/.kira/memory.db"))
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        self._conn = sqlite3.connect(memory_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        cur = self._conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_message_counter (
                id INTEGER PRIMARY KEY,
                current_index INTEGER NOT NULL,
                last_updated TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_purgatory (
                id TEXT PRIMARY KEY,
                word TEXT,
                pos TEXT,
                simple_pos TEXT,
                sentence TEXT,
                sentence_index INTEGER,
                message_id TEXT,
                message_index INTEGER,
                user_id TEXT
            )
            """
        )

        for tier in ["short", "medium", "long"]:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS { _table_for_tier(tier) } (
                    id TEXT PRIMARY KEY,
                    pattern_key TEXT,
                    word1 TEXT,
                    word2 TEXT,
                    pos1 TEXT,
                    pos2 TEXT,
                    category_rel TEXT,
                    joined_sentence TEXT,
                    correlation_score REAL,
                    reinforcement_count INTEGER,
                    decay_count INTEGER,
                    decay_at_message INTEGER,
                    last_seen_message_index INTEGER,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_decay (
                id TEXT PRIMARY KEY,
                pattern_key TEXT,
                word1 TEXT,
                word2 TEXT,
                pos1 TEXT,
                pos2 TEXT,
                category_rel TEXT,
                joined_sentence TEXT,
                correlation_score REAL,
                reinforcement_count INTEGER,
                decay_count INTEGER,
                decay_at_message INTEGER,
                last_seen_message_index INTEGER,
                created_at TEXT,
                updated_at TEXT,
                decayed_from TEXT,
                decayed_at TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_phrases (
                id TEXT PRIMARY KEY,
                phrase_key TEXT,
                words TEXT,
                pos_tags TEXT,
                source_correlations TEXT,
                correlation_score REAL,
                reinforcement_count INTEGER,
                decay_count INTEGER,
                decay_at_message INTEGER,
                tier TEXT,
                last_seen_message_index INTEGER
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_interactions (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                content_hash TEXT,
                user_message TEXT,
                response TEXT,
                summary TEXT,
                importance REAL,
                tags TEXT
            )
            """
        )

        self._conn.commit()

    def _log(self, message: str) -> None:
        if self._verbose:
            print(f"[KIRA] {message}")

    def _can_write_db(self) -> bool:
        now = time.time()
        if self._db_blocked and now >= self._db_retry_at:
            self._db_blocked = False
        return not self._db_blocked

    def _block_db(self) -> None:
        self._db_blocked = True
        self._db_retry_at = time.time() + self._db_block_duration

    def _get_and_increment_message_index(self) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO chat_message_counter (id, current_index, last_updated) VALUES (1, 0, ?)",
            (_now_iso(),),
        )
        cur.execute(
            "UPDATE chat_message_counter SET current_index = current_index + 1, last_updated = ? WHERE id = 1",
            (_now_iso(),),
        )
        cur.execute("SELECT current_index FROM chat_message_counter WHERE id = 1")
        row = cur.fetchone()
        self._conn.commit()
        return int(row["current_index"]) if row else 1

    def _get_current_message_index(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT current_index FROM chat_message_counter WHERE id = 1")
        row = cur.fetchone()
        return int(row["current_index"]) if row else 0

    def _load_correlation_cache(self, pattern_keys: List[str]) -> Dict[str, Dict[str, Any]]:
        cache: Dict[str, Dict[str, Any]] = {}
        if not pattern_keys:
            return cache

        cur = self._conn.cursor()
        for tier in ["short", "medium", "long"]:
            table = _table_for_tier(tier)
            placeholders = ",".join(["?"] * len(pattern_keys))
            cur.execute(
                f"SELECT * FROM {table} WHERE pattern_key IN ({placeholders})",
                pattern_keys,
            )
            for row in cur.fetchall():
                row_dict = dict(row)
                row_dict["currentTier"] = tier
                cache[row_dict["pattern_key"]] = row_dict

        remaining_keys = [k for k in pattern_keys if k not in cache]
        if remaining_keys:
            placeholders = ",".join(["?"] * len(remaining_keys))
            cur.execute(
                f"SELECT * FROM chat_decay WHERE pattern_key IN ({placeholders})",
                remaining_keys,
            )
            for row in cur.fetchall():
                row_dict = dict(row)
                row_dict["currentTier"] = "decay"
                cache[row_dict["pattern_key"]] = row_dict

        return cache

    def _move_correlation(self, correlation: Dict[str, Any], from_tier: str, to_tier: str) -> bool:
        if not self._can_write_db():
            return False

        from_table = "chat_decay" if from_tier == "decay" else _table_for_tier(from_tier)
        to_table = _table_for_tier(to_tier)

        cur = self._conn.cursor()
        cur.execute(f"DELETE FROM {from_table} WHERE id = ?", (correlation["id"],))

        new_record = {
            "id": correlation["id"],
            "pattern_key": correlation["pattern_key"],
            "word1": correlation["word1"],
            "word2": correlation["word2"],
            "pos1": correlation["pos1"],
            "pos2": correlation["pos2"],
            "category_rel": correlation["category_rel"],
            "joined_sentence": correlation["joined_sentence"],
            "correlation_score": correlation["correlation_score"],
            "reinforcement_count": correlation["reinforcement_count"],
            "decay_count": correlation.get("decay_count", 0),
            "decay_at_message": correlation["decay_at_message"],
            "last_seen_message_index": correlation["last_seen_message_index"],
            "created_at": correlation.get("created_at") or _now_iso(),
            "updated_at": _now_iso(),
        }

        columns = ",".join(new_record.keys())
        placeholders = ",".join(["?"] * len(new_record))
        cur.execute(
            f"INSERT INTO {to_table} ({columns}) VALUES ({placeholders})",
            list(new_record.values()),
        )

        self._conn.commit()
        return True

    def _words_to_purgatory(
        self, message_text: str, message_id: str, user_id: Optional[str], message_index: int
    ) -> List[Dict[str, Any]]:
        sentences = _split_sentences(message_text)
        all_words: List[Dict[str, Any]] = []

        for sentence in sentences:
            tokens = _tokenize_message(sentence)
            for token in tokens:
                all_words.append(
                    {
                        "id": str(uuid.uuid4()),
                        "word": token["word"],
                        "pos": token["pos"],
                        "simple_pos": token["simplePOS"],
                        "sentence": sentence,
                        "sentence_index": token["index"],
                        "message_id": message_id,
                        "message_index": message_index,
                        "user_id": user_id,
                    }
                )

        if not all_words:
            return []

        if not self._can_write_db():
            return all_words

        cur = self._conn.cursor()
        cur.executemany(
            """
            INSERT INTO chat_purgatory (
                id, word, pos, simple_pos, sentence, sentence_index, message_id, message_index, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    w["id"],
                    w["word"],
                    w["pos"],
                    w["simple_pos"],
                    w["sentence"],
                    w["sentence_index"],
                    w["message_id"],
                    w["message_index"],
                    w["user_id"],
                )
                for w in all_words
            ],
        )
        self._conn.commit()
        return all_words

    def _cleanup_purgatory(self, message_index: int) -> None:
        if not self._can_write_db():
            return
        cur = self._conn.cursor()
        cur.execute("DELETE FROM chat_purgatory WHERE message_index = ?", (message_index,))
        self._conn.commit()

    def _run_correlator(self, message_index: int, words_from_purgatory: Optional[List[Dict[str, Any]]]) -> Dict[str, int]:
        purgatory_words = words_from_purgatory or []
        if not purgatory_words:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT * FROM chat_purgatory WHERE message_index = ? ORDER BY sentence_index ASC",
                (message_index,),
            )
            purgatory_words = [dict(r) for r in cur.fetchall()]

        if len(purgatory_words) < 2:
            return {"newCorrelations": 0, "reinforced": 0, "promoted": 0}

        sentence_groups: Dict[str, List[Dict[str, Any]]] = {}
        for word in purgatory_words:
            key = word["sentence"]
            sentence_groups.setdefault(key, []).append(word)

        all_pattern_keys: List[str] = []
        for words in sentence_groups.values():
            if len(words) < 2:
                continue
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    pattern_key = _generate_pattern_key(words[i]["word"], words[j]["word"])
                    if pattern_key not in all_pattern_keys:
                        all_pattern_keys.append(pattern_key)

        correlation_cache = self._load_correlation_cache(all_pattern_keys)

        new_correlations = 0
        reinforced = 0
        promoted = 0

        pending_inserts: List[Dict[str, Any]] = []
        pending_updates: List[Dict[str, Any]] = []
        pending_moves: List[Dict[str, Any]] = []

        for sentence, words in sentence_groups.items():
            if len(words) < 2:
                continue

            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    word_a = words[i]
                    word_b = words[j]
                    distance = j - i - 1

                    pattern_key = _generate_pattern_key(word_a["word"], word_b["word"])
                    existing = correlation_cache.get(pattern_key)

                    if existing:
                        score_result = _calculate_initial_score(
                            word_a.get("simple_pos", "noun"),
                            word_b.get("simple_pos", "noun"),
                            distance,
                        )

                        new_score = min(1.0, float(existing["correlation_score"]) + score_result["score"])
                        new_tier = _get_tier_for_score(new_score, self._thresholds)
                        current_tier = existing["currentTier"]

                        updated_corr = {
                            **existing,
                            "correlation_score": new_score,
                            "reinforcement_count": int(existing.get("reinforcement_count", 0)) + 1,
                            "last_seen_message_index": message_index,
                            "decay_at_message": message_index + self._decay_config[new_tier]["interval"],
                        }

                        if current_tier == "decay":
                            pending_moves.append(
                                {"correlation": updated_corr, "fromTier": "decay", "toTier": new_tier}
                            )
                            promoted += 1
                        elif new_tier != current_tier:
                            pending_moves.append(
                                {"correlation": updated_corr, "fromTier": current_tier, "toTier": new_tier}
                            )
                            promoted += 1
                        else:
                            pending_updates.append({"tier": current_tier, "correlation": updated_corr})

                        reinforced += 1

                    else:
                        score_result = _calculate_initial_score(
                            word_a.get("simple_pos", "noun"),
                            word_b.get("simple_pos", "noun"),
                            distance,
                        )
                        tier = _get_tier_for_score(score_result["score"], self._thresholds)

                        new_corr = {
                            "id": str(uuid.uuid4()),
                            "pattern_key": pattern_key,
                            "word1": word_a["word"],
                            "word2": word_b["word"],
                            "pos1": word_a.get("pos", "NN"),
                            "pos2": word_b.get("pos", "NN"),
                            "category_rel": score_result["categoryRel"],
                            "joined_sentence": sentence,
                            "correlation_score": score_result["score"],
                            "reinforcement_count": 1,
                            "decay_count": 0,
                            "decay_at_message": message_index + self._decay_config[tier]["interval"],
                            "last_seen_message_index": message_index,
                            "created_at": _now_iso(),
                            "updated_at": _now_iso(),
                        }

                        correlation_cache[pattern_key] = {**new_corr, "currentTier": tier}
                        pending_inserts.append({"tier": tier, "correlation": new_corr})
                        new_correlations += 1

        if self._can_write_db():
            inserts_by_tier: Dict[str, List[Dict[str, Any]]] = {"short": [], "medium": [], "long": []}
            for item in pending_inserts:
                inserts_by_tier[item["tier"]].append(item["correlation"])

            cur = self._conn.cursor()
            for tier, correlations in inserts_by_tier.items():
                if not correlations:
                    continue
                cols = [
                    "id",
                    "pattern_key",
                    "word1",
                    "word2",
                    "pos1",
                    "pos2",
                    "category_rel",
                    "joined_sentence",
                    "correlation_score",
                    "reinforcement_count",
                    "decay_count",
                    "decay_at_message",
                    "last_seen_message_index",
                    "created_at",
                    "updated_at",
                ]
                placeholders = ",".join(["?"] * len(cols))
                cur.executemany(
                    f"INSERT OR IGNORE INTO {_table_for_tier(tier)} ({','.join(cols)}) VALUES ({placeholders})",
                    [tuple(c[col] for col in cols) for c in correlations],
                )

            updates_by_tier: Dict[str, List[Dict[str, Any]]] = {"short": [], "medium": [], "long": []}
            for item in pending_updates:
                updates_by_tier[item["tier"]].append(item["correlation"])

            for tier, correlations in updates_by_tier.items():
                if not correlations:
                    continue
                for corr in correlations:
                    cur.execute(
                        f"""
                        UPDATE {_table_for_tier(tier)}
                        SET correlation_score = ?, reinforcement_count = ?, last_seen_message_index = ?,
                            decay_at_message = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            corr["correlation_score"],
                            corr["reinforcement_count"],
                            corr["last_seen_message_index"],
                            corr["decay_at_message"],
                            _now_iso(),
                            corr["id"],
                        ),
                    )

            for move in pending_moves:
                self._move_correlation(move["correlation"], move["fromTier"], move["toTier"])

            self._conn.commit()

        self._cleanup_purgatory(message_index)
        return {"newCorrelations": new_correlations, "reinforced": reinforced, "promoted": promoted}

    def _build_phrases(self, message_index: int) -> int:
        if not self._can_write_db():
            return 0

        cur = self._conn.cursor()
        all_correlations: List[Dict[str, Any]] = []

        for tier in ["short", "medium", "long"]:
            cur.execute(
                f"SELECT * FROM {_table_for_tier(tier)} ORDER BY correlation_score DESC LIMIT 100"
            )
            rows = cur.fetchall()
            for row in rows:
                row_dict = dict(row)
                row_dict["tier"] = tier
                all_correlations.append(row_dict)

        if len(all_correlations) < 2:
            return 0

        new_phrases = 0
        processed_pairs = set()
        pending_phrase_inserts: List[Dict[str, Any]] = []
        pending_phrase_updates: List[Dict[str, Any]] = []

        phrase_keys: List[str] = []
        for i in range(len(all_correlations)):
            for j in range(i + 1, len(all_correlations)):
                corr_a = all_correlations[i]
                corr_b = all_correlations[j]
                if corr_a["pattern_key"] == corr_b["pattern_key"]:
                    continue

                words_a = [corr_a["word1"], corr_a["word2"]]
                words_b = [corr_b["word1"], corr_b["word2"]]
                shared = [w for w in words_a if w in words_b]
                if not shared:
                    continue

                all_words = list({corr_a["word1"], corr_a["word2"], corr_b["word1"], corr_b["word2"]})
                phrase_key = "_".join(sorted(all_words))
                if phrase_key not in phrase_keys:
                    phrase_keys.append(phrase_key)

        phrase_cache: Dict[str, Dict[str, Any]] = {}
        if phrase_keys:
            placeholders = ",".join(["?"] * len(phrase_keys))
            cur.execute(
                f"SELECT * FROM chat_phrases WHERE phrase_key IN ({placeholders})",
                phrase_keys,
            )
            for row in cur.fetchall():
                phrase_cache[row["phrase_key"]] = dict(row)

        for i in range(len(all_correlations)):
            for j in range(i + 1, len(all_correlations)):
                corr_a = all_correlations[i]
                corr_b = all_correlations[j]
                if corr_a["pattern_key"] == corr_b["pattern_key"]:
                    continue

                words_a = [corr_a["word1"], corr_a["word2"]]
                words_b = [corr_b["word1"], corr_b["word2"]]
                shared = [w for w in words_a if w in words_b]
                if not shared:
                    continue

                all_words = list({corr_a["word1"], corr_a["word2"], corr_b["word1"], corr_b["word2"]})
                phrase_key = "_".join(sorted(all_words))
                if phrase_key in processed_pairs:
                    continue
                processed_pairs.add(phrase_key)

                existing = phrase_cache.get(phrase_key)
                if existing:
                    combined_score = min(1.0, corr_a["correlation_score"] + corr_b["correlation_score"])
                    new_score = min(1.0, existing["correlation_score"] + combined_score * 0.5)
                    new_tier = _get_tier_for_score(new_score, self._thresholds)
                    pending_phrase_updates.append(
                        {
                            "id": existing["id"],
                            "correlation_score": new_score,
                            "reinforcement_count": int(existing.get("reinforcement_count", 0)) + 1,
                            "tier": new_tier,
                            "decay_at_message": message_index + self._decay_config[new_tier]["interval"],
                            "last_seen_message_index": message_index,
                        }
                    )
                else:
                    combined_score = min(1.0, (corr_a["correlation_score"] + corr_b["correlation_score"]) * 0.5)
                    tier = _get_tier_for_score(combined_score, self._thresholds)
                    new_phrase = {
                        "id": str(uuid.uuid4()),
                        "phrase_key": phrase_key,
                        "words": json.dumps(all_words),
                        "pos_tags": json.dumps(
                            list(
                                {
                                    corr_a["pos1"],
                                    corr_a["pos2"],
                                    corr_b["pos1"],
                                    corr_b["pos2"],
                                }
                            )
                        ),
                        "source_correlations": json.dumps([corr_a["id"], corr_b["id"]]),
                        "correlation_score": combined_score,
                        "reinforcement_count": 1,
                        "decay_count": 0,
                        "decay_at_message": message_index + self._decay_config[tier]["interval"],
                        "tier": tier,
                        "last_seen_message_index": message_index,
                    }
                    pending_phrase_inserts.append(new_phrase)
                    new_phrases += 1

        if pending_phrase_inserts:
            cols = [
                "id",
                "phrase_key",
                "words",
                "pos_tags",
                "source_correlations",
                "correlation_score",
                "reinforcement_count",
                "decay_count",
                "decay_at_message",
                "tier",
                "last_seen_message_index",
            ]
            placeholders = ",".join(["?"] * len(cols))
            cur.executemany(
                f"INSERT INTO chat_phrases ({','.join(cols)}) VALUES ({placeholders})",
                [tuple(p[c] for c in cols) for p in pending_phrase_inserts],
            )

        for update in pending_phrase_updates:
            cur.execute(
                """
                UPDATE chat_phrases
                SET correlation_score = ?, reinforcement_count = ?, tier = ?, decay_at_message = ?,
                    last_seen_message_index = ?
                WHERE id = ?
                """,
                (
                    update["correlation_score"],
                    update["reinforcement_count"],
                    update["tier"],
                    update["decay_at_message"],
                    update["last_seen_message_index"],
                    update["id"],
                ),
            )

        self._conn.commit()
        return new_phrases
    def _check_decay(self, current_message_index: int) -> Dict[str, int]:
        if not self._can_write_db():
            return {"decayed": 0, "demoted": 0, "toGraveyard": 0}

        cur = self._conn.cursor()
        tiers = ["short", "medium", "long"]
        total_decayed = 0
        total_demoted = 0
        total_to_graveyard = 0

        for tier in tiers:
            table = _table_for_tier(tier)
            config = self._decay_config[tier]

            cur.execute(
                f"SELECT * FROM {table} WHERE decay_at_message <= ?",
                (current_message_index,),
            )
            due_for_decay = [dict(r) for r in cur.fetchall()]
            if not due_for_decay:
                continue

            for corr in due_for_decay:
                new_score = corr["correlation_score"] * (1 - config["rate"])
                new_decay_count = int(corr.get("decay_count", 0)) + 1

                if new_score < self._thresholds["DECAY_MIN"]:
                    cur.execute(f"DELETE FROM {table} WHERE id = ?", (corr["id"],))
                    cur.execute(
                        """
                        INSERT INTO chat_decay (
                            id, pattern_key, word1, word2, pos1, pos2, category_rel,
                            joined_sentence, correlation_score, reinforcement_count, decay_count,
                            decay_at_message, last_seen_message_index, created_at, updated_at,
                            decayed_from, decayed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            corr["id"],
                            corr["pattern_key"],
                            corr["word1"],
                            corr["word2"],
                            corr["pos1"],
                            corr["pos2"],
                            corr["category_rel"],
                            corr["joined_sentence"],
                            new_score,
                            corr["reinforcement_count"],
                            new_decay_count,
                            corr["decay_at_message"],
                            corr["last_seen_message_index"],
                            corr.get("created_at") or _now_iso(),
                            _now_iso(),
                            tier,
                            _now_iso(),
                        ),
                    )
                    total_to_graveyard += 1
                else:
                    new_tier = _get_tier_for_score(new_score, self._thresholds)
                    if new_tier != tier:
                        corr["correlation_score"] = new_score
                        corr["decay_count"] = new_decay_count
                        corr["decay_at_message"] = current_message_index + self._decay_config[new_tier]["interval"]
                        moved = self._move_correlation(corr, tier, new_tier)
                        if moved:
                            total_demoted += 1
                    else:
                        cur.execute(
                            f"""
                            UPDATE {table}
                            SET correlation_score = ?, decay_count = ?, decay_at_message = ?
                            WHERE id = ?
                            """,
                            (
                                new_score,
                                new_decay_count,
                                current_message_index + config["interval"],
                                corr["id"],
                            ),
                        )
                        total_decayed += 1

        self._conn.commit()
        self._log(f"Decay totals - decayed: {total_decayed}, demoted: {total_demoted}, graveyard: {total_to_graveyard}")
        return {"decayed": total_decayed, "demoted": total_demoted, "toGraveyard": total_to_graveyard}

    def process_message(self, message_text: str, message_id: Optional[str] = None, user_id: Optional[str] = None) -> Dict[str, Any]:
        if not message_text:
            return {"processed": False, "reason": "Empty message"}

        with self._lock:
            if self._is_processing:
                if len(self._pending_queue) >= self._max_queue:
                    return {"processed": False, "reason": "Queue overflow"}
                promise = {"message_text": message_text, "message_id": message_id, "user_id": user_id}
                self._pending_queue.append(promise)
                return {"processed": False, "reason": "Queued"}

            self._is_processing = True

        try:
            return self._process_message_internal(message_text, message_id, user_id)
        finally:
            with self._lock:
                self._is_processing = False
                if self._pending_queue:
                    next_item = self._pending_queue.pop(0)
                    self._process_message_internal(
                        next_item["message_text"], next_item["message_id"], next_item["user_id"]
                    )

    def _process_message_internal(
        self, message_text: str, message_id: Optional[str], user_id: Optional[str]
    ) -> Dict[str, Any]:
        message_id = message_id or str(uuid.uuid4())
        message_index = self._get_and_increment_message_index()
        self._log(f"Processing message index {message_index}")

        words = self._words_to_purgatory(message_text, message_id, user_id, message_index)
        if len(words) < 2:
            self._cleanup_purgatory(message_index)
            return {"processed": True, "messageIndex": message_index, "newCorrelations": 0, "reinforced": 0}

        corr_result = self._run_correlator(message_index, words)

        self._phrase_tick += 1
        self._decay_tick += 1

        new_phrases = 0
        decay_result = {"decayed": 0, "demoted": 0, "toGraveyard": 0}

        if self._phrase_tick % self._phrase_interval == 0:
            self._log("Phrase build triggered")
            new_phrases = self._build_phrases(message_index)
            self._log(f"New phrases: {new_phrases}")

        if self._decay_tick % self._decay_interval == 0:
            self._log("Decay check triggered")
            decay_result = self._check_decay(message_index)
            self._log(f"Decay results: {decay_result}")

        return {
            "processed": True,
            "messageIndex": message_index,
            "wordsProcessed": len(words),
            **corr_result,
            "newPhrases": new_phrases,
            **decay_result,
        }

    def get_memory_context(self, limit: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM chat_short ORDER BY correlation_score DESC LIMIT ?", (limit,))
        short_mem = [dict(r) | {"tier": "short"} for r in cur.fetchall()]

        cur.execute("SELECT * FROM chat_medium ORDER BY correlation_score DESC LIMIT ?", (limit,))
        medium_mem = [dict(r) | {"tier": "medium"} for r in cur.fetchall()]

        cur.execute("SELECT * FROM chat_long ORDER BY correlation_score DESC LIMIT ?", (limit,))
        long_mem = [dict(r) | {"tier": "long"} for r in cur.fetchall()]

        cur.execute("SELECT * FROM chat_phrases ORDER BY correlation_score DESC LIMIT 20")
        phrases = [dict(r) for r in cur.fetchall()]

        return {"short": short_mem, "medium": medium_mem, "long": long_mem, "phrases": phrases}

    def search_by_word(self, word: str) -> List[Dict[str, Any]]:
        normalized = word.lower()
        results: List[Dict[str, Any]] = []
        cur = self._conn.cursor()

        for table in ["chat_short", "chat_medium", "chat_long"]:
            cur.execute(
                f"SELECT * FROM {table} WHERE word1 = ? OR word2 = ? ORDER BY correlation_score DESC",
                (normalized, normalized),
            )
            for row in cur.fetchall():
                row_dict = dict(row)
                row_dict["tier"] = table.replace("chat_", "")
                results.append(row_dict)

        return results

    def get_memory_stats(self) -> Dict[str, Any]:
        cur = self._conn.cursor()
        counts = {}
        for table in ["chat_short", "chat_medium", "chat_long", "chat_decay", "chat_phrases"]:
            cur.execute(f"SELECT COUNT(*) AS c FROM {table}")
            counts[table] = int(cur.fetchone()["c"])

        cur.execute("SELECT current_index FROM chat_message_counter WHERE id = 1")
        row = cur.fetchone()
        current_index = int(row["current_index"]) if row else 0

        return {
            "tiers": {
                "short": counts["chat_short"],
                "medium": counts["chat_medium"],
                "long": counts["chat_long"],
            },
            "decay": counts["chat_decay"],
            "phrases": counts["chat_phrases"],
            "messagesProcessed": current_index,
            "dbBlocked": self._db_blocked,
            "dbRetryAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self._db_retry_at)) if self._db_blocked else None,
            "queueLength": len(self._pending_queue),
            "maxQueue": self._max_queue,
            "phraseTick": self._phrase_tick,
            "decayTick": self._decay_tick,
        }

    def store_interaction(self, user_message: str, response: str) -> None:
        content_hash = hashlib.sha256((user_message + response).encode("utf-8")).hexdigest()
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO memory_interactions (
                id, timestamp, content_hash, user_message, response, summary, importance, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                _now_iso(),
                content_hash,
                user_message,
                response,
                None,
                1.0,
                json.dumps([]),
            ),
        )
        self._conn.commit()

    def reinforce_if_needed(self) -> None:
        self._log("Reinforcement check invoked")
        return
