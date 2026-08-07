"""
Synthetic document corpus generator.
Produces a deterministic set of documents across capability lanes and splits.
No internet access required — all text is generated locally from a fixed seed.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import List

from core.config import SEED, LANES, SPLIT_TRAIN, SPLIT_EVAL, SPLIT_VALIDATION


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    lane: str
    split: str          # train / eval / validation
    source: str
    content_hash: str

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]


# ── Templates per lane ────────────────────────────────────────────────────────
_TEMPLATES: dict[str, list[str]] = {
    "web": [
        "India has a rich cultural heritage spanning thousands of years.",
        "The monsoon season brings essential rainfall to the Indian subcontinent.",
        "Bangalore is known as the Silicon Valley of India.",
        "The Ganges river is considered sacred in Hinduism.",
        "Cricket is the most popular sport in India.",
        "India launched the Chandrayaan-3 mission to the moon in 2023.",
        "The Indian economy is one of the fastest growing in the world.",
        "Yoga originated in ancient India more than five thousand years ago.",
    ],
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "import numpy as np\nmatrix = np.zeros((10, 10))\nfor i in range(10):\n    matrix[i, i] = 1.0",
        "class BinaryTree:\n    def __init__(self, val):\n        self.val = val\n        self.left = None\n        self.right = None",
        "SELECT name, age FROM users WHERE age > 18 ORDER BY name ASC;",
        "for epoch in range(100):\n    optimizer.zero_grad()\n    loss = criterion(model(x), y)\n    loss.backward()\n    optimizer.step()",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    return quicksort([x for x in arr if x < pivot]) + [pivot] + quicksort([x for x in arr if x > pivot])",
    ],
    "indic": [
        "भारत एक विविधताओं से भरा देश है जहाँ अनेक भाषाएँ और संस्कृतियाँ मिलती हैं।",
        "हिंदी भाषा भारत की राजभाषा है और करोड़ों लोग इसे बोलते हैं।",
        "భారతదేశం ఒక గొప్ప దేశం, ఇక్కడ అనేక సంస్కృతులు మరియు భాషలు ఉన్నాయి।",
        "ভারত একটি বিশাল দেশ যেখানে বিভিন্ন সংস্কৃতি ও ভাষা রয়েছে।",
        "महाराष्ट्र हे भारतातील एक महत्त्वाचे राज्य आहे।",
        "இந்தியா ஒரு பண்டைய நாகரிகம் கொண்ட நாடு.",
    ],
    "stem": [
        "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
        "Newton's second law states that force equals mass times acceleration.",
        "DNA carries the genetic information that determines the characteristics of living organisms.",
        "The Pythagorean theorem states that in a right triangle a squared plus b squared equals c squared.",
        "Quantum entanglement is a phenomenon where two particles become correlated regardless of distance.",
    ],
    "agentic": [
        "User: Find all Python files in the repository.\nAssistant: <tool_call>list_files(pattern='*.py')</tool_call>\nObservation: ['main.py', 'utils.py', 'config.py']\nAssistant: I found 3 Python files.",
        "User: Run the test suite.\nAssistant: <tool_call>run_command('pytest tests/')</tool_call>\nObservation: 15 passed, 0 failed\nAssistant: All 15 tests passed successfully.",
        "User: Check the current git branch.\nAssistant: <tool_call>run_command('git branch --show-current')</tool_call>\nObservation: main\nAssistant: You are on the main branch.",
    ],
    "reasoning": [
        "Problem: If a train travels 120 km in 2 hours, what is its speed?\nStep 1: Speed = Distance / Time\nStep 2: Speed = 120 / 2 = 60 km/h\nAnswer: The train travels at 60 km/h.",
        "Problem: Prove that the square root of 2 is irrational.\nAssuming sqrt(2) = p/q in lowest terms, then 2 = p^2/q^2, so p^2 = 2q^2. This means p is even, so p = 2k. Then 4k^2 = 2q^2, so q^2 = 2k^2, meaning q is also even. Contradiction with lowest terms.",
        "Question: What is the sum of all integers from 1 to 100?\nUsing Gauss's formula: sum = n*(n+1)/2 = 100*101/2 = 5050.",
    ],
    "longctx": [
        "Chapter 1: The Origins of Machine Learning\n" + "Machine learning has its roots in statistics and computer science. " * 8,
        "Section 2.3: Data Preprocessing\n" + "Before training any model, data must be carefully prepared. " * 8,
        "Abstract: This paper presents a novel approach to neural architecture search. " * 10,
    ],
}

_EVAL_TEXTS = [
    "What is the capital of France? The capital of France is Paris.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "The sum of angles in a triangle is always 180 degrees.",
]

_VALIDATION_TEXTS = [
    "The Indian Space Research Organisation was founded in 1969.",
    "Python is a high-level interpreted programming language.",
    "The mitochondria is the powerhouse of the cell.",
]


def generate_corpus(seed: int = SEED) -> List[Document]:
    """Return a deterministic list of Documents across all lanes and splits."""
    rng = random.Random(seed)
    docs: List[Document] = []
    counter = 0

    def _make(text: str, lane: str, split: str, source: str) -> Document:
        nonlocal counter
        doc_id = f"doc_{counter:04d}_{lane}_{split}"
        counter += 1
        # Vary text slightly per document to avoid exact duplicates
        full_text = text + f" [src:{source}]"
        return Document(
            doc_id=doc_id,
            text=full_text,
            lane=lane,
            split=split,
            source=source,
            content_hash=Document._hash(full_text),
        )

    # Train documents: ~8 per lane
    for lane in LANES:
        templates = _TEMPLATES.get(lane, _TEMPLATES["web"])
        for i in range(8):
            tmpl = templates[i % len(templates)]
            # Add variation
            variation = f" Variant {rng.randint(1000, 9999)}."
            docs.append(_make(tmpl + variation, lane, SPLIT_TRAIN, f"{lane}_corpus_v1"))

    # Eval documents (must NEVER enter training)
    for i, text in enumerate(_EVAL_TEXTS):
        docs.append(_make(text, "eval", SPLIT_EVAL, "benchmark_set_v1"))

    # Validation documents (readable during training, no gradient)
    for i, text in enumerate(_VALIDATION_TEXTS):
        docs.append(_make(text, "validation", SPLIT_VALIDATION, "validation_set_v1"))

    return docs
