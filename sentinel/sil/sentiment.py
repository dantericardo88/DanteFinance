"""FinBERT sentiment pipeline — financial text → positive/negative/neutral with scores."""
from __future__ import annotations
import asyncio
from typing import Optional
from sentinel.core.types import SentimentResult
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_pipeline = None  # Lazy-loaded on first use


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading FinBERT sentiment model...")
        _pipeline = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,  # Return all class scores
            device=-1,   # CPU; set to 0 for CUDA
        )
        logger.info("FinBERT loaded")
    return _pipeline


async def score_sentiment(text: str) -> SentimentResult:
    """Score a single text string for financial sentiment."""
    loop = asyncio.get_event_loop()
    try:
        pipe = _get_pipeline()
        result = await loop.run_in_executor(None, lambda: pipe(text[:512]))
        # result is [[{label, score}, ...]]
        scores = result[0] if isinstance(result[0], list) else result
        label_scores = {r["label"].lower(): r["score"] for r in scores}
        best = max(scores, key=lambda x: x["score"])
        return SentimentResult(
            text=text[:200],
            label=best["label"].lower(),
            score=best["score"],
            positive=label_scores.get("positive", 0.0),
            negative=label_scores.get("negative", 0.0),
            neutral=label_scores.get("neutral", 0.0),
        )
    except Exception as exc:
        logger.warning("FinBERT inference error", error=str(exc))
        return SentimentResult(
            text=text[:200],
            label="neutral",
            score=1.0,
            positive=0.0,
            negative=0.0,
            neutral=1.0,
        )


async def score_sentiment_batch(texts: list[str], batch_size: int = 32) -> list[SentimentResult]:
    """Score a batch of texts using FinBERT. Processes in chunks for memory efficiency."""
    results = []
    loop = asyncio.get_event_loop()
    try:
        pipe = _get_pipeline()
        for i in range(0, len(texts), batch_size):
            chunk = [t[:512] for t in texts[i:i + batch_size]]
            batch_result = await loop.run_in_executor(None, lambda c=chunk: pipe(c))
            for text, item in zip(chunk, batch_result):
                scores = item if isinstance(item, list) else [item]
                label_scores = {r["label"].lower(): r["score"] for r in scores}
                best = max(scores, key=lambda x: x["score"])
                results.append(SentimentResult(
                    text=text[:200],
                    label=best["label"].lower(),
                    score=best["score"],
                    positive=label_scores.get("positive", 0.0),
                    negative=label_scores.get("negative", 0.0),
                    neutral=label_scores.get("neutral", 0.0),
                ))
    except Exception as exc:
        logger.error("FinBERT batch error", error=str(exc))
        # Fallback: return neutral for all
        results = [
            SentimentResult(text=t[:200], label="neutral", score=1.0,
                            positive=0.0, negative=0.0, neutral=1.0)
            for t in texts[len(results):]
        ]
    return results


def aggregate_sentiment(results: list[SentimentResult]) -> dict:
    """Aggregate a list of SentimentResults into summary statistics."""
    if not results:
        return {}
    pos = sum(1 for r in results if r.label == "positive")
    neg = sum(1 for r in results if r.label == "negative")
    neu = sum(1 for r in results if r.label == "neutral")
    total = len(results)
    avg_pos = sum(r.positive for r in results) / total
    avg_neg = sum(r.negative for r in results) / total
    net = (pos - neg) / total
    return {
        "total": total,
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "positive_pct": round(pos / total * 100, 1),
        "negative_pct": round(neg / total * 100, 1),
        "neutral_pct": round(neu / total * 100, 1),
        "avg_positive_score": round(avg_pos, 3),
        "avg_negative_score": round(avg_neg, 3),
        "net_sentiment": round(net, 3),
    }
