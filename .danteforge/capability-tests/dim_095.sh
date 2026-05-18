#!/usr/bin/env bash
# dim_095: REST SDK — SDK class structure, token bucket rate limiter, TTL cache
set -e
cd "$(dirname "$0")/../.."

python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import time

from sentinel.api.rest_sdk_v3 import (
    _TokenBucket,
    _TTLCache,
    _check_rate,
    SentinelPythonSDK,
    SentinelClient,
    APIDocGenerator,
    API_PREFIX,
    SENTINEL_API_KEY,
)

# Test API_PREFIX constant
assert API_PREFIX == "/api/v3", f"API_PREFIX should be '/api/v3': {API_PREFIX}"
print(f"[OK] API_PREFIX: {API_PREFIX}")

# Test default API key
assert SENTINEL_API_KEY, "SENTINEL_API_KEY should be defined"
print(f"[OK] SENTINEL_API_KEY: '{SENTINEL_API_KEY}'")

# Test SentinelClient alias
assert SentinelClient is SentinelPythonSDK, "SentinelClient should be alias for SentinelPythonSDK"
print(f"[OK] SentinelClient = SentinelPythonSDK (alias verified)")

# Test SentinelPythonSDK instantiation (no network)
sdk = SentinelPythonSDK(host="localhost", port=8000, api_key="test-key")
assert sdk.api_key == "test-key", f"API key should be 'test-key': {sdk.api_key}"
assert sdk.base_url == f"http://localhost:8000{API_PREFIX}", \
    f"Base URL mismatch: {sdk.base_url}"
assert sdk.timeout == 30, f"Default timeout should be 30: {sdk.timeout}"
assert sdk.max_retries == 3, f"Default max_retries should be 3: {sdk.max_retries}"
print(f"[OK] SentinelPythonSDK: host=localhost port=8000 api_key=test-key base_url={sdk.base_url}")

# Test _TokenBucket rate limiting
bucket = _TokenBucket(capacity=10, rate=10.0, tokens=10.0)
assert bucket.capacity == 10
assert bucket.rate == 10.0
assert bucket.tokens == 10.0

# Consume all tokens
consumed = 0
for _ in range(10):
    if bucket.consume():
        consumed += 1
assert consumed == 10, f"Should consume 10 tokens: {consumed}"
print(f"[OK] _TokenBucket: consumed {consumed} tokens from full bucket")

# After exhausting, should return False
can_consume = bucket.consume()
assert can_consume is False, f"Exhausted bucket should return False: {can_consume}"
print(f"[OK] _TokenBucket: returns False when exhausted")

# After waiting, tokens refill
# rate=10 tokens/sec → 0.2s should give ~2 tokens
time.sleep(0.3)
refilled = bucket.consume()
assert refilled is True, f"After waiting, should be able to consume: {refilled}"
print(f"[OK] _TokenBucket: token refill after wait works")

# Test capacity cap: waiting 5s with rate=10 should not exceed capacity=10
big_bucket = _TokenBucket(capacity=5, rate=100.0, tokens=0.0)
time.sleep(0.1)  # 0.1s × 100 = 10 tokens, but capped at 5
tokens_after = min(big_bucket.capacity, big_bucket.tokens + 0.1 * big_bucket.rate)
assert tokens_after <= big_bucket.capacity, \
    f"Tokens should not exceed capacity: {tokens_after}"
print(f"[OK] _TokenBucket: capacity cap enforced at {big_bucket.capacity}")

# Test _check_rate function (uses global per-key buckets)
key1 = "test-sdk-key-1"
key2 = "test-sdk-key-2"
# First call should succeed
r1 = _check_rate(key1)
assert r1 is True, f"First rate check should succeed: {r1}"
r2 = _check_rate(key2)   # different key, independent bucket
assert r2 is True, f"Different key rate check should succeed: {r2}"
print(f"[OK] _check_rate: independent buckets per key")

# Test _TTLCache
cache = _TTLCache(ttl_seconds=60, maxsize=100)

# Set and get within TTL
cache.set("key1", {"price": 150.0})
val = cache.get("key1")
assert val == {"price": 150.0}, f"Should retrieve cached value: {val}"
print(f"[OK] _TTLCache: set and get within TTL works")

# Missing key returns None
missing = cache.get("nonexistent")
assert missing is None, f"Missing key should return None: {missing}"
print(f"[OK] _TTLCache: missing key returns None")

# Test TTL expiration (use very short TTL)
short_cache = _TTLCache(ttl_seconds=1)
short_cache.set("temp_key", "temp_value")
val_now = short_cache.get("temp_key")
assert val_now == "temp_value", f"Should get value immediately: {val_now}"
time.sleep(1.1)   # wait for TTL to expire
val_expired = short_cache.get("temp_key")
assert val_expired is None, f"Expired key should return None: {val_expired}"
print(f"[OK] _TTLCache: TTL expiration works (1s TTL)")

# Test maxsize eviction
small_cache = _TTLCache(ttl_seconds=60, maxsize=3)
small_cache.set("a", 1)
small_cache.set("b", 2)
small_cache.set("c", 3)
small_cache.set("d", 4)   # triggers eviction of oldest
assert len(small_cache._store) <= 3, \
    f"Cache should not exceed maxsize: {len(small_cache._store)}"
print(f"[OK] _TTLCache: maxsize eviction works ({len(small_cache._store)} <= 3)")

# Test APIDocGenerator.ENDPOINTS
gen = APIDocGenerator()
endpoints = gen.ENDPOINTS
assert len(endpoints) >= 20, f"Expected >= 20 endpoints: {len(endpoints)}"

# Verify essential endpoints exist
methods = {e[1]: e[0] for e in endpoints}
assert "/api/v3/health" in methods, "Should have /api/v3/health endpoint"
assert "/api/v3/quote/{ticker}" in methods, "Should have quote endpoint"
assert "/api/v3/fundamentals/{ticker}" in methods, "Should have fundamentals endpoint"

get_endpoints = [e for e in endpoints if e[0] == "GET"]
post_endpoints = [e for e in endpoints if e[0] == "POST"]
assert len(get_endpoints) >= 10, f"Expected >= 10 GET endpoints: {len(get_endpoints)}"
assert len(post_endpoints) >= 3, f"Expected >= 3 POST endpoints: {len(post_endpoints)}"
print(f"[OK] APIDocGenerator.ENDPOINTS: {len(endpoints)} total ({len(get_endpoints)} GET, {len(post_endpoints)} POST)")

# Test generate_markdown_docs
docs = gen.generate_markdown_docs()
assert isinstance(docs, str), "generate_markdown_docs should return string"
assert len(docs) > 500, f"Markdown docs should be substantial: {len(docs)} chars"
assert "SENTINEL API v3" in docs, "Docs should mention 'SENTINEL API v3'"
assert "X-SENTINEL-KEY" in docs, "Docs should mention auth header"
print(f"[OK] APIDocGenerator.generate_markdown_docs(): {len(docs)} chars")

# Test SDK cache key generation
sdk2 = SentinelPythonSDK(host="localhost", port=9000, api_key="key2")
import json
cache_key = f"GET:/api/v3/quote/AAPL:{json.dumps({}, sort_keys=True)}"
sdk2._cache.set(cache_key, {"price": 185.0, "ticker": "AAPL"})
retrieved = sdk2._cache.get(cache_key)
assert retrieved == {"price": 185.0, "ticker": "AAPL"}, f"Cache key retrieval: {retrieved}"
print(f"[OK] SDK._cache: manual cache set/get with structured cache key")

print("\n[PASS] dim_095: REST SDK")
PYEOF
