const test = require('node:test');
const assert = require('node:assert/strict');

const {
  computeChatLatencyMetrics,
  buildLatencyText,
  formatLatency,
} = require('./chat_latency.js');

test('computeChatLatencyMetrics returns rounded chat timings', () => {
  const metrics = computeChatLatencyMetrics({
    requestStart: 1000,
    headersReceivedAt: 1450.4,
    firstRenderedAt: 1825.6,
    finishedAt: 2401.2,
  });

  assert.deepEqual(metrics, {
    frontendTtfbMs: 450,
    firstTokenMs: 826,
    finalResponseMs: 1401,
  });
});

test('computeChatLatencyMetrics clamps negative and invalid timings', () => {
  const metrics = computeChatLatencyMetrics({
    requestStart: 1000,
    headersReceivedAt: 950,
    firstRenderedAt: Number.NaN,
    finishedAt: 900,
  });

  assert.deepEqual(metrics, {
    frontendTtfbMs: 0,
    firstTokenMs: null,
    finalResponseMs: 0,
  });
});

test('buildLatencyText renders all visible metrics in expected order', () => {
  const text = buildLatencyText({
    frontendTtfbMs: 420,
    firstTokenMs: 730,
    finalResponseMs: 2100,
    backendProcessingMs: 1800,
  });

  assert.equal(
    text,
    'Token đầu: 730 ms • TTFB: 420 ms • Hoàn tất: 2.1 s • Backend: 1.8 s',
  );
});

test('formatLatency returns null for missing values', () => {
  assert.equal(formatLatency(null), null);
  assert.equal(formatLatency(undefined), null);
});
