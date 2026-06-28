(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
    return;
  }
  root.RoomsChatLatency = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  function toDuration(end, start) {
    if (!Number.isFinite(end) || !Number.isFinite(start)) {
      return null;
    }
    return Math.max(0, end - start);
  }

  function roundDuration(value) {
    if (!Number.isFinite(value)) {
      return null;
    }
    return Math.round(Math.max(0, value));
  }

  function computeChatLatencyMetrics(timestamps) {
    const requestStart = timestamps && timestamps.requestStart;
    const headersReceivedAt = timestamps && timestamps.headersReceivedAt;
    const firstRenderedAt = timestamps && timestamps.firstRenderedAt;
    const finishedAt = timestamps && timestamps.finishedAt;

    const frontendTtfbMs = roundDuration(toDuration(headersReceivedAt, requestStart));
    const firstTokenMs = roundDuration(toDuration(firstRenderedAt, requestStart));
    const finalResponseMs = roundDuration(toDuration(finishedAt, requestStart));

    return {
      frontendTtfbMs,
      firstTokenMs,
      finalResponseMs,
    };
  }

  function formatLatency(value) {
    if (value === null || value === undefined || !Number.isFinite(value)) {
      return null;
    }
    return value >= 1000 ? (value / 1000).toFixed(1) + ' s' : Math.round(value) + ' ms';
  }

  function buildLatencyText(metrics) {
    const parts = [];
    if (metrics.firstTokenMs !== null && metrics.firstTokenMs !== undefined) {
      parts.push('Token đầu: ' + formatLatency(metrics.firstTokenMs));
    }
    if (metrics.frontendTtfbMs !== null && metrics.frontendTtfbMs !== undefined) {
      parts.push('TTFB: ' + formatLatency(metrics.frontendTtfbMs));
    }
    if (metrics.finalResponseMs !== null && metrics.finalResponseMs !== undefined) {
      parts.push('Hoàn tất: ' + formatLatency(metrics.finalResponseMs));
    }
    if (metrics.backendProcessingMs !== null && metrics.backendProcessingMs !== undefined) {
      parts.push('Backend: ' + formatLatency(metrics.backendProcessingMs));
    }
    return parts.join(' • ');
  }

  return {
    buildLatencyText,
    computeChatLatencyMetrics,
    formatLatency,
  };
});
