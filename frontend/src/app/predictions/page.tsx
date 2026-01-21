'use client';

import { useEffect, useState } from 'react';
import { getPredictions, triggerPrediction } from '@/lib/api';
import type { Prediction } from '@/lib/types';

export default function PredictionsPage() {
  const [market, setMarket] = useState<'mcx' | 'comex'>('mcx');
  const [interval, setInterval] = useState<'30m' | '1h' | '4h' | 'daily'>('30m');
  const [filter, setFilter] = useState<'all' | 'verified' | 'pending'>('all');
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    async function loadPredictions() {
      setLoading(true);
      try {
        const data = await getPredictions(
          'silver',
          market,
          interval,
          100,
          filter === 'verified'
        );

        let filtered = data;
        if (filter === 'pending') {
          filtered = data.filter((p) => !p.verified_at);
        }

        setPredictions(filtered);
      } catch (error) {
        console.error('Error loading predictions:', error);
      } finally {
        setLoading(false);
      }
    }

    loadPredictions();
  }, [market, interval, filter]);

  const handleGeneratePrediction = async () => {
    setGenerating(true);
    try {
      const newPrediction = await triggerPrediction('silver', market, interval);
      if (newPrediction) {
        setPredictions((prev) => [newPrediction, ...prev]);
      }
    } catch (error) {
      console.error('Error generating prediction:', error);
    } finally {
      setGenerating(false);
    }
  };

  const currency = market === 'mcx' ? '₹' : '$';

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Prediction History</h1>
          <p className="text-sm text-zinc-400">
            View all predictions and their verification results
          </p>
        </div>
        <button
          onClick={handleGeneratePrediction}
          disabled={generating}
          className="btn-primary mt-4 sm:mt-0 flex items-center gap-2"
        >
          {generating ? (
            <>
              <div className="spinner-sm"></div>
              Generating...
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Generate Prediction
            </>
          )}
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-4 mb-6">
        <div className="segmented-control">
          {(['mcx', 'comex'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMarket(m)}
              className={market === m ? 'active' : ''}
            >
              {m.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="segmented-control">
          {(['30m', '1h', '4h', 'daily'] as const).map((i) => (
            <button
              key={i}
              onClick={() => setInterval(i)}
              className={interval === i ? 'active' : ''}
            >
              {i}
            </button>
          ))}
        </div>
        <div className="segmented-control">
          {(['all', 'verified', 'pending'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={filter === f ? 'active' : ''}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Predictions List */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="spinner"></div>
        </div>
      ) : predictions.length === 0 ? (
        <div className="card">
          <div className="empty-state">
            <svg className="empty-state-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
            <p className="empty-state-title">No Predictions Found</p>
            <p className="empty-state-description">
              Generate a new prediction or adjust the filters to see results.
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {predictions.map((pred) => (
            <div key={pred.id} className="card card-hover">
              <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
                {/* Left: Direction and Info */}
                <div className="flex items-center gap-4">
                  <div
                    className={`w-14 h-14 rounded-2xl flex items-center justify-center ${
                      pred.predicted_direction === 'bullish'
                        ? 'bg-gradient-to-br from-green-500/20 to-emerald-500/10 border border-green-500/30'
                        : pred.predicted_direction === 'bearish'
                        ? 'bg-gradient-to-br from-red-500/20 to-rose-500/10 border border-red-500/30'
                        : 'bg-gradient-to-br from-yellow-500/20 to-amber-500/10 border border-yellow-500/30'
                    }`}
                  >
                    <span className="text-3xl">
                      {pred.predicted_direction === 'bullish'
                        ? '↗'
                        : pred.predicted_direction === 'bearish'
                        ? '↘'
                        : '→'}
                    </span>
                  </div>
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      {pred.predicted_direction ? (
                        <span className={`badge-${pred.predicted_direction}`}>
                          {pred.predicted_direction.toUpperCase()}
                        </span>
                      ) : (
                        <span className="text-zinc-500">Unknown</span>
                      )}
                      <span className="text-xs text-zinc-500 uppercase bg-white/5 px-2 py-0.5 rounded border border-white/5">
                        {pred.market || 'N/A'}
                      </span>
                      <span className="text-xs text-zinc-500">{pred.interval || 'N/A'}</span>
                    </div>
                    <p className="text-sm text-zinc-400">
                      Target:{' '}
                      <span className="text-zinc-300">
                        {pred.target_time
                          ? new Date(pred.target_time).toLocaleString(undefined, {
                              month: 'short',
                              day: 'numeric',
                              hour: '2-digit',
                              minute: '2-digit',
                            })
                          : 'N/A'}
                      </span>
                    </p>
                  </div>
                </div>

                {/* Middle: Prices */}
                <div className="flex items-center gap-4 lg:gap-8">
                  <div className="text-center">
                    <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Current</p>
                    <p className="text-lg font-bold text-white font-mono">
                      {currency}{pred.current_price?.toFixed(2) ?? 'N/A'}
                    </p>
                  </div>
                  <div className="text-zinc-600">→</div>
                  <div className="text-center">
                    <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Predicted</p>
                    <p className={`text-lg font-bold font-mono ${
                      pred.predicted_direction === 'bullish' ? 'text-green-400' :
                      pred.predicted_direction === 'bearish' ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                      {currency}{pred.predicted_price?.toFixed(2) ?? 'N/A'}
                    </p>
                  </div>
                  {pred.actual_price && (
                    <>
                      <div className="text-zinc-600">→</div>
                      <div className="text-center">
                        <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Actual</p>
                        <p className={`text-lg font-bold font-mono ${
                          pred.is_direction_correct ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {currency}{pred.actual_price.toFixed(2)}
                        </p>
                      </div>
                    </>
                  )}
                </div>

                {/* Right: Confidence and Result */}
                <div className="flex items-center gap-6">
                  <div className="text-center">
                    <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Confidence</p>
                    <p className="text-lg font-bold text-cyan-400">
                      {pred.direction_confidence
                        ? (pred.direction_confidence * 100).toFixed(1) + '%'
                        : 'N/A'}
                    </p>
                  </div>
                  <div className="text-center min-w-[100px]">
                    <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Result</p>
                    {pred.verified_at ? (
                      pred.is_direction_correct ? (
                        <span className="badge-success">✓ Correct</span>
                      ) : (
                        <span className="badge-error">✗ Wrong</span>
                      )
                    ) : (
                      <span className="badge-warning">Pending</span>
                    )}
                  </div>
                </div>
              </div>

              {/* Expanded Details */}
              <div className="mt-6 pt-6 border-t border-white/5 grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="p-3 rounded-xl bg-white/5">
                  <p className="text-xs text-zinc-500 mb-1">50% CI</p>
                  <p className="text-sm font-mono text-zinc-300">
                    {currency}{(pred.ci_50_lower ?? 0).toFixed(2)} - {currency}{(pred.ci_50_upper ?? 0).toFixed(2)}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-white/5">
                  <p className="text-xs text-zinc-500 mb-1">80% CI</p>
                  <p className="text-sm font-mono text-zinc-300">
                    {currency}{(pred.ci_80_lower ?? 0).toFixed(2)} - {currency}{(pred.ci_80_upper ?? 0).toFixed(2)}
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-white/5">
                  <p className="text-xs text-zinc-500 mb-1">95% CI</p>
                  <p className="text-sm font-mono text-zinc-300">
                    {currency}{(pred.ci_95_lower ?? 0).toFixed(2)} - {currency}{(pred.ci_95_upper ?? 0).toFixed(2)}
                  </p>
                </div>
                {pred.price_error_percent !== undefined && pred.price_error_percent !== null && (
                  <div className="p-3 rounded-xl bg-white/5">
                    <p className="text-xs text-zinc-500 mb-1">Error</p>
                    <p className={`text-sm font-mono ${
                      Math.abs(pred.price_error_percent) < 1 ? 'text-green-400' : 'text-yellow-400'
                    }`}>
                      {pred.price_error_percent >= 0 ? '+' : ''}{pred.price_error_percent.toFixed(3)}%
                    </p>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
