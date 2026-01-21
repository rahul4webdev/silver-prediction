'use client';

import { useEffect, useState } from 'react';
import { getPredictions, triggerPrediction } from '@/lib/api';
import type { Prediction } from '@/lib/types';
import MarketSelector from '@/components/widgets/MarketSelector';
import IntervalSelector from '@/components/widgets/IntervalSelector';

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

        // Filter based on selection
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

  const getResultBadge = (pred: Prediction) => {
    if (!pred.verified_at) {
      const targetTime = new Date(pred.target_time);
      const now = new Date();
      if (targetTime > now) {
        const diff = targetTime.getTime() - now.getTime();
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(minutes / 60);
        if (hours > 0) {
          return (
            <span className="text-sm text-gray-500">
              {hours}h {minutes % 60}m remaining
            </span>
          );
        }
        return <span className="text-sm text-gray-500">{minutes}m remaining</span>;
      }
      return <span className="text-sm text-yellow-600">Awaiting verification</span>;
    }

    return (
      <div className="flex items-center gap-2">
        {pred.is_direction_correct ? (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
            Correct
          </span>
        ) : (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
            Wrong
          </span>
        )}
        {pred.within_ci_80 && (
          <span className="text-xs text-gray-500" title="Actual price was within 80% CI">
            80% CI
          </span>
        )}
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Prediction History</h1>
          <p className="text-sm text-gray-500 mt-1">
            View all predictions and their verification results
          </p>
        </div>
        <button
          onClick={handleGeneratePrediction}
          disabled={generating}
          className="mt-4 sm:mt-0 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50"
        >
          {generating ? (
            <>
              <div className="spinner-sm mr-2"></div>
              Generating...
            </>
          ) : (
            'Generate New Prediction'
          )}
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-4 mb-6">
        <MarketSelector value={market} onChange={setMarket} />
        <IntervalSelector value={interval} onChange={setInterval} />
        <div className="flex rounded-lg border border-gray-200 overflow-hidden">
          {(['all', 'verified', 'pending'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                filter === f
                  ? 'bg-gray-800 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-50'
              } ${f !== 'all' ? 'border-l border-gray-200' : ''}`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Predictions List */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="spinner"></div>
        </div>
      ) : predictions.length === 0 ? (
        <div className="card text-center py-12">
          <p className="text-gray-500">No predictions found</p>
        </div>
      ) : (
        <div className="space-y-4">
          {predictions.map((pred) => (
            <div key={pred.id} className="card">
              <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
                {/* Left: Time and Market Info */}
                <div className="flex items-start gap-4">
                  <div
                    className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                      pred.predicted_direction === 'bullish'
                        ? 'bg-green-100'
                        : pred.predicted_direction === 'bearish'
                        ? 'bg-red-100'
                        : 'bg-yellow-100'
                    }`}
                  >
                    <span className="text-2xl">
                      {pred.predicted_direction === 'bullish'
                        ? '↗'
                        : pred.predicted_direction === 'bearish'
                        ? '↘'
                        : '→'}
                    </span>
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      {pred.predicted_direction ? (
                        <span className={`badge-${pred.predicted_direction}`}>
                          {pred.predicted_direction.toUpperCase()}
                        </span>
                      ) : (
                        <span className="text-xs text-gray-400">Unknown</span>
                      )}
                      <span className="text-xs text-gray-500 uppercase bg-gray-100 px-2 py-0.5 rounded">
                        {pred.market || 'N/A'}
                      </span>
                      <span className="text-xs text-gray-500">{pred.interval || 'N/A'}</span>
                    </div>
                    <p className="text-sm text-gray-600 mt-1">
                      Target:{' '}
                      {pred.target_time
                        ? new Date(pred.target_time).toLocaleString(undefined, {
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit',
                          })
                        : 'N/A'}
                    </p>
                  </div>
                </div>

                {/* Middle: Prices */}
                <div className="flex items-center gap-6 lg:gap-8">
                  <div className="text-center">
                    <p className="text-xs text-gray-500">Current</p>
                    <p className="font-mono font-medium">
                      {pred.current_price?.toFixed(2) ?? 'N/A'}
                    </p>
                  </div>
                  <div className="text-gray-300">→</div>
                  <div className="text-center">
                    <p className="text-xs text-gray-500">Predicted</p>
                    <p className="font-mono font-medium text-primary-600">
                      {pred.predicted_price?.toFixed(2) ?? 'N/A'}
                    </p>
                  </div>
                  {pred.actual_price && (
                    <>
                      <div className="text-gray-300">→</div>
                      <div className="text-center">
                        <p className="text-xs text-gray-500">Actual</p>
                        <p
                          className={`font-mono font-medium ${
                            pred.is_direction_correct ? 'text-green-600' : 'text-red-600'
                          }`}
                        >
                          {pred.actual_price.toFixed(2)}
                        </p>
                      </div>
                    </>
                  )}
                </div>

                {/* Right: Confidence and Result */}
                <div className="flex items-center gap-6">
                  <div className="text-center">
                    <p className="text-xs text-gray-500">Confidence</p>
                    <p className="font-semibold">
                      {pred.direction_confidence
                        ? (pred.direction_confidence * 100).toFixed(1) + '%'
                        : 'N/A'}
                    </p>
                  </div>
                  <div className="text-center min-w-[120px]">
                    <p className="text-xs text-gray-500 mb-1">Result</p>
                    {getResultBadge(pred)}
                  </div>
                </div>
              </div>

              {/* Expanded Details */}
              <div className="mt-4 pt-4 border-t border-gray-100 grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-gray-500">50% CI</p>
                  <p className="font-mono">
                    {(pred.ci_50_lower ?? 0).toFixed(2)} - {(pred.ci_50_upper ?? 0).toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-gray-500">80% CI</p>
                  <p className="font-mono">
                    {(pred.ci_80_lower ?? 0).toFixed(2)} - {(pred.ci_80_upper ?? 0).toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-gray-500">95% CI</p>
                  <p className="font-mono">
                    {(pred.ci_95_lower ?? 0).toFixed(2)} - {(pred.ci_95_upper ?? 0).toFixed(2)}
                  </p>
                </div>
                {pred.price_error_percent !== undefined && pred.price_error_percent !== null && (
                  <div>
                    <p className="text-gray-500">Error</p>
                    <p
                      className={`font-mono ${
                        Math.abs(pred.price_error_percent) < 1
                          ? 'text-green-600'
                          : 'text-yellow-600'
                      }`}
                    >
                      {pred.price_error_percent >= 0 ? '+' : ''}
                      {pred.price_error_percent.toFixed(3)}%
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
