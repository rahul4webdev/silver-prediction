'use client';

import { useEffect, useState } from 'react';
import { getPredictions } from '@/lib/api';
import { cn } from '@/lib/utils';
import type { Prediction, Market, Interval } from '@/lib/types';
import PriceChart from '@/components/PriceChart';

const markets: { value: Market; label: string }[] = [
  { value: 'mcx', label: 'MCX (India)' },
  { value: 'comex', label: 'COMEX (US)' },
];

const intervals: { value: Interval | 'all'; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: '30m', label: '30 Min' },
  { value: '1h', label: '1 Hour' },
  { value: '4h', label: '4 Hours' },
  { value: '1d', label: 'Daily' },
];

export default function PredictionsPage() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedMarket, setSelectedMarket] = useState<Market>('mcx');
  const [selectedInterval, setSelectedInterval] = useState<Interval | 'all'>('all');

  useEffect(() => {
    async function fetchPredictions() {
      try {
        setLoading(true);
        const intervalParam = selectedInterval === 'all' ? undefined : selectedInterval;
        const data = await getPredictions('silver', selectedMarket, intervalParam, 100);
        setPredictions(data);
      } catch {
        setPredictions([]);
      } finally {
        setLoading(false);
      }
    }

    fetchPredictions();
  }, [selectedMarket, selectedInterval]);

  const formatPrice = (price: number, market: Market) => {
    if (market === 'mcx') {
      return `₹${price.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
    }
    return `$${price.toFixed(2)}`;
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString('en-IN', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getCI80 = (prediction: Prediction) => {
    if (prediction.confidence_intervals?.ci_80) {
      return {
        lower: prediction.confidence_intervals.ci_80.lower,
        upper: prediction.confidence_intervals.ci_80.upper,
      };
    }
    return null;
  };

  const getVerification = (prediction: Prediction) => {
    return prediction.verification;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-card p-6">
        <h1 className="text-2xl font-bold text-white mb-2">Predictions History</h1>
        <p className="text-zinc-400">View all predictions with their outcomes and confidence intervals.</p>
      </div>

      {/* Chart Section */}
      <PriceChart market={selectedMarket} interval={selectedInterval === 'all' ? '1h' : selectedInterval} />

      {/* Filters */}
      <div className="flex flex-wrap gap-4">
        {/* Market Filter */}
        <div className="glass-card p-1 flex">
          {markets.map((market) => (
            <button
              key={market.value}
              onClick={() => setSelectedMarket(market.value)}
              className={cn(
                'px-4 py-2 rounded-lg text-sm font-medium transition-all',
                selectedMarket === market.value
                  ? 'bg-cyan-500/20 text-cyan-400'
                  : 'text-zinc-400 hover:text-white'
              )}
            >
              {market.label}
            </button>
          ))}
        </div>

        {/* Interval Filter */}
        <div className="glass-card p-1 flex">
          {intervals.map((interval) => (
            <button
              key={interval.value}
              onClick={() => setSelectedInterval(interval.value)}
              className={cn(
                'px-3 py-2 rounded-lg text-sm font-medium transition-all',
                selectedInterval === interval.value
                  ? 'bg-cyan-500/20 text-cyan-400'
                  : 'text-zinc-400 hover:text-white'
              )}
            >
              {interval.label}
            </button>
          ))}
        </div>
      </div>

      {/* Predictions Table */}
      <div className="glass-card overflow-hidden">
        {loading ? (
          <div className="p-8 text-center">
            <div className="inline-block w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-zinc-400 mt-4">Loading predictions...</p>
          </div>
        ) : predictions.length === 0 ? (
          <div className="p-8 text-center">
            <p className="text-zinc-400">No predictions found for this filter.</p>
            <p className="text-xs text-zinc-600 mt-2">
              Try selecting a different interval or market.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="text-left text-xs text-zinc-500 font-medium p-4">Time</th>
                  <th className="text-left text-xs text-zinc-500 font-medium p-4">Interval</th>
                  <th className="text-left text-xs text-zinc-500 font-medium p-4">Direction</th>
                  <th className="text-right text-xs text-zinc-500 font-medium p-4">Current</th>
                  <th className="text-right text-xs text-zinc-500 font-medium p-4">Predicted</th>
                  <th className="text-right text-xs text-zinc-500 font-medium p-4">Actual</th>
                  <th className="text-right text-xs text-zinc-500 font-medium p-4">80% CI</th>
                  <th className="text-center text-xs text-zinc-500 font-medium p-4">Result</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((prediction, index) => {
                  const ci80 = getCI80(prediction);
                  const verification = getVerification(prediction);

                  return (
                    <tr
                      key={prediction.id || index}
                      className="border-b border-white/5 hover:bg-white/5 transition-colors"
                    >
                      <td className="p-4">
                        <div className="text-sm text-white">
                          {formatTime(prediction.prediction_time)}
                        </div>
                        <div className="text-xs text-zinc-500">
                          Target: {formatTime(prediction.target_time)}
                        </div>
                      </td>
                      <td className="p-4">
                        <span className="text-sm text-zinc-300 uppercase">
                          {prediction.interval}
                        </span>
                      </td>
                      <td className="p-4">
                        <span className={cn(
                          'inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium',
                          prediction.predicted_direction === 'bullish'
                            ? 'bg-green-500/20 text-green-400'
                            : prediction.predicted_direction === 'bearish'
                            ? 'bg-red-500/20 text-red-400'
                            : 'bg-zinc-500/20 text-zinc-400'
                        )}>
                          {prediction.predicted_direction === 'bullish' ? '↑' : prediction.predicted_direction === 'bearish' ? '↓' : '→'}
                          {prediction.predicted_direction}
                          <span className="opacity-60">
                            ({(prediction.direction_confidence * 100).toFixed(0)}%)
                          </span>
                        </span>
                      </td>
                      <td className="p-4 text-right">
                        <span className="text-sm text-white">
                          {formatPrice(prediction.current_price, selectedMarket)}
                        </span>
                      </td>
                      <td className="p-4 text-right">
                        <span className="text-sm text-cyan-400 font-medium">
                          {formatPrice(prediction.predicted_price, selectedMarket)}
                        </span>
                      </td>
                      <td className="p-4 text-right">
                        {verification ? (
                          <span className={cn(
                            'text-sm font-medium',
                            verification.is_direction_correct ? 'text-green-400' : 'text-red-400'
                          )}>
                            {formatPrice(verification.actual_price, selectedMarket)}
                          </span>
                        ) : (
                          <span className="text-sm text-zinc-500">-</span>
                        )}
                      </td>
                      <td className="p-4 text-right">
                        <span className="text-xs text-zinc-400">
                          {ci80 ? (
                            <>
                              {formatPrice(ci80.lower, selectedMarket)}
                              {' - '}
                              {formatPrice(ci80.upper, selectedMarket)}
                            </>
                          ) : (
                            '-'
                          )}
                        </span>
                      </td>
                      <td className="p-4 text-center">
                        {verification ? (
                          <span className={cn(
                            'inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium',
                            verification.is_direction_correct
                              ? 'bg-green-500/20 text-green-400'
                              : 'bg-red-500/20 text-red-400'
                          )}>
                            {verification.is_direction_correct ? '✓ Correct' : '✗ Wrong'}
                            {verification.within_ci_80 && (
                              <span className="opacity-60">(in CI)</span>
                            )}
                          </span>
                        ) : (
                          <span className="text-xs text-zinc-500">Pending</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Stats Summary */}
      {predictions.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="glass-card p-4 text-center">
            <div className="text-2xl font-bold text-white">{predictions.length}</div>
            <div className="text-xs text-zinc-500">Total Predictions</div>
          </div>
          <div className="glass-card p-4 text-center">
            <div className="text-2xl font-bold text-white">
              {predictions.filter(p => p.verification).length}
            </div>
            <div className="text-xs text-zinc-500">Verified</div>
          </div>
          <div className="glass-card p-4 text-center">
            <div className="text-2xl font-bold text-green-400">
              {predictions.filter(p => p.verification?.is_direction_correct).length}
            </div>
            <div className="text-xs text-zinc-500">Correct</div>
          </div>
          <div className="glass-card p-4 text-center">
            <div className="text-2xl font-bold text-cyan-400">
              {(() => {
                const verified = predictions.filter(p => p.verification);
                if (verified.length === 0) return '0%';
                const correct = verified.filter(p => p.verification?.is_direction_correct).length;
                return `${((correct / verified.length) * 100).toFixed(1)}%`;
              })()}
            </div>
            <div className="text-xs text-zinc-500">Accuracy</div>
          </div>
        </div>
      )}
    </div>
  );
}
