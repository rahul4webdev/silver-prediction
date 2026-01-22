'use client';

import { useEffect, useState } from 'react';
import { getPredictions } from '@/lib/api';
import { cn } from '@/lib/utils';
import type { Prediction, Market, Interval } from '@/lib/types';

const markets: { value: Market; label: string }[] = [
  { value: 'mcx', label: 'MCX (India)' },
  { value: 'comex', label: 'COMEX (US)' },
];

const intervals: { value: Interval; label: string }[] = [
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
        const data = await getPredictions('silver', selectedMarket, 50);
        setPredictions(data);
      } catch {
        setPredictions([]);
      } finally {
        setLoading(false);
      }
    }

    fetchPredictions();
  }, [selectedMarket]);

  const filteredPredictions = selectedInterval === 'all'
    ? predictions
    : predictions.filter(p => p.interval === selectedInterval);

  const formatPrice = (price: number, market: Market) => {
    if (market === 'mcx') {
      return `₹${price.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-card p-6">
        <h1 className="text-2xl font-bold text-white mb-2">Predictions History</h1>
        <p className="text-zinc-400">View all predictions with their outcomes and confidence intervals.</p>
      </div>

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
          <button
            onClick={() => setSelectedInterval('all')}
            className={cn(
              'px-3 py-2 rounded-lg text-sm font-medium transition-all',
              selectedInterval === 'all'
                ? 'bg-cyan-500/20 text-cyan-400'
                : 'text-zinc-400 hover:text-white'
            )}
          >
            All
          </button>
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
        ) : filteredPredictions.length === 0 ? (
          <div className="p-8 text-center">
            <p className="text-zinc-400">No predictions found.</p>
            <p className="text-xs text-zinc-600 mt-2">
              Predictions will appear here once the system generates them.
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
                  <th className="text-right text-xs text-zinc-500 font-medium p-4">80% CI</th>
                  <th className="text-center text-xs text-zinc-500 font-medium p-4">Result</th>
                </tr>
              </thead>
              <tbody>
                {filteredPredictions.map((prediction, index) => (
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
                      <span className="text-xs text-zinc-400">
                        {prediction.ci_80_lower && prediction.ci_80_upper ? (
                          <>
                            {formatPrice(prediction.ci_80_lower, selectedMarket)}
                            {' - '}
                            {formatPrice(prediction.ci_80_upper, selectedMarket)}
                          </>
                        ) : (
                          '-'
                        )}
                      </span>
                    </td>
                    <td className="p-4 text-center">
                      {prediction.actual_price !== null && prediction.actual_price !== undefined ? (
                        <span className={cn(
                          'inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium',
                          prediction.is_direction_correct
                            ? 'bg-green-500/20 text-green-400'
                            : 'bg-red-500/20 text-red-400'
                        )}>
                          {prediction.is_direction_correct ? '✓ Correct' : '✗ Wrong'}
                        </span>
                      ) : (
                        <span className="text-xs text-zinc-500">Pending</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
