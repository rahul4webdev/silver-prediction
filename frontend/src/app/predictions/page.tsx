'use client';

import { useEffect, useState } from 'react';
import { getPredictions } from '@/lib/api';
import { cn } from '@/lib/utils';
import type { Prediction, Market, Interval, Asset } from '@/lib/types';
import PriceChart from '@/components/PriceChart';

const assets: { value: Asset; label: string; icon: string }[] = [
  { value: 'silver', label: 'Silver', icon: '🥈' },
  { value: 'gold', label: 'Gold', icon: '🥇' },
];

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
  const [selectedAsset, setSelectedAsset] = useState<Asset>('silver');
  const [selectedMarket, setSelectedMarket] = useState<Market>('mcx');
  const [selectedInterval, setSelectedInterval] = useState<Interval | 'all'>('all');

  // Stats
  const [stats, setStats] = useState({
    total: 0,
    successful: 0,
    failed: 0,
    pending: 0,
    accuracy: 0,
  });

  useEffect(() => {
    async function fetchPredictions() {
      try {
        setLoading(true);
        const intervalParam = selectedInterval === 'all' ? undefined : selectedInterval;
        const data = await getPredictions(selectedAsset, selectedMarket, intervalParam, 100);

        // Show all recent predictions (no date filter)
        // API already limits to last 100 predictions sorted by recency
        setPredictions(data);

        // Calculate stats
        const verified = data.filter(p => p.verification);
        const successful = verified.filter(p => p.verification?.is_direction_correct);
        const failed = verified.filter(p => !p.verification?.is_direction_correct);
        const pending = data.filter(p => !p.verification);

        setStats({
          total: data.length,
          successful: successful.length,
          failed: failed.length,
          pending: pending.length,
          accuracy: verified.length > 0 ? (successful.length / verified.length) * 100 : 0,
        });
      } catch {
        setPredictions([]);
      } finally {
        setLoading(false);
      }
    }

    fetchPredictions();
  }, [selectedAsset, selectedMarket, selectedInterval]);

  const formatPrice = (price: number, market: string) => {
    if (market === 'mcx') {
      return `₹${price.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
    }
    return `$${price.toFixed(2)}`;
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString('en-IN', {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getIntervalLabel = (interval: string) => {
    const labels: Record<string, string> = {
      '30m': '30 Min',
      '1h': '1 Hour',
      '4h': '4 Hours',
      '1d': 'Daily',
    };
    return labels[interval] || interval;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="glass-card p-6">
        <h1 className="text-2xl font-bold text-white mb-2">Recent Predictions</h1>
        <p className="text-zinc-400">View predictions with their outcomes and confidence intervals.</p>
      </div>

      {/* Chart Section */}
      <PriceChart asset={selectedAsset} market={selectedMarket} interval={selectedInterval === 'all' ? '1h' : selectedInterval} />

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="glass-card p-4 text-center">
          <div className="text-3xl font-bold text-white">{stats.total}</div>
          <div className="text-xs text-zinc-500 mt-1">Total Predictions</div>
        </div>
        <div className="glass-card p-4 text-center">
          <div className="text-3xl font-bold text-green-400">{stats.successful}</div>
          <div className="text-xs text-zinc-500 mt-1">Successful</div>
        </div>
        <div className="glass-card p-4 text-center">
          <div className="text-3xl font-bold text-red-400">{stats.failed}</div>
          <div className="text-xs text-zinc-500 mt-1">Failed</div>
        </div>
        <div className="glass-card p-4 text-center">
          <div className="text-3xl font-bold text-yellow-400">{stats.pending}</div>
          <div className="text-xs text-zinc-500 mt-1">Pending</div>
        </div>
        <div className="glass-card p-4 text-center">
          <div className="text-3xl font-bold text-cyan-400">{stats.accuracy.toFixed(1)}%</div>
          <div className="text-xs text-zinc-500 mt-1">Accuracy</div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-4">
        {/* Asset Filter */}
        <div className="glass-card p-1 flex">
          {assets.map((asset) => (
            <button
              key={asset.value}
              onClick={() => setSelectedAsset(asset.value)}
              className={cn(
                'px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-1.5',
                selectedAsset === asset.value
                  ? 'bg-gradient-to-r from-amber-500/20 to-yellow-500/20 text-amber-400'
                  : 'text-zinc-400 hover:text-white'
              )}
            >
              <span>{asset.icon}</span>
              <span>{asset.label}</span>
            </button>
          ))}
        </div>

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
            <p className="text-zinc-400">No predictions found.</p>
            <p className="text-xs text-zinc-600 mt-2">
              Predictions are generated during trading hours (9 AM - 11:30 PM IST, Mon-Fri).
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/5">
                  <th className="text-left text-xs text-zinc-500 font-medium p-4">Time</th>
                  <th className="text-left text-xs text-zinc-500 font-medium p-4">Market</th>
                  <th className="text-left text-xs text-zinc-500 font-medium p-4">Contract</th>
                  <th className="text-left text-xs text-zinc-500 font-medium p-4">Type</th>
                  <th className="text-left text-xs text-zinc-500 font-medium p-4">Direction</th>
                  <th className="text-right text-xs text-zinc-500 font-medium p-4">Current</th>
                  <th className="text-right text-xs text-zinc-500 font-medium p-4">Predicted</th>
                  <th className="text-right text-xs text-zinc-500 font-medium p-4">Actual</th>
                  <th className="text-right text-xs text-zinc-500 font-medium p-4">Error %</th>
                  <th className="text-center text-xs text-zinc-500 font-medium p-4">Result</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((prediction, index) => {
                  const verification = prediction.verification;
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
                        <span className={cn(
                          'inline-flex px-2 py-1 rounded-full text-xs font-medium',
                          prediction.market === 'mcx'
                            ? 'bg-orange-500/20 text-orange-400'
                            : 'bg-blue-500/20 text-blue-400'
                        )}>
                          {prediction.market.toUpperCase()}
                        </span>
                      </td>
                      <td className="p-4">
                        {prediction.contract_type ? (
                          <div>
                            <span className="text-sm text-zinc-300">{prediction.contract_type}</span>
                            {prediction.trading_symbol && (
                              <span className="block text-[10px] text-zinc-500" title={prediction.trading_symbol}>
                                {(() => {
                                  const parts = prediction.trading_symbol.split(' ');
                                  return parts.length >= 5 ? `${parts[2]} ${parts[3]} ${parts[4]}` : '';
                                })()}
                              </span>
                            )}
                          </div>
                        ) : (
                          <span className="text-sm text-zinc-500">-</span>
                        )}
                      </td>
                      <td className="p-4">
                        <span className="text-sm text-zinc-300">
                          {getIntervalLabel(prediction.interval)}
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
                          {formatPrice(prediction.current_price, prediction.market)}
                        </span>
                      </td>
                      <td className="p-4 text-right">
                        <span className="text-sm text-cyan-400 font-medium">
                          {formatPrice(prediction.predicted_price, prediction.market)}
                        </span>
                      </td>
                      <td className="p-4 text-right">
                        {verification && verification.actual_price != null ? (
                          <span className={cn(
                            'text-sm font-medium',
                            verification.is_direction_correct ? 'text-green-400' : 'text-red-400'
                          )}>
                            {formatPrice(verification.actual_price, prediction.market)}
                          </span>
                        ) : (
                          <span className="text-sm text-zinc-500">-</span>
                        )}
                      </td>
                      <td className="p-4 text-right">
                        {verification && verification.price_error_percent != null ? (
                          <span className={cn(
                            'text-sm font-medium',
                            Math.abs(verification.price_error_percent) < 1 ? 'text-green-400' :
                            Math.abs(verification.price_error_percent) < 2 ? 'text-yellow-400' : 'text-red-400'
                          )}>
                            {verification.price_error_percent > 0 ? '+' : ''}
                            {verification.price_error_percent.toFixed(2)}%
                          </span>
                        ) : (
                          <span className="text-sm text-zinc-500">-</span>
                        )}
                      </td>
                      <td className="p-4 text-center">
                        {verification ? (
                          <span className={cn(
                            'inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium',
                            verification.is_direction_correct
                              ? 'bg-green-500/20 text-green-400'
                              : 'bg-red-500/20 text-red-400'
                          )}>
                            {verification.is_direction_correct ? '✓ Success' : '✗ Failed'}
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium bg-yellow-500/20 text-yellow-400">
                            ⏳ Pending
                          </span>
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
    </div>
  );
}
