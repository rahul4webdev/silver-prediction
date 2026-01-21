'use client';

import { useEffect, useState, useRef } from 'react';
import { getPriceData, getLatestPrediction, getPredictions } from '@/lib/api';
import type { PriceCandle, Prediction } from '@/lib/types';
import PredictionChart from '@/components/charts/PredictionChart';

export default function DashboardPage() {
  const [market, setMarket] = useState<'mcx' | 'comex'>('mcx');
  const [interval, setInterval] = useState<'30m' | '1h' | '4h' | 'daily'>('30m');
  const [priceData, setPriceData] = useState<PriceCandle[]>([]);
  const [latestPrediction, setLatestPrediction] = useState<Prediction | null>(null);
  const [recentPredictions, setRecentPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      setError(null);

      try {
        const [prices, prediction, predictions] = await Promise.all([
          getPriceData('silver', market, interval, 500).catch(() => []),
          getLatestPrediction('silver', market, interval).catch(() => null),
          getPredictions('silver', market, interval, 10).catch(() => []),
        ]);

        setPriceData(prices);
        setLatestPrediction(prediction);
        setRecentPredictions(predictions);
      } catch (err) {
        setError('Failed to load data');
        console.error('Error loading data:', err);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, [market, interval]);

  // WebSocket connection for real-time updates
  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'wss://predictionapi.gahfaudio.in';

    try {
      const ws = new WebSocket(`${wsUrl}/ws/prices/silver`);

      ws.onopen = () => {
        ws.send(JSON.stringify({ action: 'subscribe', market, interval }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'price_update' && data.market === market) {
            setPriceData((prev) => {
              const newCandle: PriceCandle = {
                timestamp: data.timestamp,
                open: data.open,
                high: data.high,
                low: data.low,
                close: data.close,
                volume: data.volume,
              };
              const updated = [...prev];
              if (updated.length > 0) {
                updated[updated.length - 1] = newCandle;
              }
              return updated;
            });
          } else if (data.type === 'new_prediction') {
            setLatestPrediction(data.prediction);
          }
        } catch {
          // Ignore parse errors
        }
      };

      ws.onerror = () => {
        console.warn('WebSocket connection error');
      };

      wsRef.current = ws;

      return () => {
        ws.close();
      };
    } catch {
      console.warn('WebSocket initialization failed');
    }
  }, [market, interval]);

  const currentPrice = priceData.length > 0 ? priceData[priceData.length - 1].close : null;
  const previousPrice = priceData.length > 1 ? priceData[priceData.length - 2].close : null;
  const priceChange = currentPrice && previousPrice ? currentPrice - previousPrice : null;
  const priceChangePercent = priceChange && previousPrice ? (priceChange / previousPrice) * 100 : null;

  const currency = market === 'mcx' ? '₹' : '$';

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Silver Dashboard</h1>
          <p className="text-sm text-zinc-400">
            Real-time price chart with ML predictions
          </p>
        </div>
        <div className="flex items-center gap-3 mt-4 sm:mt-0">
          {/* Market Selector */}
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
          {/* Interval Selector */}
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
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {/* Price Header */}
      <div className="card mb-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <p className="text-sm text-zinc-500 mb-1">Current Price ({market.toUpperCase()})</p>
            <div className="flex items-baseline gap-4">
              <span className="text-4xl font-bold text-white font-mono">
                {currentPrice
                  ? `${currency}${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}`
                  : '--'}
              </span>
              {priceChange !== null && (
                <span className={`text-lg font-semibold ${priceChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {priceChange >= 0 ? '+' : ''}{currency}{Math.abs(priceChange).toFixed(2)}
                  <span className="text-sm ml-1">
                    ({priceChangePercent !== null ? `${priceChangePercent >= 0 ? '+' : ''}${priceChangePercent.toFixed(2)}%` : '--'})
                  </span>
                </span>
              )}
            </div>
          </div>

          {/* Quick Stats */}
          <div className="flex items-center gap-6">
            <div className="text-center">
              <p className="text-xs text-zinc-500 uppercase tracking-wider">24h High</p>
              <p className="text-lg font-semibold text-green-400 font-mono">
                {priceData.length > 0
                  ? `${currency}${Math.max(...priceData.slice(-48).map((c) => c.high)).toLocaleString(undefined, { minimumFractionDigits: 2 })}`
                  : '--'}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-zinc-500 uppercase tracking-wider">24h Low</p>
              <p className="text-lg font-semibold text-red-400 font-mono">
                {priceData.length > 0
                  ? `${currency}${Math.min(...priceData.slice(-48).map((c) => c.low)).toLocaleString(undefined, { minimumFractionDigits: 2 })}`
                  : '--'}
              </p>
            </div>
            <div className="text-center">
              <p className="text-xs text-zinc-500 uppercase tracking-wider">24h Vol</p>
              <p className="text-lg font-semibold text-zinc-300 font-mono">
                {priceData.length > 0
                  ? priceData.slice(-48).reduce((sum, c) => sum + c.volume, 0).toLocaleString()
                  : '--'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Chart Section - 3 columns */}
        <div className="lg:col-span-3">
          <div className="card p-0 overflow-hidden">
            {loading ? (
              <div className="flex items-center justify-center h-[500px]">
                <div className="spinner"></div>
              </div>
            ) : priceData.length > 0 ? (
              <PredictionChart
                priceData={priceData}
                prediction={latestPrediction}
                market={market}
                interval={interval}
              />
            ) : (
              <div className="flex flex-col items-center justify-center h-[500px] text-center">
                <svg className="w-16 h-16 text-zinc-600 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <p className="text-lg font-semibold text-zinc-400">No Price Data Available</p>
                <p className="text-sm text-zinc-500 mt-1">Price data will appear when the market is active</p>
              </div>
            )}
          </div>
        </div>

        {/* Prediction Panel - 1 column */}
        <div className="lg:col-span-1 space-y-6">
          {/* Latest Prediction */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Latest Prediction</h3>
              {latestPrediction?.predicted_direction && (
                <span className={`badge-${latestPrediction.predicted_direction}`}>
                  {latestPrediction.predicted_direction.toUpperCase()}
                </span>
              )}
            </div>

            {latestPrediction && latestPrediction.predicted_direction ? (
              <div className="space-y-4">
                {/* Price Prediction */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Current</p>
                    <p className="text-lg font-bold text-white font-mono">
                      {currency}{latestPrediction.current_price?.toLocaleString(undefined, { minimumFractionDigits: 2 }) ?? 'N/A'}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Predicted</p>
                    <p className={`text-lg font-bold font-mono ${
                      latestPrediction.predicted_direction === 'bullish' ? 'text-green-400' :
                      latestPrediction.predicted_direction === 'bearish' ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                      {currency}{latestPrediction.predicted_price?.toLocaleString(undefined, { minimumFractionDigits: 2 }) ?? 'N/A'}
                    </p>
                  </div>
                </div>

                {/* Change */}
                <div className="p-3 rounded-xl bg-white/5 text-center">
                  <span className={`text-xl font-bold ${
                    (latestPrediction.predicted_price ?? 0) >= (latestPrediction.current_price ?? 0)
                      ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {((latestPrediction.predicted_price ?? 0) - (latestPrediction.current_price ?? 0)) >= 0 ? '+' : ''}
                    {currency}{((latestPrediction.predicted_price ?? 0) - (latestPrediction.current_price ?? 0)).toFixed(2)}
                  </span>
                </div>

                {/* Confidence */}
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-zinc-500">Confidence</span>
                    <span className="font-semibold text-cyan-400">
                      {latestPrediction.direction_confidence
                        ? (latestPrediction.direction_confidence * 100).toFixed(1) + '%'
                        : 'N/A'}
                    </span>
                  </div>
                  <div className="progress-bar">
                    <div
                      className="progress-bar-fill progress-bar-fill-blue"
                      style={{ width: `${(latestPrediction.direction_confidence ?? 0) * 100}%` }}
                    />
                  </div>
                </div>

                {/* CI Ranges */}
                <div className="border-t border-white/5 pt-4 space-y-2">
                  <p className="text-xs text-zinc-500 uppercase tracking-wider mb-2">Confidence Intervals</p>
                  <div className="flex justify-between text-sm">
                    <span className="text-zinc-500">50% CI</span>
                    <span className="font-mono text-zinc-300">
                      {currency}{(latestPrediction.ci_50_lower ?? 0).toFixed(2)} - {currency}{(latestPrediction.ci_50_upper ?? 0).toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-zinc-500">80% CI</span>
                    <span className="font-mono text-zinc-300">
                      {currency}{(latestPrediction.ci_80_lower ?? 0).toFixed(2)} - {currency}{(latestPrediction.ci_80_upper ?? 0).toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-zinc-500">95% CI</span>
                    <span className="font-mono text-zinc-300">
                      {currency}{(latestPrediction.ci_95_lower ?? 0).toFixed(2)} - {currency}{(latestPrediction.ci_95_upper ?? 0).toFixed(2)}
                    </span>
                  </div>
                </div>

                {/* Target Time */}
                <div className="border-t border-white/5 pt-4">
                  <div className="flex justify-between text-sm">
                    <span className="text-zinc-500">Target Time</span>
                    <span className="text-zinc-300">
                      {latestPrediction.target_time
                        ? new Date(latestPrediction.target_time).toLocaleString(undefined, {
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit',
                          })
                        : 'N/A'}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <svg className="w-12 h-12 text-zinc-600 mx-auto mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
                <p className="text-sm text-zinc-500">No prediction available</p>
              </div>
            )}
          </div>

          {/* Recent Predictions */}
          <div className="card">
            <h3 className="text-lg font-semibold text-white mb-4">Recent Predictions</h3>
            {recentPredictions.length > 0 ? (
              <div className="space-y-3">
                {recentPredictions.slice(0, 5).map((pred) => (
                  <div
                    key={pred.id}
                    className="flex items-center justify-between py-2 border-b border-white/5 last:border-0"
                  >
                    <div>
                      {pred.predicted_direction ? (
                        <span className={`badge-${pred.predicted_direction} text-xs`}>
                          {pred.predicted_direction.toUpperCase()}
                        </span>
                      ) : (
                        <span className="text-xs text-zinc-500">Unknown</span>
                      )}
                      <p className="text-xs text-zinc-500 mt-1">
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
                    <div className="text-right">
                      {pred.verified_at ? (
                        <span className={`text-sm font-medium ${
                          pred.is_direction_correct ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {pred.is_direction_correct ? '✓ Correct' : '✗ Wrong'}
                        </span>
                      ) : (
                        <span className="badge-info text-xs">Pending</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-zinc-500 text-center py-4">No recent predictions</p>
            )}
          </div>

          {/* Model Weights */}
          {latestPrediction?.model_weights && (
            <div className="card">
              <h3 className="text-lg font-semibold text-white mb-4">Model Weights</h3>
              <div className="space-y-3">
                {Object.entries(latestPrediction.model_weights).map(([model, weight]) => (
                  <div key={model}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="capitalize text-zinc-400">{model}</span>
                      <span className="font-medium text-zinc-300">{((weight as number) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="progress-bar">
                      <div
                        className="progress-bar-fill progress-bar-fill-blue"
                        style={{ width: `${(weight as number) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
