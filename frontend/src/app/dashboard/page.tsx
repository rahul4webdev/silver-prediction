'use client';

import { useEffect, useState, useRef } from 'react';
import { getPriceData, getLatestPrediction, getPredictions } from '@/lib/api';
import type { PriceCandle, Prediction } from '@/lib/types';
import PredictionChart from '@/components/charts/PredictionChart';
import PredictionCard from '@/components/widgets/PredictionCard';
import MarketSelector from '@/components/widgets/MarketSelector';
import IntervalSelector from '@/components/widgets/IntervalSelector';

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
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
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
            // Update last candle or add new one
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
  }, [market, interval]);

  const currentPrice = priceData.length > 0 ? priceData[priceData.length - 1].close : null;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Silver Price Dashboard</h1>
          <p className="text-sm text-gray-500 mt-1">
            Real-time price chart with ML predictions
          </p>
        </div>
        <div className="flex items-center gap-4 mt-4 sm:mt-0">
          <MarketSelector value={market} onChange={setMarket} />
          <IntervalSelector value={interval} onChange={setInterval} />
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Chart Section - 3 columns */}
        <div className="lg:col-span-3">
          <div className="card p-0 overflow-hidden">
            {loading ? (
              <div className="flex items-center justify-center h-[500px]">
                <div className="spinner"></div>
              </div>
            ) : (
              <PredictionChart
                priceData={priceData}
                prediction={latestPrediction}
                market={market}
                interval={interval}
              />
            )}
          </div>

          {/* Price Stats */}
          {currentPrice && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-4">
              <div className="card">
                <p className="text-sm text-gray-500">Current Price</p>
                <p className="text-xl font-bold text-gray-900">
                  {market === 'mcx' ? '₹' : '$'}
                  {currentPrice.toLocaleString(undefined, {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </p>
              </div>
              <div className="card">
                <p className="text-sm text-gray-500">24h High</p>
                <p className="text-xl font-bold text-green-600">
                  {market === 'mcx' ? '₹' : '$'}
                  {priceData.length > 0
                    ? Math.max(...priceData.slice(-48).map((c) => c.high)).toLocaleString(
                        undefined,
                        { minimumFractionDigits: 2, maximumFractionDigits: 2 }
                      )
                    : '-'}
                </p>
              </div>
              <div className="card">
                <p className="text-sm text-gray-500">24h Low</p>
                <p className="text-xl font-bold text-red-600">
                  {market === 'mcx' ? '₹' : '$'}
                  {priceData.length > 0
                    ? Math.min(...priceData.slice(-48).map((c) => c.low)).toLocaleString(
                        undefined,
                        { minimumFractionDigits: 2, maximumFractionDigits: 2 }
                      )
                    : '-'}
                </p>
              </div>
              <div className="card">
                <p className="text-sm text-gray-500">24h Volume</p>
                <p className="text-xl font-bold text-gray-900">
                  {priceData.length > 0
                    ? priceData
                        .slice(-48)
                        .reduce((sum, c) => sum + c.volume, 0)
                        .toLocaleString()
                    : '-'}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Prediction Panel - 1 column */}
        <div className="lg:col-span-1 space-y-4">
          {/* Latest Prediction */}
          {latestPrediction ? (
            <PredictionCard prediction={latestPrediction} market={market} />
          ) : (
            <div className="card">
              <h3 className="text-lg font-medium text-gray-900 mb-2">Latest Prediction</h3>
              <p className="text-sm text-gray-500">No prediction available</p>
            </div>
          )}

          {/* Recent Predictions */}
          <div className="card">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Predictions</h3>
            {recentPredictions.length > 0 ? (
              <div className="space-y-3">
                {recentPredictions.slice(0, 5).map((pred) => (
                  <div
                    key={pred.id}
                    className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0"
                  >
                    <div>
                      <span
                        className={`badge-${pred.predicted_direction} text-xs`}
                      >
                        {pred.predicted_direction.toUpperCase()}
                      </span>
                      <p className="text-xs text-gray-500 mt-1">
                        {new Date(pred.target_time).toLocaleString(undefined, {
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit',
                        })}
                      </p>
                    </div>
                    <div className="text-right">
                      {pred.verified_at ? (
                        <span
                          className={`text-sm font-medium ${
                            pred.is_direction_correct ? 'text-green-600' : 'text-red-600'
                          }`}
                        >
                          {pred.is_direction_correct ? '✓ Correct' : '✗ Wrong'}
                        </span>
                      ) : (
                        <span className="text-sm text-gray-400">Pending</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-500">No recent predictions</p>
            )}
          </div>

          {/* Model Weights */}
          {latestPrediction?.model_weights && (
            <div className="card">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Model Weights</h3>
              <div className="space-y-3">
                {Object.entries(latestPrediction.model_weights).map(([model, weight]) => (
                  <div key={model}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="capitalize text-gray-600">{model}</span>
                      <span className="font-medium">{((weight as number) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-primary-600 h-2 rounded-full"
                        style={{ width: `${(weight as number) * 100}%` }}
                      ></div>
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
