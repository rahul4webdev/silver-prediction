'use client';

import { useEffect, useState } from 'react';
import { getAccuracySummary, getAccuracyTrend, getPredictions } from '@/lib/api';
import type { AccuracySummary, Prediction } from '@/lib/types';

export default function AccuracyPage() {
  const [period, setPeriod] = useState<'7d' | '30d' | '90d' | 'all'>('30d');
  const [accuracy, setAccuracy] = useState<AccuracySummary | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const periodDays =
          period === '7d' ? 7 : period === '30d' ? 30 : period === '90d' ? 90 : 365;

        const [accuracyData, predictionsData] = await Promise.all([
          getAccuracySummary('silver', undefined, undefined, periodDays).catch(() => null),
          getPredictions('silver', undefined, undefined, 50, true).catch(() => []),
        ]);

        setAccuracy(accuracyData);
        setPredictions(predictionsData);
      } catch (error) {
        console.error('Error loading accuracy data:', error);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, [period]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="spinner"></div>
      </div>
    );
  }

  // Safe access with defaults
  const directionAccuracy = accuracy?.direction_accuracy?.overall ?? 0;
  const totalPredictions = accuracy?.total_predictions ?? 0;
  const verifiedPredictions = accuracy?.verified_predictions ?? 0;
  const correctPredictions = accuracy?.direction_accuracy?.correct ?? 0;

  // Handle both ci_coverage and confidence_interval_coverage
  const ciCoverage = accuracy?.ci_coverage || accuracy?.confidence_interval_coverage || {
    ci_50: 0,
    ci_80: 0,
    ci_95: 0,
  };

  // Safe access for streaks
  const streaks = accuracy?.streaks || {
    current: { type: 'win' as const, count: 0 },
    best_win: 0,
    worst_loss: 0,
  };

  // Safe access for error metrics
  const errorMetrics = accuracy?.error_metrics || {
    mae: 0,
    mape: 0,
    rmse: 0,
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Accuracy Metrics</h1>
          <p className="text-sm text-zinc-400">
            Track prediction accuracy and model performance
          </p>
        </div>
        <div className="mt-4 sm:mt-0">
          <div className="segmented-control">
            {(['7d', '30d', '90d', 'all'] as const).map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={period === p ? 'active' : ''}
              >
                {p === 'all' ? 'All' : p.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Accuracy Gauge */}
      <div className="card mb-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-center">
          {/* Gauge */}
          <div className="flex flex-col items-center">
            <div className="relative w-48 h-48">
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 120 120">
                <defs>
                  <linearGradient id="gradient-accuracy" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#00d4ff" />
                    <stop offset="100%" stopColor="#7c3aed" />
                  </linearGradient>
                </defs>
                <circle
                  cx="60"
                  cy="60"
                  r="50"
                  fill="none"
                  stroke="rgba(255,255,255,0.1)"
                  strokeWidth="10"
                />
                <circle
                  cx="60"
                  cy="60"
                  r="50"
                  fill="none"
                  stroke="url(#gradient-accuracy)"
                  strokeWidth="10"
                  strokeLinecap="round"
                  strokeDasharray={`${directionAccuracy * 314} 314`}
                  style={{ transition: 'stroke-dasharray 1s ease-out' }}
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-4xl font-bold gradient-text">
                  {(directionAccuracy * 100).toFixed(1)}%
                </span>
                <span className="text-sm text-zinc-500">Direction Accuracy</span>
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 gap-4">
            <div className="stat-item">
              <div className="stat-value text-green-400">{correctPredictions}</div>
              <div className="stat-label">Correct</div>
            </div>
            <div className="stat-item">
              <div className="stat-value text-red-400">{totalPredictions - correctPredictions}</div>
              <div className="stat-label">Wrong</div>
            </div>
            <div className="stat-item">
              <div className="stat-value text-cyan-400">{totalPredictions}</div>
              <div className="stat-label">Total</div>
            </div>
            <div className="stat-item">
              <div className="stat-value text-purple-400">{verifiedPredictions}</div>
              <div className="stat-label">Verified</div>
            </div>
          </div>

          {/* Streak */}
          <div className="text-center lg:text-right">
            <div className="inline-block p-6 rounded-2xl bg-gradient-to-br from-white/5 to-white/0 border border-white/10">
              <p className="text-sm text-zinc-500 mb-2">Current Streak</p>
              <p className={`text-4xl font-bold ${
                streaks.current.type === 'win' ? 'text-green-400' : 'text-red-400'
              }`}>
                {streaks.current.count} {streaks.current.type === 'win' ? 'Wins' : 'Losses'}
              </p>
              <p className="text-sm text-zinc-500 mt-2">
                Best: <span className="text-green-400">{streaks.best_win} wins</span>
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* CI Coverage & Error Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* CI Coverage */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-6">Confidence Interval Coverage</h3>
          <p className="text-sm text-zinc-500 mb-6">
            How well-calibrated are our confidence intervals? Target coverage should match the CI percentage.
          </p>
          <div className="space-y-6">
            {[
              { label: '50% CI', value: ciCoverage.ci_50, target: 0.5, color: 'from-blue-500 to-blue-400' },
              { label: '80% CI', value: ciCoverage.ci_80, target: 0.8, color: 'from-purple-500 to-purple-400' },
              { label: '95% CI', value: ciCoverage.ci_95, target: 0.95, color: 'from-pink-500 to-pink-400' },
            ].map((ci) => (
              <div key={ci.label}>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-zinc-400">{ci.label}</span>
                  <div className="flex items-center gap-2">
                    <span className={`font-semibold ${
                      Math.abs((ci.value ?? 0) - ci.target) < 0.1 ? 'text-green-400' : 'text-yellow-400'
                    }`}>
                      {((ci.value ?? 0) * 100).toFixed(1)}%
                    </span>
                    <span className="text-zinc-600">/</span>
                    <span className="text-zinc-500">{(ci.target * 100).toFixed(0)}%</span>
                  </div>
                </div>
                <div className="relative h-3 rounded-full bg-white/5 overflow-hidden">
                  {/* Target marker */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-white/30"
                    style={{ left: `${ci.target * 100}%` }}
                  />
                  {/* Actual value */}
                  <div
                    className={`h-full rounded-full bg-gradient-to-r ${ci.color} transition-all duration-1000`}
                    style={{ width: `${(ci.value ?? 0) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Error Metrics */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-6">Error Metrics</h3>
          <p className="text-sm text-zinc-500 mb-6">
            Quantitative measures of prediction accuracy. Lower values indicate better performance.
          </p>
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 rounded-xl bg-white/5 border border-white/5">
              <p className="text-3xl font-bold text-cyan-400">{errorMetrics.mae?.toFixed(2) ?? '0.00'}</p>
              <p className="text-xs text-zinc-500 mt-1 uppercase tracking-wider">MAE</p>
              <p className="text-xs text-zinc-600">Mean Absolute Error</p>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/5">
              <p className="text-3xl font-bold text-purple-400">{errorMetrics.mape?.toFixed(2) ?? '0.00'}%</p>
              <p className="text-xs text-zinc-500 mt-1 uppercase tracking-wider">MAPE</p>
              <p className="text-xs text-zinc-600">Mean Absolute % Error</p>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/5">
              <p className="text-3xl font-bold text-pink-400">{errorMetrics.rmse?.toFixed(2) ?? '0.00'}</p>
              <p className="text-xs text-zinc-500 mt-1 uppercase tracking-wider">RMSE</p>
              <p className="text-xs text-zinc-600">Root Mean Square Error</p>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/5">
              <p className="text-3xl font-bold text-green-400">
                {((accuracy?.direction_accuracy?.bullish_accuracy ?? 0) * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-zinc-500 mt-1 uppercase tracking-wider">Bullish</p>
              <p className="text-xs text-zinc-600">Bullish Direction Accuracy</p>
            </div>
          </div>
        </div>
      </div>

      {/* Breakdown by Market/Interval */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* By Market */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">By Market</h3>
          {Object.keys(accuracy?.by_market || {}).length > 0 ? (
            <div className="space-y-4">
              {Object.entries(accuracy?.by_market || {}).map(([market, data]) => (
                <div key={market} className="flex items-center justify-between p-4 rounded-xl bg-white/5">
                  <div>
                    <p className="font-semibold text-white">{market.toUpperCase()}</p>
                    <p className="text-sm text-zinc-500">{data.total || data.predictions} predictions</p>
                  </div>
                  <div className="text-right">
                    <p className={`text-2xl font-bold ${
                      data.accuracy >= 0.55 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {(data.accuracy * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-zinc-500">{data.correct || Math.round(data.accuracy * (data.total || data.predictions))} correct</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-zinc-500">No market data available</p>
            </div>
          )}
        </div>

        {/* By Interval */}
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">By Interval</h3>
          {Object.keys(accuracy?.by_interval || {}).length > 0 ? (
            <div className="space-y-4">
              {Object.entries(accuracy?.by_interval || {}).map(([interval, data]) => (
                <div key={interval} className="flex items-center justify-between p-4 rounded-xl bg-white/5">
                  <div>
                    <p className="font-semibold text-white">{interval}</p>
                    <p className="text-sm text-zinc-500">{data.total || data.predictions} predictions</p>
                  </div>
                  <div className="text-right">
                    <p className={`text-2xl font-bold ${
                      data.accuracy >= 0.55 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {(data.accuracy * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-zinc-500">{data.correct || Math.round(data.accuracy * (data.total || data.predictions))} correct</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-zinc-500">No interval data available</p>
            </div>
          )}
        </div>
      </div>

      {/* Recent Verified Predictions */}
      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-6">Recent Verified Predictions</h3>
        {predictions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Direction</th>
                  <th>Predicted</th>
                  <th>Actual</th>
                  <th>Error</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {predictions.slice(0, 10).map((pred) => (
                  <tr key={pred.id}>
                    <td>
                      {pred.target_time
                        ? new Date(pred.target_time).toLocaleString(undefined, {
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit',
                          })
                        : 'N/A'}
                    </td>
                    <td>
                      {pred.predicted_direction ? (
                        <span className={`badge-${pred.predicted_direction}`}>
                          {pred.predicted_direction.toUpperCase()}
                        </span>
                      ) : (
                        <span className="text-zinc-500">-</span>
                      )}
                    </td>
                    <td className="font-mono">{pred.predicted_price?.toFixed(2) ?? '-'}</td>
                    <td className="font-mono">{pred.actual_price?.toFixed(2) ?? '-'}</td>
                    <td className={`font-mono ${
                      Math.abs(pred.price_error_percent ?? 0) < 1 ? 'text-green-400' : 'text-yellow-400'
                    }`}>
                      {pred.price_error_percent !== undefined && pred.price_error_percent !== null
                        ? `${pred.price_error_percent >= 0 ? '+' : ''}${pred.price_error_percent.toFixed(3)}%`
                        : '-'}
                    </td>
                    <td>
                      {pred.is_direction_correct !== null && pred.is_direction_correct !== undefined ? (
                        pred.is_direction_correct ? (
                          <span className="badge-success">✓ Correct</span>
                        ) : (
                          <span className="badge-error">✗ Wrong</span>
                        )
                      ) : (
                        <span className="badge-warning">Pending</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12">
            <svg className="w-16 h-16 text-zinc-600 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
            <p className="text-lg font-semibold text-zinc-400">No Verified Predictions</p>
            <p className="text-sm text-zinc-500 mt-1">Predictions will appear here after verification</p>
          </div>
        )}
      </div>
    </div>
  );
}
