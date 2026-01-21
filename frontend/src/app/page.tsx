'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { getHealth, getModelsHealth, getLatestPrediction, getAccuracySummary } from '@/lib/api';
import type { HealthStatus, Prediction, AccuracySummary } from '@/lib/types';

export default function HomePage() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [modelsHealth, setModelsHealth] = useState<HealthStatus | null>(null);
  const [latestPrediction, setLatestPrediction] = useState<Prediction | null>(null);
  const [accuracy, setAccuracy] = useState<AccuracySummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        const [healthData, modelsData, predictionData, accuracyData] = await Promise.all([
          getHealth().catch(() => null),
          getModelsHealth().catch(() => null),
          getLatestPrediction().catch(() => null),
          getAccuracySummary().catch(() => null),
        ]);

        setHealth(healthData);
        setModelsHealth(modelsData);
        setLatestPrediction(predictionData);
        setAccuracy(accuracyData);
      } catch (error) {
        console.error('Error loading data:', error);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="spinner"></div>
      </div>
    );
  }

  const accuracyValue = accuracy?.direction_accuracy?.overall ?? 0;
  const accuracyPercent = (accuracyValue * 100).toFixed(1);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero Section */}
      <div className="text-center mb-16 relative">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 mb-6">
          <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-sm text-zinc-400">Real-time ML Predictions</span>
        </div>

        <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold mb-6">
          <span className="text-white">Silver Price</span>
          <br />
          <span className="gradient-text">Prediction System</span>
        </h1>

        <p className="text-lg text-zinc-400 max-w-2xl mx-auto mb-8">
          Advanced ensemble ML models combining Prophet, LSTM, and XGBoost for
          accurate MCX & COMEX silver price forecasting with confidence intervals.
        </p>

        <div className="flex flex-wrap items-center justify-center gap-4">
          <Link href="/dashboard" className="btn-primary">
            View Dashboard
          </Link>
          <Link href="/predictions" className="btn-secondary">
            Prediction History
          </Link>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
        <div className="stat-item">
          <div className="stat-value text-cyan-400">{accuracyPercent}%</div>
          <div className="stat-label">Accuracy</div>
        </div>
        <div className="stat-item">
          <div className="stat-value text-purple-400">{accuracy?.total_predictions ?? 0}</div>
          <div className="stat-label">Predictions</div>
        </div>
        <div className="stat-item">
          <div className="stat-value text-pink-400">{modelsHealth?.trained_models ?? 0}</div>
          <div className="stat-label">Models</div>
        </div>
        <div className="stat-item">
          <div className={`stat-value ${health?.status === 'healthy' ? 'text-green-400' : 'text-yellow-400'}`}>
            {health?.status === 'healthy' ? 'Online' : 'Pending'}
          </div>
          <div className="stat-label">Status</div>
        </div>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        {/* System Status */}
        <div className="card card-hover">
          <div className="flex items-start justify-between mb-4">
            <div className="p-3 rounded-xl bg-gradient-to-br from-green-500/20 to-emerald-500/10 border border-green-500/20">
              <svg className="w-6 h-6 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <span className={health?.status === 'healthy' ? 'badge-success' : 'badge-warning'}>
              {health?.status || 'Unknown'}
            </span>
          </div>
          <h3 className="text-lg font-semibold text-white mb-1">System Status</h3>
          <p className="text-sm text-zinc-400">
            Environment: <span className="text-zinc-300">{health?.environment || 'N/A'}</span>
          </p>
        </div>

        {/* ML Models */}
        <div className="card card-hover">
          <div className="flex items-start justify-between mb-4">
            <div className="p-3 rounded-xl bg-gradient-to-br from-purple-500/20 to-violet-500/10 border border-purple-500/20">
              <svg className="w-6 h-6 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
            </div>
            <span className={modelsHealth?.status === 'healthy' ? 'badge-success' : 'badge-warning'}>
              {modelsHealth?.status === 'healthy' ? 'trained' : 'needs_training'}
            </span>
          </div>
          <h3 className="text-lg font-semibold text-white mb-1">ML Models</h3>
          <p className="text-sm text-zinc-400">
            Trained models: <span className="text-zinc-300">{modelsHealth?.trained_models ?? 0}</span>
          </p>
        </div>

        {/* Accuracy */}
        <div className="card card-hover">
          <div className="flex items-start justify-between mb-4">
            <div className="p-3 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-500/10 border border-cyan-500/20">
              <svg className="w-6 h-6 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <span className="text-3xl font-bold gradient-text">{accuracyPercent}%</span>
          </div>
          <h3 className="text-lg font-semibold text-white mb-1">Direction Accuracy</h3>
          <p className="text-sm text-zinc-400">
            {accuracy ? `${accuracy.direction_accuracy?.correct ?? 0} correct / ${accuracy.total_predictions} total` : 'No data'}
          </p>
        </div>
      </div>

      {/* Latest Prediction */}
      <div className="card mb-12">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold text-white">Latest Prediction</h3>
          {latestPrediction?.predicted_direction && (
            <span className={`badge-${latestPrediction.predicted_direction}`}>
              {latestPrediction.predicted_direction.toUpperCase()}
            </span>
          )}
        </div>

        {latestPrediction && latestPrediction.predicted_direction ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div>
              <p className="text-sm text-zinc-500 mb-1">Current Price</p>
              <p className="text-2xl font-bold text-white font-mono">
                ₹{latestPrediction.current_price?.toLocaleString(undefined, { minimumFractionDigits: 2 }) ?? 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-sm text-zinc-500 mb-1">Predicted Price</p>
              <p className={`text-2xl font-bold font-mono ${
                latestPrediction.predicted_direction === 'bullish' ? 'text-green-400' :
                latestPrediction.predicted_direction === 'bearish' ? 'text-red-400' : 'text-yellow-400'
              }`}>
                ₹{latestPrediction.predicted_price?.toLocaleString(undefined, { minimumFractionDigits: 2 }) ?? 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-sm text-zinc-500 mb-1">Confidence</p>
              <p className="text-2xl font-bold text-cyan-400">
                {latestPrediction.direction_confidence
                  ? (latestPrediction.direction_confidence * 100).toFixed(1) + '%'
                  : 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-sm text-zinc-500 mb-1">Target Time</p>
              <p className="text-lg font-medium text-zinc-300">
                {latestPrediction.target_time
                  ? new Date(latestPrediction.target_time).toLocaleString(undefined, {
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                    })
                  : 'N/A'}
              </p>
            </div>
          </div>
        ) : (
          <div className="empty-state">
            <svg className="empty-state-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <p className="empty-state-title">No Predictions Available</p>
            <p className="empty-state-description">
              Train the models first to generate predictions. Visit the dashboard to start training.
            </p>
          </div>
        )}
      </div>

      {/* Quick Links */}
      <h3 className="text-xl font-semibold text-white mb-6">Quick Actions</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Link href="/dashboard" className="card card-hover group">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-500/10 border border-cyan-500/20 group-hover:border-cyan-500/40 transition-colors">
              <svg className="w-6 h-6 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <div>
              <h4 className="text-lg font-semibold text-white group-hover:text-cyan-400 transition-colors">Dashboard</h4>
              <p className="text-sm text-zinc-500">Real-time charts & predictions</p>
            </div>
          </div>
          <p className="text-sm text-zinc-400">
            View live price charts, current predictions, and market data for MCX and COMEX silver.
          </p>
        </Link>

        <Link href="/accuracy" className="card card-hover group">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 rounded-xl bg-gradient-to-br from-purple-500/20 to-violet-500/10 border border-purple-500/20 group-hover:border-purple-500/40 transition-colors">
              <svg className="w-6 h-6 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <div>
              <h4 className="text-lg font-semibold text-white group-hover:text-purple-400 transition-colors">Accuracy Metrics</h4>
              <p className="text-sm text-zinc-500">Track model performance</p>
            </div>
          </div>
          <p className="text-sm text-zinc-400">
            Monitor prediction accuracy, CI coverage, error metrics, and performance trends over time.
          </p>
        </Link>

        <Link href="/predictions" className="card card-hover group">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 rounded-xl bg-gradient-to-br from-pink-500/20 to-rose-500/10 border border-pink-500/20 group-hover:border-pink-500/40 transition-colors">
              <svg className="w-6 h-6 text-pink-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <h4 className="text-lg font-semibold text-white group-hover:text-pink-400 transition-colors">Prediction History</h4>
              <p className="text-sm text-zinc-500">View past predictions</p>
            </div>
          </div>
          <p className="text-sm text-zinc-400">
            Browse all past predictions with their verification results and detailed analysis.
          </p>
        </Link>
      </div>

      {/* Disclaimer */}
      <div className="mt-12 p-6 rounded-2xl bg-gradient-to-br from-yellow-500/10 to-orange-500/5 border border-yellow-500/20">
        <div className="flex items-start gap-4">
          <div className="p-2 rounded-lg bg-yellow-500/20">
            <svg className="w-5 h-5 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <div>
            <h4 className="text-sm font-semibold text-yellow-400 mb-1">Disclaimer</h4>
            <p className="text-sm text-zinc-400">
              This is a decision support tool, not financial advice. Past performance does not guarantee future results.
              Predictions are probabilistic, not certainties. Always use proper risk management.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
