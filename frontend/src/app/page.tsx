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
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero Section */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Silver Price Prediction System
        </h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          Real-time predictions for MCX and COMEX silver using ensemble ML models.
          Track accuracy, view predictions, and make informed decisions.
        </p>
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {/* System Status */}
        <div className="card">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-gray-900">System Status</h3>
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                health?.status === 'healthy'
                  ? 'bg-green-100 text-green-800'
                  : 'bg-yellow-100 text-yellow-800'
              }`}
            >
              {health?.status || 'Unknown'}
            </span>
          </div>
          <p className="text-sm text-gray-500 mt-2">
            Environment: {health?.environment || 'N/A'}
          </p>
        </div>

        {/* Models Status */}
        <div className="card">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-gray-900">ML Models</h3>
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                modelsHealth?.status === 'healthy'
                  ? 'bg-green-100 text-green-800'
                  : 'bg-yellow-100 text-yellow-800'
              }`}
            >
              {modelsHealth?.status || 'Unknown'}
            </span>
          </div>
          <p className="text-sm text-gray-500 mt-2">
            Trained models: {modelsHealth?.trained_models || 0}
          </p>
        </div>

        {/* Accuracy */}
        <div className="card">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-gray-900">Accuracy</h3>
            {accuracy && (
              <span className="text-2xl font-bold text-primary-600">
                {(accuracy.direction_accuracy.overall * 100).toFixed(1)}%
              </span>
            )}
          </div>
          <p className="text-sm text-gray-500 mt-2">
            {accuracy
              ? `${accuracy.direction_accuracy.correct} correct / ${accuracy.total_predictions} total`
              : 'No data'}
          </p>
        </div>
      </div>

      {/* Latest Prediction */}
      {latestPrediction && latestPrediction.predicted_direction ? (
        <div className="card mb-8">
          <h3 className="text-lg font-medium text-gray-900 mb-4">
            Latest Prediction
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-500">Direction</p>
              <span
                className={`badge-${latestPrediction.predicted_direction} mt-1`}
              >
                {latestPrediction.predicted_direction.toUpperCase()}
              </span>
            </div>
            <div>
              <p className="text-sm text-gray-500">Current Price</p>
              <p className="text-lg font-semibold">
                {latestPrediction.current_price?.toFixed(2) ?? 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Predicted Price</p>
              <p className="text-lg font-semibold">
                {latestPrediction.predicted_price?.toFixed(2) ?? 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Confidence</p>
              <p className="text-lg font-semibold">
                {latestPrediction.direction_confidence
                  ? (latestPrediction.direction_confidence * 100).toFixed(1) + '%'
                  : 'N/A'}
              </p>
            </div>
          </div>
          <div className="mt-4 text-sm text-gray-500">
            Target time:{' '}
            {latestPrediction.target_time
              ? new Date(latestPrediction.target_time).toLocaleString()
              : 'N/A'}
          </div>
        </div>
      ) : (
        <div className="card mb-8">
          <h3 className="text-lg font-medium text-gray-900 mb-4">
            Latest Prediction
          </h3>
          <p className="text-sm text-gray-500">
            No predictions available yet. Train the models first to generate predictions.
          </p>
        </div>
      )}

      {/* Quick Links */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Link href="/dashboard" className="card hover:shadow-md transition-shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Dashboard
          </h3>
          <p className="text-sm text-gray-600">
            View real-time charts, predictions, and market data.
          </p>
        </Link>

        <Link href="/accuracy" className="card hover:shadow-md transition-shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Accuracy Metrics
          </h3>
          <p className="text-sm text-gray-600">
            Track prediction accuracy, CI coverage, and model performance.
          </p>
        </Link>

        <Link href="/predictions" className="card hover:shadow-md transition-shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Prediction History
          </h3>
          <p className="text-sm text-gray-600">
            View past predictions and their verification results.
          </p>
        </Link>
      </div>

      {/* Disclaimer */}
      <div className="mt-12 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
        <h4 className="text-sm font-medium text-yellow-800">Disclaimer</h4>
        <p className="text-sm text-yellow-700 mt-1">
          This is a decision support tool, not financial advice. Past performance
          does not guarantee future results. Predictions are probabilistic, not
          certainties. Always use proper risk management.
        </p>
      </div>
    </div>
  );
}
