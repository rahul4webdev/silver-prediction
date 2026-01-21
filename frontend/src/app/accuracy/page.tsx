'use client';

import { useEffect, useState } from 'react';
import { getAccuracySummary, getAccuracyTrend, getPredictions } from '@/lib/api';
import type { AccuracySummary, Prediction } from '@/lib/types';
import AccuracyCard from '@/components/accuracy/AccuracyCard';
import AccuracyTrendChart from '@/components/accuracy/AccuracyTrendChart';
import CICoverageChart from '@/components/accuracy/CICoverageChart';
import PredictionHistoryTable from '@/components/accuracy/PredictionHistoryTable';

export default function AccuracyPage() {
  const [period, setPeriod] = useState<'7d' | '30d' | '90d' | 'all'>('30d');
  const [accuracy, setAccuracy] = useState<AccuracySummary | null>(null);
  const [trend, setTrend] = useState<{ date: string; accuracy: number }[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const periodDays =
          period === '7d' ? 7 : period === '30d' ? 30 : period === '90d' ? 90 : 365;

        const [accuracyData, trendData, predictionsData] = await Promise.all([
          getAccuracySummary('silver', undefined, undefined, periodDays).catch(() => null),
          getAccuracyTrend('silver', periodDays).catch(() => []),
          getPredictions('silver', undefined, undefined, 50, true).catch(() => []),
        ]);

        setAccuracy(accuracyData);
        setTrend(trendData);
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
      <div className="flex items-center justify-center min-h-[50vh]">
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Accuracy Metrics</h1>
          <p className="text-sm text-gray-500 mt-1">
            Track prediction accuracy and model performance
          </p>
        </div>
        <div className="mt-4 sm:mt-0">
          <div className="flex rounded-lg border border-gray-200 overflow-hidden">
            {(['7d', '30d', '90d', 'all'] as const).map((p) => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className={`px-4 py-2 text-sm font-medium transition-colors ${
                  period === p
                    ? 'bg-primary-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                } ${p !== '7d' ? 'border-l border-gray-200' : ''}`}
              >
                {p === 'all' ? 'All Time' : p.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </div>

      {accuracy ? (
        <>
          {/* Key Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <AccuracyCard
              title="Direction Accuracy"
              value={`${(accuracy.direction_accuracy.overall * 100).toFixed(1)}%`}
              subtitle={`${accuracy.direction_accuracy.correct} / ${accuracy.total_predictions} correct`}
              trend={accuracy.direction_accuracy.overall >= 0.55 ? 'up' : 'down'}
              target="55%"
            />
            <AccuracyCard
              title="80% CI Coverage"
              value={`${(accuracy.ci_coverage.ci_80 * 100).toFixed(1)}%`}
              subtitle="Should be ~80%"
              trend={Math.abs(accuracy.ci_coverage.ci_80 - 0.8) < 0.1 ? 'neutral' : 'down'}
              target="80%"
            />
            <AccuracyCard
              title="Total Predictions"
              value={accuracy.total_predictions.toString()}
              subtitle={`${accuracy.verified_predictions} verified`}
            />
            <AccuracyCard
              title="Current Streak"
              value={
                accuracy.streaks.current.type === 'win'
                  ? `${accuracy.streaks.current.count} Wins`
                  : `${accuracy.streaks.current.count} Losses`
              }
              subtitle={`Best: ${accuracy.streaks.best_win} wins`}
              trend={accuracy.streaks.current.type === 'win' ? 'up' : 'down'}
            />
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            {/* Accuracy Trend */}
            <div className="card">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Accuracy Trend</h3>
              <AccuracyTrendChart data={trend} />
            </div>

            {/* CI Coverage */}
            <div className="card">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Confidence Interval Coverage
              </h3>
              <CICoverageChart
                ci50={accuracy.ci_coverage.ci_50}
                ci80={accuracy.ci_coverage.ci_80}
                ci95={accuracy.ci_coverage.ci_95}
              />
            </div>
          </div>

          {/* Breakdown by Market/Interval */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            {/* By Market */}
            <div className="card">
              <h3 className="text-lg font-medium text-gray-900 mb-4">By Market</h3>
              <div className="space-y-4">
                {Object.entries(accuracy.by_market).map(([market, data]) => (
                  <div key={market} className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-900">{market.toUpperCase()}</p>
                      <p className="text-sm text-gray-500">{data.predictions} predictions</p>
                    </div>
                    <div className="text-right">
                      <p
                        className={`text-xl font-bold ${
                          data.accuracy >= 0.55 ? 'text-green-600' : 'text-red-600'
                        }`}
                      >
                        {(data.accuracy * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* By Interval */}
            <div className="card">
              <h3 className="text-lg font-medium text-gray-900 mb-4">By Interval</h3>
              <div className="space-y-4">
                {Object.entries(accuracy.by_interval).map(([interval, data]) => (
                  <div key={interval} className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-gray-900">{interval}</p>
                      <p className="text-sm text-gray-500">{data.predictions} predictions</p>
                    </div>
                    <div className="text-right">
                      <p
                        className={`text-xl font-bold ${
                          data.accuracy >= 0.55 ? 'text-green-600' : 'text-red-600'
                        }`}
                      >
                        {(data.accuracy * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Error Metrics */}
          <div className="card mb-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Error Metrics</h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-gray-900">
                  {accuracy.error_metrics.mae.toFixed(2)}
                </p>
                <p className="text-sm text-gray-500">MAE</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-gray-900">
                  {accuracy.error_metrics.mape.toFixed(2)}%
                </p>
                <p className="text-sm text-gray-500">MAPE</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-gray-900">
                  {accuracy.error_metrics.rmse.toFixed(2)}
                </p>
                <p className="text-sm text-gray-500">RMSE</p>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <p className="text-2xl font-bold text-gray-900">
                  {(accuracy.direction_accuracy.bullish_accuracy * 100).toFixed(1)}%
                </p>
                <p className="text-sm text-gray-500">Bullish Accuracy</p>
              </div>
            </div>
          </div>

          {/* Prediction History */}
          <div className="card">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Predictions</h3>
            <PredictionHistoryTable predictions={predictions} />
          </div>
        </>
      ) : (
        <div className="card text-center py-12">
          <p className="text-gray-500">No accuracy data available</p>
        </div>
      )}
    </div>
  );
}
