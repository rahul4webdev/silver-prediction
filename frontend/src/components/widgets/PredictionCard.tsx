'use client';

import type { Prediction } from '@/lib/types';

interface PredictionCardProps {
  prediction: Prediction;
  market: 'mcx' | 'comex';
}

export default function PredictionCard({ prediction, market }: PredictionCardProps) {
  const currency = market === 'mcx' ? '₹' : '$';
  const priceChange = prediction.predicted_price - prediction.current_price;
  const priceChangePercent = (priceChange / prediction.current_price) * 100;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900">Latest Prediction</h3>
        <span className={`badge-${prediction.predicted_direction}`}>
          {prediction.predicted_direction.toUpperCase()}
        </span>
      </div>

      <div className="space-y-4">
        {/* Current vs Predicted */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wide">Current</p>
            <p className="text-xl font-bold text-gray-900">
              {currency}
              {prediction.current_price.toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500 uppercase tracking-wide">Predicted</p>
            <p className="text-xl font-bold text-primary-600">
              {currency}
              {prediction.predicted_price.toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}
            </p>
          </div>
        </div>

        {/* Change */}
        <div className="flex items-center justify-center py-2 px-3 rounded-lg bg-gray-50">
          <span
            className={`text-lg font-semibold ${
              priceChange >= 0 ? 'text-green-600' : 'text-red-600'
            }`}
          >
            {priceChange >= 0 ? '+' : ''}
            {currency}
            {priceChange.toLocaleString(undefined, {
              minimumFractionDigits: 2,
              maximumFractionDigits: 2,
            })}
            <span className="text-sm ml-1">
              ({priceChangePercent >= 0 ? '+' : ''}
              {priceChangePercent.toFixed(2)}%)
            </span>
          </span>
        </div>

        {/* Confidence */}
        <div>
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-500">Direction Confidence</span>
            <span className="font-medium">
              {(prediction.direction_confidence * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className={`h-2 rounded-full ${
                prediction.predicted_direction === 'bullish'
                  ? 'bg-green-500'
                  : prediction.predicted_direction === 'bearish'
                  ? 'bg-red-500'
                  : 'bg-yellow-500'
              }`}
              style={{ width: `${prediction.direction_confidence * 100}%` }}
            ></div>
          </div>
        </div>

        {/* Confidence Intervals */}
        <div className="border-t border-gray-100 pt-4">
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">
            Confidence Intervals
          </p>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">50% CI</span>
              <span className="font-mono">
                {currency}
                {prediction.ci_50_lower.toFixed(2)} - {currency}
                {prediction.ci_50_upper.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">80% CI</span>
              <span className="font-mono">
                {currency}
                {prediction.ci_80_lower.toFixed(2)} - {currency}
                {prediction.ci_80_upper.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">95% CI</span>
              <span className="font-mono">
                {currency}
                {prediction.ci_95_lower.toFixed(2)} - {currency}
                {prediction.ci_95_upper.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        {/* Timing */}
        <div className="border-t border-gray-100 pt-4 text-sm">
          <div className="flex justify-between mb-1">
            <span className="text-gray-500">Target Time</span>
            <span className="font-medium">
              {new Date(prediction.target_time).toLocaleString(undefined, {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
              })}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Predicted At</span>
            <span className="text-gray-600">
              {new Date(prediction.prediction_time).toLocaleString(undefined, {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
              })}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
