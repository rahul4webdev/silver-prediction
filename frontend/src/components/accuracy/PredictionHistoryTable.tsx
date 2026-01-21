'use client';

import type { Prediction } from '@/lib/types';

interface PredictionHistoryTableProps {
  predictions: Prediction[];
}

export default function PredictionHistoryTable({ predictions }: PredictionHistoryTableProps) {
  if (predictions.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No predictions to display
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Market</th>
            <th>Interval</th>
            <th>Direction</th>
            <th>Predicted</th>
            <th>Actual</th>
            <th>Error</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((pred) => {
            const error = pred.actual_price
              ? ((pred.predicted_price - pred.actual_price) / pred.actual_price) * 100
              : null;

            return (
              <tr key={pred.id}>
                <td>
                  <div className="text-sm">
                    {new Date(pred.target_time).toLocaleDateString(undefined, {
                      month: 'short',
                      day: 'numeric',
                    })}
                  </div>
                  <div className="text-xs text-gray-500">
                    {new Date(pred.target_time).toLocaleTimeString(undefined, {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </div>
                </td>
                <td>
                  <span className="uppercase text-xs font-medium bg-gray-100 px-2 py-1 rounded">
                    {pred.market}
                  </span>
                </td>
                <td className="text-sm">{pred.interval}</td>
                <td>
                  <span className={`badge-${pred.predicted_direction} text-xs`}>
                    {pred.predicted_direction.toUpperCase()}
                  </span>
                </td>
                <td className="font-mono text-sm">
                  {pred.predicted_price.toFixed(2)}
                </td>
                <td className="font-mono text-sm">
                  {pred.actual_price ? pred.actual_price.toFixed(2) : '-'}
                </td>
                <td>
                  {error !== null ? (
                    <span
                      className={`text-sm font-medium ${
                        Math.abs(error) < 1 ? 'text-green-600' : 'text-yellow-600'
                      }`}
                    >
                      {error >= 0 ? '+' : ''}
                      {error.toFixed(2)}%
                    </span>
                  ) : (
                    <span className="text-gray-400">-</span>
                  )}
                </td>
                <td>
                  {pred.verified_at ? (
                    <div className="flex items-center gap-2">
                      <span
                        className={`w-2 h-2 rounded-full ${
                          pred.is_direction_correct ? 'bg-green-500' : 'bg-red-500'
                        }`}
                      ></span>
                      <span
                        className={`text-sm font-medium ${
                          pred.is_direction_correct ? 'text-green-600' : 'text-red-600'
                        }`}
                      >
                        {pred.is_direction_correct ? 'Correct' : 'Wrong'}
                      </span>
                      {pred.within_ci_80 && (
                        <span className="text-xs text-gray-400" title="Within 80% CI">
                          (80%)
                        </span>
                      )}
                    </div>
                  ) : (
                    <span className="text-sm text-gray-400">Pending</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
