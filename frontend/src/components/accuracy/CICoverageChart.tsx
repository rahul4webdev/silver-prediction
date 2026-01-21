'use client';

interface CICoverageChartProps {
  ci50: number;
  ci80: number;
  ci95: number;
}

export default function CICoverageChart({ ci50, ci80, ci95 }: CICoverageChartProps) {
  const targets = { ci50: 0.5, ci80: 0.8, ci95: 0.95 };

  const bars = [
    { label: '50% CI', actual: ci50, target: targets.ci50, color: 'bg-purple-400' },
    { label: '80% CI', actual: ci80, target: targets.ci80, color: 'bg-purple-500' },
    { label: '95% CI', actual: ci95, target: targets.ci95, color: 'bg-purple-600' },
  ];

  return (
    <div className="space-y-6">
      {bars.map((bar) => {
        const actualPercent = bar.actual * 100;
        const targetPercent = bar.target * 100;
        const isCalibrated = Math.abs(bar.actual - bar.target) < 0.1;

        return (
          <div key={bar.label}>
            <div className="flex justify-between text-sm mb-2">
              <span className="font-medium text-gray-700">{bar.label}</span>
              <span
                className={`font-semibold ${
                  isCalibrated ? 'text-green-600' : 'text-yellow-600'
                }`}
              >
                {actualPercent.toFixed(1)}%
                <span className="text-gray-400 font-normal ml-1">
                  (target: {targetPercent}%)
                </span>
              </span>
            </div>
            <div className="relative">
              {/* Background */}
              <div className="w-full bg-gray-100 rounded-full h-4">
                {/* Actual coverage */}
                <div
                  className={`${bar.color} h-4 rounded-full transition-all duration-500`}
                  style={{ width: `${Math.min(actualPercent, 100)}%` }}
                ></div>
              </div>
              {/* Target marker */}
              <div
                className="absolute top-0 h-4 w-0.5 bg-green-600"
                style={{ left: `${targetPercent}%` }}
              >
                <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-green-600 rounded-full"></div>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              {isCalibrated
                ? 'Well calibrated'
                : bar.actual < bar.target
                ? 'Under-confident (intervals too narrow)'
                : 'Over-confident (intervals too wide)'}
            </p>
          </div>
        );
      })}

      {/* Explanation */}
      <div className="mt-4 p-3 bg-gray-50 rounded-lg">
        <p className="text-xs text-gray-600">
          <strong>What this means:</strong> If the 80% CI coverage is 78%, it means 78% of
          actual prices fell within our 80% confidence interval. Ideally, this should be
          close to 80%. If coverage is lower than target, our intervals are too narrow
          (over-confident). If higher, they're too wide (under-confident).
        </p>
      </div>
    </div>
  );
}
