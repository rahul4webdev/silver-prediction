'use client';

interface AccuracyCardProps {
  title: string;
  value: string;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  target?: string;
}

export default function AccuracyCard({
  title,
  value,
  subtitle,
  trend,
  target,
}: AccuracyCardProps) {
  return (
    <div className="card">
      <div className="flex items-start justify-between">
        <p className="text-sm font-medium text-gray-500">{title}</p>
        {trend && (
          <span
            className={`text-xs px-2 py-0.5 rounded-full ${
              trend === 'up'
                ? 'bg-green-100 text-green-700'
                : trend === 'down'
                ? 'bg-red-100 text-red-700'
                : 'bg-gray-100 text-gray-700'
            }`}
          >
            {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→'}
          </span>
        )}
      </div>
      <p className="text-3xl font-bold text-gray-900 mt-2">{value}</p>
      <div className="flex items-center justify-between mt-2">
        {subtitle && <p className="text-sm text-gray-500">{subtitle}</p>}
        {target && (
          <p className="text-xs text-gray-400">Target: {target}</p>
        )}
      </div>
    </div>
  );
}
