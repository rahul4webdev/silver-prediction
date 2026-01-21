'use client';

interface IntervalSelectorProps {
  value: '30m' | '1h' | '4h' | 'daily';
  onChange: (value: '30m' | '1h' | '4h' | 'daily') => void;
}

const intervals = [
  { value: '30m', label: '30m' },
  { value: '1h', label: '1H' },
  { value: '4h', label: '4H' },
  { value: 'daily', label: '1D' },
] as const;

export default function IntervalSelector({ value, onChange }: IntervalSelectorProps) {
  return (
    <div className="flex rounded-lg border border-gray-200 overflow-hidden">
      {intervals.map((interval) => (
        <button
          key={interval.value}
          onClick={() => onChange(interval.value)}
          className={`px-3 py-2 text-sm font-medium transition-colors ${
            value === interval.value
              ? 'bg-primary-600 text-white'
              : 'bg-white text-gray-700 hover:bg-gray-50'
          } ${interval.value !== '30m' ? 'border-l border-gray-200' : ''}`}
        >
          {interval.label}
        </button>
      ))}
    </div>
  );
}
