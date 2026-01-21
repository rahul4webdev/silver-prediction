'use client';

interface MarketSelectorProps {
  value: 'mcx' | 'comex';
  onChange: (value: 'mcx' | 'comex') => void;
}

export default function MarketSelector({ value, onChange }: MarketSelectorProps) {
  return (
    <div className="flex rounded-lg border border-gray-200 overflow-hidden">
      <button
        onClick={() => onChange('mcx')}
        className={`px-4 py-2 text-sm font-medium transition-colors ${
          value === 'mcx'
            ? 'bg-primary-600 text-white'
            : 'bg-white text-gray-700 hover:bg-gray-50'
        }`}
      >
        MCX (India)
      </button>
      <button
        onClick={() => onChange('comex')}
        className={`px-4 py-2 text-sm font-medium transition-colors border-l border-gray-200 ${
          value === 'comex'
            ? 'bg-primary-600 text-white'
            : 'bg-white text-gray-700 hover:bg-gray-50'
        }`}
      >
        COMEX (US)
      </button>
    </div>
  );
}
