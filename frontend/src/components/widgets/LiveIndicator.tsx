'use client';

import { usePriceUpdates } from '@/hooks/useWebSocket';

interface LiveIndicatorProps {
  asset?: string;
  showPrice?: boolean;
}

export default function LiveIndicator({ asset = 'silver', showPrice = false }: LiveIndicatorProps) {
  const { price, isConnected } = usePriceUpdates(asset);

  return (
    <div className="flex items-center gap-2">
      {/* Connection status indicator */}
      <div className="flex items-center gap-1.5">
        <div className={`relative w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-zinc-500'}`}>
          {isConnected && (
            <span className="absolute inset-0 rounded-full bg-green-400 animate-ping opacity-75" />
          )}
        </div>
        <span className="text-xs text-zinc-500">
          {isConnected ? 'Live' : 'Offline'}
        </span>
      </div>

      {/* Optional real-time price display */}
      {showPrice && price && (
        <div className="flex items-center gap-1 text-xs">
          <span className="text-white font-mono">
            {price.price.toLocaleString('en-IN', {
              style: 'currency',
              currency: 'INR',
              minimumFractionDigits: 2
            })}
          </span>
          <span className={price.changePercent >= 0 ? 'text-green-400' : 'text-red-400'}>
            ({price.changePercent >= 0 ? '+' : ''}{price.changePercent.toFixed(2)}%)
          </span>
        </div>
      )}
    </div>
  );
}
