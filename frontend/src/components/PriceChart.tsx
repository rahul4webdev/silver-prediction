'use client';

import { useEffect, useRef, useState } from 'react';
import { getPriceData } from '@/lib/api';
import type { PriceCandle, Market } from '@/lib/types';

interface PriceChartProps {
  market: Market;
  interval?: string;
}

export default function PriceChart({ market, interval = '1d' }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let chart: any = null;

    async function initChart() {
      if (!containerRef.current) return;

      try {
        setLoading(true);

        // Dynamically import lightweight-charts
        const { createChart } = await import('lightweight-charts');

        // Fetch price data
        const candles = await getPriceData('silver', market, interval, 100);

        if (candles.length === 0) {
          setError('No chart data available');
          setLoading(false);
          return;
        }

        // Clear previous chart
        if (chartRef.current) {
          chartRef.current.remove();
        }

        // Create chart
        chart = createChart(containerRef.current, {
          layout: {
            background: { color: 'transparent' },
            textColor: '#a1a1aa',
          },
          grid: {
            vertLines: { color: 'rgba(255, 255, 255, 0.03)' },
            horzLines: { color: 'rgba(255, 255, 255, 0.03)' },
          },
          width: containerRef.current.clientWidth,
          height: 400,
          crosshair: {
            mode: 1,
            vertLine: {
              color: 'rgba(6, 182, 212, 0.3)',
              labelBackgroundColor: '#16213e',
            },
            horzLine: {
              color: 'rgba(6, 182, 212, 0.3)',
              labelBackgroundColor: '#16213e',
            },
          },
          rightPriceScale: {
            borderColor: 'rgba(255, 255, 255, 0.1)',
          },
          timeScale: {
            borderColor: 'rgba(255, 255, 255, 0.1)',
            timeVisible: true,
          },
        });

        chartRef.current = chart;

        // Add candlestick series
        const candleSeries = chart.addCandlestickSeries({
          upColor: '#22c55e',
          downColor: '#ef4444',
          borderDownColor: '#ef4444',
          borderUpColor: '#22c55e',
          wickDownColor: '#ef4444',
          wickUpColor: '#22c55e',
        });

        // Format data for chart
        const chartData = candles.map((candle: PriceCandle) => ({
          time: Math.floor(new Date(candle.timestamp).getTime() / 1000) as any,
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
        }));

        candleSeries.setData(chartData);
        chart.timeScale().fitContent();

        setError(null);
      } catch (err) {
        console.error('Chart error:', err);
        setError('Failed to load chart');
      } finally {
        setLoading(false);
      }
    }

    initChart();

    // Handle resize
    const handleResize = () => {
      if (chartRef.current && containerRef.current) {
        chartRef.current.applyOptions({ width: containerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [market, interval]);

  return (
    <div className="glass-card p-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-white font-semibold">Silver {market.toUpperCase()}</h3>
          <p className="text-zinc-500 text-xs">Candlestick Chart</p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-green-500 rounded"></div>
            <span className="text-zinc-400">Bullish</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-red-500 rounded"></div>
            <span className="text-zinc-400">Bearish</span>
          </div>
        </div>
      </div>

      {loading && (
        <div className="h-[400px] flex items-center justify-center">
          <div className="text-zinc-500">Loading chart...</div>
        </div>
      )}

      {error && !loading && (
        <div className="h-[400px] flex items-center justify-center">
          <div className="text-zinc-500">{error}</div>
        </div>
      )}

      <div ref={containerRef} className={loading || error ? 'hidden' : ''} />
    </div>
  );
}
