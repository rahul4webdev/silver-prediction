'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { getPriceData } from '@/lib/api';
import type { PriceCandle, Market } from '@/lib/types';

interface PriceChartProps {
  market: Market;
  interval?: string;
}

export default function PriceChart({ market, interval = '1h' }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const seriesRef = useRef<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartInfo, setChartInfo] = useState<{
    lastPrice: number | null;
    change: number | null;
    candleCount: number;
  }>({ lastPrice: null, change: null, candleCount: 0 });

  // Reset zoom to fit all data
  const handleResetZoom = useCallback(() => {
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  }, []);

  // Zoom in
  const handleZoomIn = useCallback(() => {
    if (chartRef.current) {
      const timeScale = chartRef.current.timeScale();
      const currentRange = timeScale.getVisibleLogicalRange();
      if (currentRange) {
        const rangeSize = currentRange.to - currentRange.from;
        const newSize = rangeSize * 0.7; // Zoom in by 30%
        const center = (currentRange.to + currentRange.from) / 2;
        timeScale.setVisibleLogicalRange({
          from: center - newSize / 2,
          to: center + newSize / 2,
        });
      }
    }
  }, []);

  // Zoom out
  const handleZoomOut = useCallback(() => {
    if (chartRef.current) {
      const timeScale = chartRef.current.timeScale();
      const currentRange = timeScale.getVisibleLogicalRange();
      if (currentRange) {
        const rangeSize = currentRange.to - currentRange.from;
        const newSize = rangeSize * 1.4; // Zoom out by 40%
        const center = (currentRange.to + currentRange.from) / 2;
        timeScale.setVisibleLogicalRange({
          from: center - newSize / 2,
          to: center + newSize / 2,
        });
      }
    }
  }, []);

  useEffect(() => {
    let chart: any = null;
    let isSubscribed = true;

    async function initChart() {
      if (!containerRef.current) return;

      try {
        setLoading(true);
        setError(null);

        // Dynamically import lightweight-charts
        const { createChart, CrosshairMode } = await import('lightweight-charts');

        // Fetch price data
        const candles = await getPriceData('silver', market, interval, 200);

        if (!isSubscribed) return;

        if (!candles || candles.length === 0) {
          setError(`No data available for ${market.toUpperCase()} ${interval}`);
          setLoading(false);
          return;
        }

        // Calculate price info
        const lastCandle = candles[candles.length - 1];
        const firstCandle = candles[0];
        const priceChange = lastCandle && firstCandle
          ? ((lastCandle.close - firstCandle.open) / firstCandle.open) * 100
          : 0;

        setChartInfo({
          lastPrice: lastCandle?.close || null,
          change: priceChange,
          candleCount: candles.length,
        });

        // Clear previous chart
        if (chartRef.current) {
          chartRef.current.remove();
          chartRef.current = null;
          seriesRef.current = null;
        }

        // Create chart with full interactivity
        chart = createChart(containerRef.current, {
          layout: {
            background: { color: 'transparent' },
            textColor: '#a1a1aa',
          },
          grid: {
            vertLines: { color: 'rgba(255, 255, 255, 0.04)' },
            horzLines: { color: 'rgba(255, 255, 255, 0.04)' },
          },
          width: containerRef.current.clientWidth,
          height: 450,
          crosshair: {
            mode: CrosshairMode.Normal,
            vertLine: {
              width: 1,
              color: 'rgba(6, 182, 212, 0.4)',
              style: 2,
              labelBackgroundColor: '#0e7490',
            },
            horzLine: {
              width: 1,
              color: 'rgba(6, 182, 212, 0.4)',
              style: 2,
              labelBackgroundColor: '#0e7490',
            },
          },
          rightPriceScale: {
            borderColor: 'rgba(255, 255, 255, 0.1)',
            scaleMargins: {
              top: 0.1,
              bottom: 0.2,
            },
          },
          timeScale: {
            borderColor: 'rgba(255, 255, 255, 0.1)',
            timeVisible: true,
            secondsVisible: false,
            rightOffset: 5,
            barSpacing: 10,
            minBarSpacing: 2,
            fixLeftEdge: false,
            fixRightEdge: false,
          },
          handleScroll: {
            mouseWheel: true,
            pressedMouseMove: true,
            horzTouchDrag: true,
            vertTouchDrag: true,
          },
          handleScale: {
            axisPressedMouseMove: {
              time: true,
              price: true,
            },
            axisDoubleClickReset: {
              time: true,
              price: true,
            },
            mouseWheel: true,
            pinch: true,
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
          priceFormat: {
            type: 'price',
            precision: market === 'comex' ? 2 : 0,
            minMove: market === 'comex' ? 0.01 : 1,
          },
        });

        seriesRef.current = candleSeries;

        // Format data for chart - sort by timestamp and filter invalid entries
        const chartData = candles
          .filter((candle: PriceCandle) =>
            candle.timestamp &&
            candle.open != null &&
            candle.high != null &&
            candle.low != null &&
            candle.close != null &&
            !isNaN(candle.open) &&
            !isNaN(candle.close)
          )
          .map((candle: PriceCandle) => {
            const time = Math.floor(new Date(candle.timestamp).getTime() / 1000);
            if (isNaN(time)) return null;
            return {
              time: time as any,
              open: candle.open,
              high: candle.high,
              low: candle.low,
              close: candle.close,
            };
          })
          .filter((item): item is NonNullable<typeof item> => item !== null)
          .sort((a: any, b: any) => a.time - b.time);

        // Remove duplicates (same timestamp)
        const uniqueData = chartData.filter((item: any, index: number, arr: any[]) =>
          index === 0 || item.time !== arr[index - 1].time
        );

        candleSeries.setData(uniqueData);

        // Add volume series if available
        const volumeData = candles
          .filter((c: PriceCandle) => c.volume && c.volume > 0)
          .map((candle: PriceCandle) => ({
            time: Math.floor(new Date(candle.timestamp).getTime() / 1000) as any,
            value: candle.volume,
            color: candle.close >= candle.open ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)',
          }))
          .sort((a: any, b: any) => a.time - b.time);

        if (volumeData.length > 0) {
          const volumeSeries = chart.addHistogramSeries({
            priceFormat: { type: 'volume' },
            priceScaleId: '',
          });

          chart.priceScale('').applyOptions({
            scaleMargins: {
              top: 0.85,
              bottom: 0,
            },
          });

          // Remove duplicate timestamps for volume
          const uniqueVolumeData = volumeData.filter((item: any, index: number, arr: any[]) =>
            index === 0 || item.time !== arr[index - 1].time
          );

          volumeSeries.setData(uniqueVolumeData);
        }

        // Fit content to show all data
        chart.timeScale().fitContent();

      } catch (err) {
        console.error('Chart error:', err);
        if (isSubscribed) {
          setError(`Failed to load chart: ${err instanceof Error ? err.message : 'Unknown error'}`);
        }
      } finally {
        if (isSubscribed) {
          setLoading(false);
        }
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
      isSubscribed = false;
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
        seriesRef.current = null;
      }
    };
  }, [market, interval]);

  const formatPrice = (price: number | null) => {
    if (price === null) return '-';
    if (market === 'comex') {
      return `$${price.toFixed(2)}`;
    }
    return `₹${price.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
  };

  return (
    <div className="glass-card p-4">
      {/* Header with controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
        <div>
          <div className="flex items-center gap-3">
            <h3 className="text-white font-semibold">
              Silver {market.toUpperCase()}
            </h3>
            {chartInfo.lastPrice && (
              <span className="text-lg font-bold text-white">
                {formatPrice(chartInfo.lastPrice)}
              </span>
            )}
            {chartInfo.change !== null && (
              <span className={`text-sm font-medium ${chartInfo.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {chartInfo.change >= 0 ? '+' : ''}{chartInfo.change.toFixed(2)}%
              </span>
            )}
          </div>
          <p className="text-zinc-500 text-xs mt-1">
            {chartInfo.candleCount} candles | Scroll to zoom, drag to pan
          </p>
        </div>

        {/* Chart Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleZoomIn}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white transition-colors"
            title="Zoom In"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v6m3-3H7" />
            </svg>
          </button>
          <button
            onClick={handleZoomOut}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white transition-colors"
            title="Zoom Out"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
            </svg>
          </button>
          <button
            onClick={handleResetZoom}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white transition-colors"
            title="Reset Zoom (Fit All)"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
            </svg>
          </button>
          <div className="h-4 w-px bg-zinc-700 mx-1"></div>
          <div className="flex items-center gap-2 text-xs">
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
      </div>

      {/* Loading State */}
      {loading && (
        <div className="h-[450px] flex items-center justify-center">
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
            <div className="text-zinc-500">Loading {market.toUpperCase()} {interval} chart...</div>
          </div>
        </div>
      )}

      {/* Error State */}
      {error && !loading && (
        <div className="h-[450px] flex items-center justify-center">
          <div className="text-center">
            <div className="text-zinc-500 mb-2">{error}</div>
            <button
              onClick={() => window.location.reload()}
              className="text-cyan-400 hover:text-cyan-300 text-sm"
            >
              Try refreshing
            </button>
          </div>
        </div>
      )}

      {/* Chart Container */}
      <div
        ref={containerRef}
        className={loading || error ? 'hidden' : ''}
        style={{ minHeight: '450px' }}
      />

      {/* Chart Tips */}
      {!loading && !error && (
        <div className="mt-3 pt-3 border-t border-white/5 text-xs text-zinc-500 flex flex-wrap gap-4">
          <span>Mouse wheel: Zoom</span>
          <span>Drag: Pan</span>
          <span>Double-click axis: Reset</span>
          <span>Pinch: Zoom on touch</span>
        </div>
      )}
    </div>
  );
}
