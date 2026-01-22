'use client';

import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { getPriceData } from '@/lib/api';
import type { PriceCandle, Market } from '@/lib/types';

interface PriceChartProps {
  market: Market;
  interval?: string;
}

interface ChartData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface VolumeData {
  time: number;
  value: number;
  color: string;
}

export default function PriceChart({ market, interval = '1h' }: PriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<any>(null);
  const candleSeriesRef = useRef<any>(null);
  const volumeSeriesRef = useRef<any>(null);
  const mountedRef = useRef(true);
  const initializingRef = useRef(false);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartData, setChartData] = useState<ChartData[]>([]);
  const [volumeData, setVolumeData] = useState<VolumeData[]>([]);
  const [retryCount, setRetryCount] = useState(0);

  // Memoized chart info
  const chartInfo = useMemo(() => {
    if (chartData.length === 0) {
      return { lastPrice: null, change: null, candleCount: 0 };
    }
    const lastCandle = chartData[chartData.length - 1];
    const firstCandle = chartData[0];
    const priceChange = ((lastCandle.close - firstCandle.open) / firstCandle.open) * 100;
    return {
      lastPrice: lastCandle.close,
      change: priceChange,
      candleCount: chartData.length,
    };
  }, [chartData]);

  // Fetch data separately from chart initialization
  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const candles = await getPriceData('silver', market, interval, 200);

      if (!mountedRef.current) return;

      if (!candles || candles.length === 0) {
        setError(`No data available for ${market.toUpperCase()} ${interval}`);
        setChartData([]);
        setVolumeData([]);
        return;
      }

      // Process and validate data
      const processedData: ChartData[] = [];
      const processedVolume: VolumeData[] = [];
      const seenTimes = new Set<number>();

      for (const candle of candles) {
        if (!candle.timestamp || candle.open == null || candle.high == null ||
            candle.low == null || candle.close == null) {
          continue;
        }

        const time = Math.floor(new Date(candle.timestamp).getTime() / 1000);
        if (isNaN(time) || seenTimes.has(time)) continue;

        seenTimes.add(time);
        processedData.push({
          time,
          open: Number(candle.open),
          high: Number(candle.high),
          low: Number(candle.low),
          close: Number(candle.close),
        });

        if (candle.volume && candle.volume > 0) {
          processedVolume.push({
            time,
            value: Number(candle.volume),
            color: candle.close >= candle.open ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)',
          });
        }
      }

      // Sort by time
      processedData.sort((a, b) => a.time - b.time);
      processedVolume.sort((a, b) => a.time - b.time);

      setChartData(processedData);
      setVolumeData(processedVolume);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch chart data:', err);
      if (mountedRef.current) {
        setError(`Failed to load data: ${err instanceof Error ? err.message : 'Network error'}`);
      }
    } finally {
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  }, [market, interval]);

  // Initialize chart when data is ready
  const initializeChart = useCallback(async () => {
    if (!containerRef.current || chartData.length === 0 || initializingRef.current) {
      return;
    }

    initializingRef.current = true;

    try {
      // Clean up existing chart
      if (chartInstanceRef.current) {
        try {
          chartInstanceRef.current.remove();
        } catch {
          // Ignore cleanup errors
        }
        chartInstanceRef.current = null;
        candleSeriesRef.current = null;
        volumeSeriesRef.current = null;
      }

      // Dynamic import
      const { createChart, CrosshairMode } = await import('lightweight-charts');

      if (!mountedRef.current || !containerRef.current) {
        initializingRef.current = false;
        return;
      }

      const container = containerRef.current;
      const chart = createChart(container, {
        layout: {
          background: { color: 'transparent' },
          textColor: '#a1a1aa',
        },
        grid: {
          vertLines: { color: 'rgba(255, 255, 255, 0.04)' },
          horzLines: { color: 'rgba(255, 255, 255, 0.04)' },
        },
        width: container.clientWidth,
        height: 400,
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
          scaleMargins: { top: 0.1, bottom: 0.2 },
        },
        timeScale: {
          borderColor: 'rgba(255, 255, 255, 0.1)',
          timeVisible: true,
          secondsVisible: false,
          rightOffset: 5,
          barSpacing: 8,
          minBarSpacing: 2,
        },
        handleScroll: {
          mouseWheel: true,
          pressedMouseMove: true,
          horzTouchDrag: true,
          vertTouchDrag: true,
        },
        handleScale: {
          axisPressedMouseMove: { time: true, price: true },
          axisDoubleClickReset: { time: true, price: true },
          mouseWheel: true,
          pinch: true,
        },
      });

      chartInstanceRef.current = chart;

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

      candleSeriesRef.current = candleSeries;
      candleSeries.setData(chartData as any);

      // Add volume series if available
      if (volumeData.length > 0) {
        const volumeSeries = chart.addHistogramSeries({
          priceFormat: { type: 'volume' },
          priceScaleId: '',
        });

        chart.priceScale('').applyOptions({
          scaleMargins: { top: 0.85, bottom: 0 },
        });

        volumeSeriesRef.current = volumeSeries;
        volumeSeries.setData(volumeData as any);
      }

      // Fit content
      chart.timeScale().fitContent();

    } catch (err) {
      console.error('Chart initialization error:', err);
      if (mountedRef.current) {
        setError(`Chart initialization failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
      }
    } finally {
      initializingRef.current = false;
    }
  }, [chartData, volumeData, market]);

  // Fetch data when market/interval changes
  useEffect(() => {
    mountedRef.current = true;
    fetchData();

    return () => {
      mountedRef.current = false;
    };
  }, [fetchData, retryCount]);

  // Initialize chart when data is ready
  useEffect(() => {
    if (chartData.length > 0 && !loading) {
      initializeChart();
    }
  }, [chartData, loading, initializeChart]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (chartInstanceRef.current && containerRef.current) {
        chartInstanceRef.current.applyOptions({
          width: containerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      if (chartInstanceRef.current) {
        try {
          chartInstanceRef.current.remove();
        } catch {
          // Ignore cleanup errors
        }
        chartInstanceRef.current = null;
      }
    };
  }, []);

  // Control handlers
  const handleZoomIn = useCallback(() => {
    if (chartInstanceRef.current) {
      const timeScale = chartInstanceRef.current.timeScale();
      const range = timeScale.getVisibleLogicalRange();
      if (range) {
        const size = range.to - range.from;
        const center = (range.to + range.from) / 2;
        const newSize = size * 0.7;
        timeScale.setVisibleLogicalRange({
          from: center - newSize / 2,
          to: center + newSize / 2,
        });
      }
    }
  }, []);

  const handleZoomOut = useCallback(() => {
    if (chartInstanceRef.current) {
      const timeScale = chartInstanceRef.current.timeScale();
      const range = timeScale.getVisibleLogicalRange();
      if (range) {
        const size = range.to - range.from;
        const center = (range.to + range.from) / 2;
        const newSize = size * 1.4;
        timeScale.setVisibleLogicalRange({
          from: center - newSize / 2,
          to: center + newSize / 2,
        });
      }
    }
  }, []);

  const handleReset = useCallback(() => {
    if (chartInstanceRef.current) {
      chartInstanceRef.current.timeScale().fitContent();
    }
  }, []);

  const handleRetry = useCallback(() => {
    setRetryCount(c => c + 1);
  }, []);

  const formatPrice = (price: number | null) => {
    if (price === null) return '-';
    if (market === 'comex') return `$${price.toFixed(2)}`;
    return `₹${price.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
  };

  return (
    <div className="relative">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
        <div>
          <div className="flex items-center gap-3">
            <h3 className="text-white font-semibold">Silver {market.toUpperCase()}</h3>
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
            {chartInfo.candleCount > 0 ? `${chartInfo.candleCount} candles` : 'Loading...'}
          </p>
        </div>

        {/* Controls */}
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
            onClick={handleReset}
            className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white transition-colors"
            title="Fit All"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
            </svg>
          </button>
          <div className="h-4 w-px bg-zinc-700 mx-1"></div>
          <div className="flex items-center gap-2 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-green-500 rounded"></div>
              <span className="text-zinc-400">Up</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-red-500 rounded"></div>
              <span className="text-zinc-400">Down</span>
            </div>
          </div>
        </div>
      </div>

      {/* Chart Container */}
      <div className="relative" style={{ minHeight: '400px' }}>
        {/* Loading overlay */}
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-900/50 z-10 rounded-lg">
            <div className="flex flex-col items-center gap-3">
              <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
              <div className="text-zinc-400 text-sm">Loading chart...</div>
            </div>
          </div>
        )}

        {/* Error state */}
        {error && !loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-900/80 z-10 rounded-lg">
            <div className="text-center p-4">
              <div className="text-red-400 mb-3">{error}</div>
              <button
                onClick={handleRetry}
                className="px-4 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg hover:bg-cyan-500/30 transition-colors"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        {/* Chart */}
        <div
          ref={containerRef}
          className="w-full rounded-lg overflow-hidden"
          style={{ height: '400px' }}
        />
      </div>

      {/* Tips */}
      {!loading && !error && chartInfo.candleCount > 0 && (
        <div className="mt-3 pt-3 border-t border-white/5 text-xs text-zinc-500 flex flex-wrap gap-4">
          <span>Scroll: Zoom</span>
          <span>Drag: Pan</span>
          <span>Double-click: Reset</span>
        </div>
      )}
    </div>
  );
}
