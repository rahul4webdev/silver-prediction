'use client';

import { useEffect, useState, useCallback } from 'react';
import { getSentiment, type SentimentData } from '@/lib/api';
import { cn } from '@/lib/utils';

interface SentimentCardProps {
  asset?: string;
}

export default function SentimentCard({ asset = 'silver' }: SentimentCardProps) {
  const [sentiment, setSentiment] = useState<SentimentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSentiment = useCallback(async () => {
    try {
      setLoading(true);
      const data = await getSentiment(asset, 3);
      if (data) {
        setSentiment(data);
        setError(null);
      } else {
        setError('Unable to fetch sentiment');
      }
    } catch (err) {
      setError('Failed to fetch sentiment');
    } finally {
      setLoading(false);
    }
  }, [asset]);

  useEffect(() => {
    fetchSentiment();
    // Refresh every 15 minutes
    const interval = setInterval(fetchSentiment, 15 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchSentiment]);

  const getSentimentColor = (label: string) => {
    switch (label) {
      case 'bullish':
        return 'text-bullish';
      case 'bearish':
        return 'text-bearish';
      default:
        return 'text-zinc-400';
    }
  };

  const getSentimentBgColor = (label: string) => {
    switch (label) {
      case 'bullish':
        return 'bg-green-500/20 border-green-500/30';
      case 'bearish':
        return 'bg-red-500/20 border-red-500/30';
      default:
        return 'bg-zinc-500/20 border-zinc-500/30';
    }
  };

  const getSentimentIcon = (label: string) => {
    switch (label) {
      case 'bullish':
        return (
          <svg className="w-5 h-5 text-bullish" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
          </svg>
        );
      case 'bearish':
        return (
          <svg className="w-5 h-5 text-bearish" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0v-8m0 8l-8-8-4 4-6-6" />
          </svg>
        );
      default:
        return (
          <svg className="w-5 h-5 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14" />
          </svg>
        );
    }
  };

  if (loading) {
    return (
      <div className="glass-card p-6">
        <div className="skeleton h-4 w-32 rounded mb-4"></div>
        <div className="skeleton h-8 w-24 rounded mb-2"></div>
        <div className="skeleton h-4 w-40 rounded"></div>
      </div>
    );
  }

  if (error || !sentiment) {
    return (
      <div className="glass-card p-6">
        <div className="text-zinc-400 text-sm">News Sentiment</div>
        <div className="text-zinc-500 mt-2 text-sm">Unable to load sentiment</div>
      </div>
    );
  }

  const { sentiment: sentimentData, stats, articles } = sentiment;

  return (
    <div className="glass-card p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <span className="text-zinc-400 text-sm font-medium">News Sentiment</span>
        <span className="text-xs text-zinc-500">{stats.article_count} articles</span>
      </div>

      {/* Main sentiment display */}
      <div className="flex items-center gap-4 mb-4">
        <div className={cn(
          "flex items-center justify-center w-12 h-12 rounded-xl border",
          getSentimentBgColor(sentimentData.label)
        )}>
          {getSentimentIcon(sentimentData.label)}
        </div>
        <div>
          <div className={cn("text-2xl font-bold capitalize", getSentimentColor(sentimentData.label))}>
            {sentimentData.label}
          </div>
          <div className="text-xs text-zinc-500">
            Confidence: {(sentimentData.confidence * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Sentiment meter */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-zinc-500 mb-1">
          <span>Bearish</span>
          <span>Neutral</span>
          <span>Bullish</span>
        </div>
        <div className="relative h-2 bg-zinc-800 rounded-full overflow-hidden">
          {/* Background gradient */}
          <div className="absolute inset-0 bg-gradient-to-r from-red-500 via-zinc-600 to-green-500 opacity-30"></div>
          {/* Indicator */}
          <div
            className="absolute top-0 w-3 h-2 bg-white rounded-full shadow-lg transition-all duration-300"
            style={{
              left: `calc(${((sentimentData.overall + 1) / 2) * 100}% - 6px)`,
            }}
          ></div>
        </div>
        <div className="text-center text-xs text-zinc-400 mt-1">
          Score: {sentimentData.overall.toFixed(2)}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        <div className="bg-green-500/10 rounded-lg p-2 text-center">
          <div className="text-sm font-semibold text-bullish">{stats.bullish_count}</div>
          <div className="text-xs text-zinc-500">Bullish</div>
        </div>
        <div className="bg-zinc-500/10 rounded-lg p-2 text-center">
          <div className="text-sm font-semibold text-zinc-400">{stats.neutral_count}</div>
          <div className="text-xs text-zinc-500">Neutral</div>
        </div>
        <div className="bg-red-500/10 rounded-lg p-2 text-center">
          <div className="text-sm font-semibold text-bearish">{stats.bearish_count}</div>
          <div className="text-xs text-zinc-500">Bearish</div>
        </div>
      </div>

      {/* Recent articles */}
      {articles.length > 0 && (
        <div className="border-t border-white/10 pt-4">
          <div className="text-xs text-zinc-500 mb-2">Recent News</div>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {articles.slice(0, 5).map((article, index) => (
              <a
                key={index}
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block group"
              >
                <div className="flex items-start gap-2">
                  <div className={cn(
                    "w-1 h-1 rounded-full mt-1.5 flex-shrink-0",
                    (article.sentiment_score ?? 0) > 0.1
                      ? "bg-bullish"
                      : (article.sentiment_score ?? 0) < -0.1
                      ? "bg-bearish"
                      : "bg-zinc-500"
                  )}></div>
                  <div className="flex-1 min-w-0">
                    <div className="text-xs text-zinc-300 truncate group-hover:text-white transition-colors">
                      {article.title}
                    </div>
                    <div className="text-xs text-zinc-600">
                      {article.source} • {new Date(article.published_at).toLocaleDateString()}
                    </div>
                  </div>
                </div>
              </a>
            ))}
          </div>
        </div>
      )}

      {/* Last updated */}
      <div className="text-xs text-zinc-600 mt-3 text-right">
        Updated: {new Date(sentiment.timestamp).toLocaleTimeString()}
      </div>
    </div>
  );
}
