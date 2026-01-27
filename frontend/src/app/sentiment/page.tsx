'use client';

import { useEffect, useState, useCallback } from 'react';
import { getSentiment, type SentimentData } from '@/lib/api';
import { cn } from '@/lib/utils';

export default function SentimentPage() {
  const [sentiment, setSentiment] = useState<SentimentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedAsset, setSelectedAsset] = useState<'silver' | 'gold'>('silver');
  const [lookbackDays, setLookbackDays] = useState(3);

  const fetchSentiment = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getSentiment(selectedAsset, lookbackDays);
      if (data) {
        setSentiment(data);
      } else {
        setError('Unable to fetch sentiment data');
      }
    } catch (err) {
      setError('Failed to fetch sentiment');
    } finally {
      setLoading(false);
    }
  }, [selectedAsset, lookbackDays]);

  useEffect(() => {
    fetchSentiment();
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

  const getSentimentBgGradient = (label: string) => {
    switch (label) {
      case 'bullish':
        return 'from-green-500/20 to-transparent';
      case 'bearish':
        return 'from-red-500/20 to-transparent';
      default:
        return 'from-zinc-500/20 to-transparent';
    }
  };

  const getArticleSentimentColor = (score: number | null) => {
    if (score === null) return 'bg-zinc-500';
    if (score > 0.2) return 'bg-green-500';
    if (score < -0.2) return 'bg-red-500';
    return 'bg-zinc-500';
  };

  const getArticleSentimentLabel = (score: number | null) => {
    if (score === null) return 'Unknown';
    if (score > 0.2) return 'Bullish';
    if (score < -0.2) return 'Bearish';
    return 'Neutral';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">News Sentiment Analysis</h1>
          <p className="text-sm text-zinc-400 mt-1">
            AI-powered sentiment analysis of financial news affecting {selectedAsset} prices
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-4">
          {/* Asset Toggle */}
          <div className="glass-card p-1 flex">
            <button
              onClick={() => setSelectedAsset('silver')}
              className={cn(
                "px-4 py-2 rounded-lg text-sm font-medium transition-all",
                selectedAsset === 'silver'
                  ? "bg-cyan-500/20 text-cyan-400"
                  : "text-zinc-400 hover:text-white"
              )}
            >
              Silver
            </button>
            <button
              onClick={() => setSelectedAsset('gold')}
              className={cn(
                "px-4 py-2 rounded-lg text-sm font-medium transition-all",
                selectedAsset === 'gold'
                  ? "bg-cyan-500/20 text-cyan-400"
                  : "text-zinc-400 hover:text-white"
              )}
            >
              Gold
            </button>
          </div>

          {/* Lookback Days */}
          <div className="glass-card p-1 flex">
            {[1, 3, 5, 7].map((days) => (
              <button
                key={days}
                onClick={() => setLookbackDays(days)}
                className={cn(
                  "px-3 py-2 rounded-lg text-sm font-medium transition-all",
                  lookbackDays === days
                    ? "bg-cyan-500/20 text-cyan-400"
                    : "text-zinc-400 hover:text-white"
                )}
              >
                {days}d
              </button>
            ))}
          </div>

          {/* Refresh Button */}
          <button
            onClick={fetchSentiment}
            disabled={loading}
            className="glass-card px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors disabled:opacity-50"
          >
            {loading ? (
              <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="glass-card p-4 border border-red-500/30 bg-red-500/10">
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {loading && !sentiment ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="glass-card p-6">
              <div className="skeleton h-6 w-32 rounded mb-4"></div>
              <div className="skeleton h-16 w-full rounded mb-4"></div>
              <div className="skeleton h-4 w-24 rounded"></div>
            </div>
          ))}
        </div>
      ) : sentiment ? (
        <>
          {/* Main Sentiment Overview */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Overall Sentiment Card */}
            <div className={cn(
              "glass-card p-6 relative overflow-hidden",
              sentiment.sentiment.label === 'bullish' ? 'glow-success' :
              sentiment.sentiment.label === 'bearish' ? 'glow-danger' : ''
            )}>
              <div className={cn(
                "absolute inset-0 bg-gradient-to-br opacity-20",
                getSentimentBgGradient(sentiment.sentiment.label)
              )} />
              <div className="relative">
                <h3 className="text-sm font-medium text-zinc-400 mb-4">Overall Sentiment</h3>
                <div className="flex items-center gap-4">
                  <div className={cn(
                    "w-16 h-16 rounded-2xl flex items-center justify-center",
                    sentiment.sentiment.label === 'bullish' ? 'bg-green-500/20' :
                    sentiment.sentiment.label === 'bearish' ? 'bg-red-500/20' : 'bg-zinc-500/20'
                  )}>
                    {sentiment.sentiment.label === 'bullish' ? (
                      <svg className="w-8 h-8 text-bullish" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                      </svg>
                    ) : sentiment.sentiment.label === 'bearish' ? (
                      <svg className="w-8 h-8 text-bearish" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0v-8m0 8l-8-8-4 4-6-6" />
                      </svg>
                    ) : (
                      <svg className="w-8 h-8 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14" />
                      </svg>
                    )}
                  </div>
                  <div>
                    <div className={cn(
                      "text-3xl font-bold capitalize",
                      getSentimentColor(sentiment.sentiment.label)
                    )}>
                      {sentiment.sentiment.label}
                    </div>
                    <div className="text-sm text-zinc-500">
                      Score: {sentiment.sentiment.overall.toFixed(3)}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Confidence Card */}
            <div className="glass-card p-6">
              <h3 className="text-sm font-medium text-zinc-400 mb-4">Confidence Level</h3>
              <div className="flex items-center gap-4">
                <div className="relative w-24 h-24">
                  <svg className="w-24 h-24 transform -rotate-90">
                    <circle
                      cx="48"
                      cy="48"
                      r="40"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="none"
                      className="text-zinc-700"
                    />
                    <circle
                      cx="48"
                      cy="48"
                      r="40"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="none"
                      strokeDasharray={`${sentiment.sentiment.confidence * 251.2} 251.2`}
                      className="text-cyan-400"
                      strokeLinecap="round"
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-xl font-bold text-white">
                      {(sentiment.sentiment.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                <div className="flex-1">
                  <p className="text-sm text-zinc-400">
                    Based on {sentiment.stats.article_count} analyzed articles
                  </p>
                  <p className="text-xs text-zinc-500 mt-1">
                    Higher confidence indicates stronger agreement among news sources
                  </p>
                </div>
              </div>
            </div>

            {/* Article Distribution Card */}
            <div className="glass-card p-6">
              <h3 className="text-sm font-medium text-zinc-400 mb-4">Sentiment Distribution</h3>
              <div className="space-y-3">
                {/* Bullish Bar */}
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-bullish">Bullish</span>
                    <span className="text-zinc-400">{sentiment.stats.bullish_count}</span>
                  </div>
                  <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-green-600 to-green-400 rounded-full transition-all duration-500"
                      style={{
                        width: `${(sentiment.stats.bullish_count / Math.max(sentiment.stats.article_count, 1)) * 100}%`
                      }}
                    ></div>
                  </div>
                </div>

                {/* Neutral Bar */}
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-zinc-400">Neutral</span>
                    <span className="text-zinc-400">{sentiment.stats.neutral_count}</span>
                  </div>
                  <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-zinc-600 to-zinc-400 rounded-full transition-all duration-500"
                      style={{
                        width: `${(sentiment.stats.neutral_count / Math.max(sentiment.stats.article_count, 1)) * 100}%`
                      }}
                    ></div>
                  </div>
                </div>

                {/* Bearish Bar */}
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-bearish">Bearish</span>
                    <span className="text-zinc-400">{sentiment.stats.bearish_count}</span>
                  </div>
                  <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-red-600 to-red-400 rounded-full transition-all duration-500"
                      style={{
                        width: `${(sentiment.stats.bearish_count / Math.max(sentiment.stats.article_count, 1)) * 100}%`
                      }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Sentiment Meter */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-medium text-zinc-400 mb-4">Sentiment Meter</h3>
            <div className="max-w-2xl mx-auto">
              <div className="flex justify-between text-sm text-zinc-500 mb-2">
                <span>Very Bearish</span>
                <span>Bearish</span>
                <span>Neutral</span>
                <span>Bullish</span>
                <span>Very Bullish</span>
              </div>
              <div className="relative h-6 bg-gradient-to-r from-red-600 via-zinc-600 to-green-600 rounded-full overflow-hidden">
                {/* Indicator */}
                <div
                  className="absolute top-0 bottom-0 w-2 bg-white rounded-full shadow-lg transform -translate-x-1/2 transition-all duration-500"
                  style={{
                    left: `${((sentiment.sentiment.overall + 1) / 2) * 100}%`,
                  }}
                >
                  <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 bg-white text-black text-xs font-bold px-2 py-1 rounded whitespace-nowrap">
                    {sentiment.sentiment.overall.toFixed(2)}
                  </div>
                </div>
              </div>
              <div className="flex justify-between text-xs text-zinc-600 mt-1">
                <span>-1.0</span>
                <span>-0.5</span>
                <span>0.0</span>
                <span>+0.5</span>
                <span>+1.0</span>
              </div>
            </div>
          </div>

          {/* How Sentiment Affects Predictions */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-medium text-zinc-400 mb-4">How Sentiment Impacts Predictions</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white/5 rounded-xl p-4">
                <div className="text-cyan-400 text-sm font-medium mb-2">Sentiment Score</div>
                <div className="text-2xl font-bold text-white">{sentiment.sentiment.overall.toFixed(3)}</div>
                <p className="text-xs text-zinc-500 mt-1">
                  Weighted average of article sentiments, used as a feature in ML models
                </p>
              </div>
              <div className="bg-white/5 rounded-xl p-4">
                <div className="text-cyan-400 text-sm font-medium mb-2">Bullish/Bearish Ratio</div>
                <div className="text-2xl font-bold text-white">
                  {sentiment.stats.bullish_count > 0 || sentiment.stats.bearish_count > 0
                    ? (sentiment.stats.bullish_count / Math.max(sentiment.stats.bearish_count, 1)).toFixed(2)
                    : 'N/A'}
                </div>
                <p className="text-xs text-zinc-500 mt-1">
                  Ratio helps identify market sentiment extremes
                </p>
              </div>
              <div className="bg-white/5 rounded-xl p-4">
                <div className="text-cyan-400 text-sm font-medium mb-2">Avg. Relevance</div>
                <div className="text-2xl font-bold text-white">{(sentiment.stats.average_relevance * 100).toFixed(0)}%</div>
                <p className="text-xs text-zinc-500 mt-1">
                  Higher relevance means articles are more specific to {selectedAsset}
                </p>
              </div>
            </div>
          </div>

          {/* News Articles */}
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-zinc-400">Analyzed News Articles</h3>
              <span className="text-xs text-zinc-500">{sentiment.articles.length} articles shown</span>
            </div>
            <div className="space-y-3">
              {sentiment.articles.map((article, index) => (
                <a
                  key={index}
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block group"
                >
                  <div className="bg-white/5 rounded-xl p-4 hover:bg-white/10 transition-colors">
                    <div className="flex items-start gap-4">
                      {/* Sentiment Indicator */}
                      <div className="flex-shrink-0">
                        <div className={cn(
                          "w-3 h-3 rounded-full mt-1",
                          getArticleSentimentColor(article.sentiment_score)
                        )}></div>
                      </div>

                      {/* Article Content */}
                      <div className="flex-1 min-w-0">
                        <h4 className="text-sm font-medium text-white group-hover:text-cyan-400 transition-colors line-clamp-2">
                          {article.title}
                        </h4>
                        <div className="flex items-center gap-3 mt-2 text-xs text-zinc-500">
                          <span>{article.source}</span>
                          <span>•</span>
                          <span>{new Date(article.published_at).toLocaleDateString()}</span>
                          <span>•</span>
                          <span className={cn(
                            article.sentiment_score !== null && article.sentiment_score > 0.2 ? 'text-bullish' :
                            article.sentiment_score !== null && article.sentiment_score < -0.2 ? 'text-bearish' : ''
                          )}>
                            {getArticleSentimentLabel(article.sentiment_score)}
                            {article.sentiment_score !== null && ` (${article.sentiment_score.toFixed(2)})`}
                          </span>
                        </div>
                      </div>

                      {/* Relevance Score */}
                      <div className="flex-shrink-0 text-right">
                        <div className="text-xs text-zinc-500">Relevance</div>
                        <div className="text-sm font-medium text-white">
                          {article.relevance_score !== null
                            ? `${(article.relevance_score * 100).toFixed(0)}%`
                            : 'N/A'}
                        </div>
                      </div>
                    </div>
                  </div>
                </a>
              ))}
            </div>
          </div>

          {/* Model Info */}
          <div className="glass-card p-6">
            <h3 className="text-sm font-medium text-zinc-400 mb-4">About the Sentiment Model</h3>
            <div className="prose prose-invert prose-sm max-w-none">
              <p className="text-zinc-400">
                Our sentiment analysis uses <span className="text-cyan-400 font-medium">FinBERT</span>,
                a financial domain-specific BERT model trained on financial communications.
                When FinBERT is not available, we fall back to a lexicon-based approach using
                curated financial sentiment dictionaries.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div className="bg-white/5 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-white mb-2">Keywords Tracked</h4>
                  <div className="flex flex-wrap gap-2">
                    {['silver price', 'precious metals', 'inflation', 'fed', 'safe haven', 'demand'].map((kw) => (
                      <span key={kw} className="text-xs bg-white/10 px-2 py-1 rounded text-zinc-400">
                        {kw}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="bg-white/5 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-white mb-2">Data Sources</h4>
                  <ul className="text-xs text-zinc-400 space-y-1">
                    <li>• Google News RSS (Primary)</li>
                    <li>• NewsAPI.org (Optional)</li>
                    <li>• Financial news aggregators</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </>
      ) : null}

      {/* Last Updated */}
      {sentiment && (
        <div className="text-center text-xs text-zinc-600">
          Last updated: {new Date(sentiment.timestamp).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })} IST
        </div>
      )}
    </div>
  );
}
