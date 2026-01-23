import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Silver Prediction System',
  description: 'Real-time silver price predictions for MCX and COMEX markets',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen">
          <nav className="glass-card mx-4 mt-4 px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-500 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <div>
                <h1 className="text-lg font-bold text-white">Silver Prediction</h1>
                <p className="text-xs text-zinc-400">MCX & COMEX Analytics</p>
              </div>
            </div>
            <div className="flex items-center gap-1 flex-wrap">
              <a href="/" className="px-3 py-2 text-sm text-zinc-300 hover:text-white hover:bg-white/5 rounded-lg transition-colors">
                Dashboard
              </a>
              <a href="/predictions" className="px-3 py-2 text-sm text-zinc-300 hover:text-white hover:bg-white/5 rounded-lg transition-colors">
                Predictions
              </a>
              <a href="/history" className="px-3 py-2 text-sm text-zinc-300 hover:text-white hover:bg-white/5 rounded-lg transition-colors">
                History
              </a>
              <a href="/accuracy" className="px-3 py-2 text-sm text-zinc-300 hover:text-white hover:bg-white/5 rounded-lg transition-colors">
                Accuracy
              </a>
              <a href="/sentiment" className="px-3 py-2 text-sm text-zinc-300 hover:text-white hover:bg-white/5 rounded-lg transition-colors">
                Sentiment
              </a>
              <a href="/confluence" className="px-3 py-2 text-sm text-zinc-300 hover:text-white hover:bg-white/5 rounded-lg transition-colors">
                Confluence
              </a>
              <a href="/alerts" className="px-3 py-2 text-sm text-zinc-300 hover:text-white hover:bg-white/5 rounded-lg transition-colors">
                Alerts
              </a>
            </div>
          </nav>
          <main className="p-4">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
