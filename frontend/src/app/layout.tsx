import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Navigation from '@/components/layout/Navigation';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Silver Prediction | AI-Powered Price Forecasting',
  description: 'Real-time silver price predictions for MCX and COMEX using advanced ML ensemble models.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-gradient-dark min-h-screen`}>
        <div className="relative min-h-screen grid-pattern">
          {/* Background glow effects */}
          <div className="hero-glow hero-glow-1" />
          <div className="hero-glow hero-glow-2" />

          {/* Navigation */}
          <Navigation />

          {/* Main content */}
          <main className="relative z-10 pt-20">
            {children}
          </main>

          {/* Footer */}
          <footer className="relative z-10 border-t border-white/5 mt-20">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
              <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-purple-600 flex items-center justify-center">
                    <span className="text-white font-bold text-sm">Ag</span>
                  </div>
                  <span className="text-sm text-zinc-400">
                    Silver Prediction System
                  </span>
                </div>
                <p className="text-xs text-zinc-500 text-center md:text-right max-w-md">
                  Predictions are probabilistic, not certainties. This is a decision support tool, not financial advice.
                </p>
              </div>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
