import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Silver Price Prediction',
  description: 'Real-time silver price prediction system with ML ensemble models',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gray-50">
          <nav className="bg-white shadow-sm border-b border-gray-200">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex justify-between h-16">
                <div className="flex items-center">
                  <a href="/" className="flex items-center">
                    <span className="text-xl font-bold text-gray-900">
                      Silver Prediction
                    </span>
                  </a>
                  <div className="hidden md:flex ml-10 space-x-8">
                    <a
                      href="/dashboard"
                      className="text-gray-600 hover:text-gray-900 px-3 py-2 text-sm font-medium"
                    >
                      Dashboard
                    </a>
                    <a
                      href="/accuracy"
                      className="text-gray-600 hover:text-gray-900 px-3 py-2 text-sm font-medium"
                    >
                      Accuracy
                    </a>
                    <a
                      href="/predictions"
                      className="text-gray-600 hover:text-gray-900 px-3 py-2 text-sm font-medium"
                    >
                      History
                    </a>
                  </div>
                </div>
                <div className="flex items-center">
                  <span className="text-sm text-gray-500">
                    MCX & COMEX Silver
                  </span>
                </div>
              </div>
            </div>
          </nav>
          <main>{children}</main>
        </div>
      </body>
    </html>
  )
}
