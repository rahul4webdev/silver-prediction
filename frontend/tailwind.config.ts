import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: 'var(--background)',
        foreground: 'var(--foreground)',
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
        bullish: {
          DEFAULT: '#22c55e',
          light: '#4ade80',
          dark: '#16a34a',
        },
        bearish: {
          DEFAULT: '#ef4444',
          light: '#f87171',
          dark: '#dc2626',
        },
        neutral: {
          DEFAULT: '#6b7280',
          light: '#9ca3af',
          dark: '#4b5563',
        },
      },
    },
  },
  plugins: [],
}

export default config
