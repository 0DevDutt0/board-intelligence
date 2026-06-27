// frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx}',
  ],
  theme: {
    extend: {
      colors: {
        surface: {
          base:     '#020617',
          panel:    '#0F172A',
          card:     '#0E1223',
          elevated: '#1E293B',
          hover:    '#263249',
        },
        accent: {
          DEFAULT: '#3B82F6',
          hover:   '#2563EB',
          dim:     '#1D4ED8',
        },
        border: {
          subtle:  '#1E293B',
          DEFAULT: '#334155',
          bright:  '#475569',
        },
        ink: {
          primary:   '#F8FAFC',
          secondary: '#94A3B8',
          tertiary:  '#64748B',
          disabled:  '#475569',
        },
        status: {
          online:  '#22C55E',
          warning: '#F59E0B',
          error:   '#EF4444',
        },
        citation: '#818CF8',
      },
      fontFamily: {
        wordmark: [
          'Manrope',
          'Inter',
          'sans-serif',
        ],
        sans: [
          'Inter',
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'Roboto',
          'sans-serif',
        ],
        mono: [
          'JetBrains Mono',
          'Consolas',
          'Monaco',
          'monospace',
        ],
      },
      keyframes: {
        pulse_dot: {
          '0%, 100%': { opacity: '1' },
          '50%':      { opacity: '0.3' },
        },
        fade_in: {
          from: { opacity: '0', transform: 'translateY(6px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        spin_slow: {
          from: { transform: 'rotate(0deg)' },
          to:   { transform: 'rotate(360deg)' },
        },
      },
      animation: {
        pulse_dot:  'pulse_dot 1.4s ease-in-out infinite',
        fade_in:    'fade_in 0.2s ease-out',
        shimmer:    'shimmer 1.6s linear infinite',
        spin_slow:  'spin_slow 1s linear infinite',
      },
      boxShadow: {
        glow_accent: '0 0 20px rgba(59,130,246,0.20)',
        glow_sm:     '0 0 8px rgba(59,130,246,0.15)',
        card:        '0 1px 3px rgba(0,0,0,0.5)',
        card_lg:     '0 4px 20px rgba(0,0,0,0.6)',
      },
    },
  },
  plugins: [],
}
