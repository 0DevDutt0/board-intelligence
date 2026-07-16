// frontend/src/components/Logo.jsx

export default function Logo() {
  return (
    <a
      href="/"
      aria-label="Resnova Home"
      className="font-wordmark font-bold text-ink-primary leading-none tracking-tight
                 cursor-pointer opacity-100 hover:opacity-90
                 transition-all duration-[250ms] ease-[ease]
                 py-2 select-none outline-none focus-visible:ring-2
                 focus-visible:ring-accent focus-visible:ring-offset-2
                 focus-visible:ring-offset-surface-panel rounded-sm"
      style={{
        fontSize: 'clamp(24px, 2.5vw, 32px)',
        letterSpacing: '-0.03em',
        lineHeight: 1,
        textDecoration: 'none',
      }}
    >
      Resnova
    </a>
  )
}
