import styles from './NotationStrip.module.css'

const ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']

const ACCIDENTAL_SYMBOL = {
  natural: '',
  'quarter-sharp': '\u{1D132}', // musical symbol quarter tone sharp
  'quarter-flat': '\u{1D133}',  // musical symbol quarter tone flat
  sharp: '♯',
  flat: '♭',
}

const ACCIDENTAL_CLASS = {
  natural: styles.natural,
  'quarter-sharp': styles.quarterTone,
  'quarter-flat': styles.quarterTone,
  sharp: styles.fullAccidental,
  flat: styles.fullAccidental,
}

export default function NotationStrip({ notes }) {
  if (!notes?.length) return null

  const maxDegree = 7
  const totalTime = notes[notes.length - 1].end_s || 1

  return (
    <div className={styles.wrap}>
      <svg viewBox={`0 0 1000 180`} preserveAspectRatio="none" className={styles.svg}>
        {/* degree guide lines */}
        {ROMAN.map((_, i) => {
          const y = 170 - (i / (maxDegree - 1)) * 150
          return (
            <line key={i} x1={0} x2={1000} y1={y} y2={y} className={styles.guideLine} />
          )
        })}

        {notes.map((n, i) => {
          const x = (n.start_s / totalTime) * 1000
          const w = Math.max(((n.end_s - n.start_s) / totalTime) * 1000, 4)
          const y = 170 - ((n.degree - 1) / (maxDegree - 1)) * 150
          const symbol = ACCIDENTAL_SYMBOL[n.accidental] ?? ''
          const cls = ACCIDENTAL_CLASS[n.accidental] ?? styles.natural

          return (
            <g key={i} className={styles.noteGroup}>
              <rect
                x={x} y={y - 9} width={w} height={18} rx={4}
                className={`${styles.noteRect} ${cls}`}
              />
              <text x={x + w / 2} y={y + 4} textAnchor="middle" className={styles.noteText}>
                {ROMAN[n.degree - 1]}{symbol}
              </text>
              <title>
                {`Degree ${ROMAN[n.degree - 1]} · ${n.cents_from_tonic}¢ from tonic · deviation ${n.deviation_cents > 0 ? '+' : ''}${n.deviation_cents}¢ (${n.accidental})`}
              </title>
            </g>
          )
        })}
      </svg>

      <div className={styles.legend}>
        <LegendItem cls={styles.natural} label="Natural (≤25¢)" />
        <LegendItem cls={styles.quarterTone} label="Quarter-tone (25–90¢)" />
        <LegendItem cls={styles.fullAccidental} label="Semitone+ (>90¢)" />
      </div>
    </div>
  )
}

function LegendItem({ cls, label }) {
  return (
    <span className={styles.legendItem}>
      <span className={`${styles.legendSwatch} ${cls}`} />
      {label}
    </span>
  )
}
