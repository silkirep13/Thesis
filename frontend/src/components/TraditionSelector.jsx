import styles from './TraditionSelector.module.css'

const TRADITIONS = [
  {
    id:       'byzantine',
    label:    'Byzantine',
    subtitle: 'Octoechos · 8 modes',
  },
  {
    id:       'greek',
    label:    'Greek Folk',
    subtitle: 'Matzore · Minore · Hijaz',
  },
  {
    id:       'cypriot',
    label:    'Cypriot',
    subtitle: 'Pentachord',
  },
  {
    id:       'arabic',
    label:    'Arabic Maqam',
    subtitle: 'Rast · Bayati · Hijaz · Nahawand…',
  },
]

export default function TraditionSelector({ value, onChange }) {
  return (
    <div className={styles.grid}>
      {TRADITIONS.map(t => (
        <button
          key={t.id}
          type="button"
          className={`${styles.btn} ${value === t.id ? styles.active : ''}`}
          onClick={() => onChange(t.id)}
        >
          <span className={styles.label}>{t.label}</span>
          <span className={styles.sub}>{t.subtitle}</span>
        </button>
      ))}
    </div>
  )
}
