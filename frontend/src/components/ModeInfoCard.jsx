import styles from './ModeInfoCard.module.css'

export default function ModeInfoCard({ info, genus, tieGroupSize }) {
  if (!info) return null

  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <span className={styles.badge}>{info.tradition}</span>
        {genus && <span className={styles.genusBadge}>{genus}</span>}
        {info.alt_name && <span className={styles.altName}>{info.alt_name}</span>}
      </div>
      <p className={styles.description}>{info.description}</p>
      {tieGroupSize > 1 && (
        <p className={styles.tieNote}>
          Resolved via melodic-register analysis: {tieGroupSize} modes shared this scale at chroma resolution.
        </p>
      )}
    </div>
  )
}
