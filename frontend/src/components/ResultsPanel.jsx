import styles from './ResultsPanel.module.css'
import PitchChart from './PitchChart'
import ModeInfoCard from './ModeInfoCard'
import ScoreView from './ScoreView'

// Plain wording for the percentile keys the backend sends. "p50: +3¢" is
// precise but statistical; a reader wants to know that half the notes fall
// below that figure, not what a percentile is.
const PERCENTILE_LABELS = {
  p10: 'Low 10%',
  p25: 'Low 25%',
  p50: 'Median',
  p75: 'High 25%',
  p90: 'High 10%',
}

export default function ResultsPanel({ data }) {
  if (!data) return null

  const voicedPct = data.total_frames > 0
    ? Math.round((data.voiced_frames / data.total_frames) * 100)
    : 0

  return (
    <div className={styles.panel}>
      <div className={styles.topRow}>
        <div className={styles.modeCard}>
          <span className={styles.label}>Detected Mode / Maqam</span>
          <span className={styles.modeName}>{data.detected_mode}</span>
          <span className={styles.tonic}>Tonic: <strong>{data.tonic}</strong></span>
        </div>
        <div className={styles.confidenceCard}>
          <span className={styles.label}>Confidence</span>
          <span className={styles.confidenceVal}>{Math.round(data.confidence * 100)}%</span>
          <div className={styles.bar}>
            <div className={styles.barFill} style={{ width: `${data.confidence * 100}%` }} />
          </div>
        </div>
      </div>

      <ModeInfoCard info={data.mode_info} genus={data.genus} tieGroupSize={data.tie_group_size} />

      <div className={styles.grid}>
        <Stat label="Pitch Range" value={`${data.pitch_range_cents} ¢`} />
        <Stat label="Duration" value={`${data.duration_seconds}s`} />
        <Stat label="Voiced Frames" value={`${voicedPct}%`} />
      </div>

      {data.pitch_contour?.length > 0 && (
        <div className={styles.section}>
          <span className={styles.label}>Pitch Contour</span>
          <p className={styles.sectionHint}>
            CREPE pitch track over time · dashed line = detected tonic ({data.tonic} · {data.tonic_hz} Hz)
          </p>
          <PitchChart
            times={data.pitch_times}
            contour={data.pitch_contour}
            tonicHz={data.tonic_hz}
            tonicName={data.tonic}
          />
        </div>
      )}

      {data.musicxml && (
        <div className={styles.section}>
          <span className={styles.label}>Παρτιτούρα</span>
          <p className={styles.sectionHint}>
            Μεταγραφή σε MusicXML · αλλοιώσεις τεταρτημορίου γράφονται μόνο όπου τις
            ορίζει ο ίδιος ο τρόπος (βυζαντινό, αραβικό) · οι αποκλίσεις της εκτέλεσης
            αναφέρονται σε σεντ παρακάτω
          </p>
          <ScoreView
            musicxml={data.musicxml}
            tempoBpm={data.tempo_bpm}
            pulseStrength={data.pulse_strength}
            metrical={data.metrical}
            filename={data.filename}
          />
        </div>
      )}

      <div className={styles.section}>
        <span className={styles.label}>Microtonal Deviations from 12-TET</span>
        <p className={styles.sectionHint}>
          How far the sung pitches sit from the piano keys, in cents. A distribution
          centred away from zero is the signature of intervals that equal temperament
          cannot express.
        </p>
        {!data.microtonal_reliable && (
          <div className={styles.warning}>
            ⚠ Voiced frame rate too low ({Math.round(data.voiced_frames / data.total_frames * 100)}%) — likely choir or polyphonic recording. Microtonal analysis requires a solo/monophonic source.
          </div>
        )}
        <div className={styles.tags}>
          {data.microtonal_deviations.length > 0
            ? data.microtonal_deviations.map((d, i) => {
                const [key, raw] = d.split(': ')
                const val = parseFloat(raw)
                return (
                  <span key={i} className={`${styles.tag} ${val >= 0 ? styles.pos : styles.neg}`}>
                    {PERCENTILE_LABELS[key] ?? key} {raw}
                  </span>
                )
              })
            : <span className={styles.empty}>
                {data.microtonal_reliable ? 'No pitch data' : 'Unavailable — see warning above'}
              </span>
          }
        </div>
      </div>

      {data.detected_instruments.length > 0 && (
        <div className={styles.section}>
          <span className={styles.label}>Detected Instruments</span>
          <div className={styles.tags}>
            {data.detected_instruments.map((inst, i) => (
              <span
                key={i}
                className={`${styles.instTag} ${inst === data.melody_instrument ? styles.instTagClassified : ''}`}
              >
                {inst}
              </span>
            ))}
          </div>
          <p className={styles.sectionHint}>
            {data.melody_instrument
              ? `Melody instrument identified by the trained classifier (${Math.round(data.melody_instrument_confidence * 100)}% confidence) · other labels are stem-energy categories`
              : 'Broad stem-energy categories — the melody instrument could not be named confidently enough to identify a specific instrument'}
          </p>
        </div>
      )}

      <div className={styles.notice}>
        Engine: <strong>{data.engine}</strong>
      </div>
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div className={styles.statBox}>
      <span className={styles.statLabel}>{label}</span>
      <span className={styles.statVal}>{value}</span>
    </div>
  )
}
