import styles from './ResultsPanel.module.css'
import PitchChart from './PitchChart'
import ModeInfoCard from './ModeInfoCard'
import NotationStrip from './NotationStrip'

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
        <Stat label="Mean Pitch" value={`${data.mean_pitch_hz} Hz`} />
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

      {data.notation?.length > 0 && (
        <div className={styles.section}>
          <span className={styles.label}>Scale-Degree Transcription</span>
          <p className={styles.sectionHint}>
            Each segmented note snapped to the nearest degree of {data.detected_mode} · symbol shows deviation from that degree's theoretical position
          </p>
          <NotationStrip notes={data.notation} />
        </div>
      )}

      <div className={styles.section}>
        <span className={styles.label}>Microtonal Deviations from 12-TET</span>
        <p className={styles.sectionHint}>
          Percentile distribution of cent deviations — how far each pitch sits from equal temperament.
        </p>
        {!data.microtonal_reliable && (
          <div className={styles.warning}>
            ⚠ Voiced frame rate too low ({Math.round(data.voiced_frames / data.total_frames * 100)}%) — likely choir or polyphonic recording. Microtonal analysis requires a solo/monophonic source.
          </div>
        )}
        <div className={styles.tags}>
          {data.microtonal_deviations.length > 0
            ? data.microtonal_deviations.map((d, i) => {
                const val = parseFloat(d.split(': ')[1])
                return (
                  <span key={i} className={`${styles.tag} ${val >= 0 ? styles.pos : styles.neg}`}>
                    {d}
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
        Engine: <strong>{data.engine}</strong> &nbsp;·&nbsp;
        {data.voiced_frames} voiced / {data.total_frames} total frames
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
