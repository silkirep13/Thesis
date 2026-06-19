import {
  LineChart, Line, XAxis, YAxis, ReferenceLine,
  ResponsiveContainer, Tooltip,
} from 'recharts'
import styles from './PitchChart.module.css'

export default function PitchChart({ times, contour, tonicHz, tonicName }) {
  const data = times.map((t, i) => ({ t, f0: contour[i] }))

  const voiced = contour.filter(v => v !== null)
  // Use 5th/95th percentile instead of min/max to ignore outlier spikes
  const sorted  = [...voiced].sort((a, b) => a - b)
  const p5  = sorted[Math.floor(sorted.length * 0.05)] ?? sorted[0] ?? 80
  const p95 = sorted[Math.floor(sorted.length * 0.95)] ?? sorted[sorted.length - 1] ?? 800
  const pad = (p95 - p5) * 0.15
  const minHz = Math.max(20,  p5  - pad)
  const maxHz = Math.min(2000, p95 + pad)

  return (
    <div className={styles.wrap}>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
          <XAxis
            dataKey="t"
            type="number"
            domain={['dataMin', 'dataMax']}
            tickFormatter={v => `${Math.round(v)}s`}
            tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
            axisLine={{ stroke: 'var(--border)' }}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[Math.round(minHz), Math.round(maxHz)]}
            tickFormatter={v => `${v}`}
            tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
            axisLine={{ stroke: 'var(--border)' }}
            tickLine={false}
            width={40}
            unit=" Hz"
          />
          <Tooltip content={<PitchTooltip tonicHz={tonicHz} tonicName={tonicName} />} />
          {tonicHz && (
            <ReferenceLine
              y={tonicHz}
              stroke="var(--gold)"
              strokeDasharray="6 3"
              strokeOpacity={0.75}
              label={{
                value: `${tonicName}  ${tonicHz} Hz`,
                position: 'insideTopRight',
                fill: 'var(--gold)',
                fontSize: 10,
                opacity: 0.85,
              }}
            />
          )}
          <Line
            type="monotone"
            dataKey="f0"
            stroke="var(--gold-dim)"
            strokeWidth={1.5}
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

function PitchTooltip({ active, payload, tonicHz, tonicName }) {
  if (!active || !payload?.length || payload[0].value === null) return null
  const { t, f0 } = payload[0].payload
  const cents = tonicHz ? Math.round(1200 * Math.log2(f0 / tonicHz)) : null
  return (
    <div className={styles.tooltip}>
      <span className={styles.ttTime}>{t.toFixed(2)} s</span>
      <span>{f0.toFixed(1)} Hz</span>
      {cents !== null && (
        <span className={styles.ttCents}>
          {cents > 0 ? '+' : ''}{cents} ¢ from {tonicName}
        </span>
      )}
    </div>
  )
}
