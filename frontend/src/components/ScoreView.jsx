import { useEffect, useRef, useState } from 'react'
import styles from './ScoreView.module.css'

const ZOOM_STEPS = [0.4, 0.55, 0.7, 0.85, 1.0]

// Canvas limits. Browsers silently produce a blank bitmap past these, with no
// error thrown, so the export scale is clamped to stay inside them rather than
// trusting the draw to fail loudly.
const MAX_CANVAS_DIM = 16000      // Chrome caps a single side at 16384
const MAX_CANVAS_AREA = 250e6     // and total area at roughly 268M pixels
const EXPORT_SCALE = 2            // 2x the on-screen size, for a crisp image

/**
 * Rasterise the engraved SVG to a PNG blob.
 *
 * PNG rather than JPEG: a score is line art — thin staff lines, stems and
 * quarter-tone accidentals — which JPEG's block compression smears into
 * visible artefacts. PNG is lossless and compresses flat white backgrounds
 * extremely well, so it is both sharper and usually smaller here.
 */
async function svgToPngBlob(svg) {
  // Size from getBBox(), the real extent of the drawn content. The element's
  // layout box is the wrong measure here: the score sits in a horizontally
  // scrollable panel, so its laid-out width is whatever fits on screen while
  // the engraving continues past it, and cropping to that cuts every system
  // off at the right edge. The largest of all three candidates is used so no
  // measure can fall outside the exported image.
  const bb = svg.getBBox()
  const rect = svg.getBoundingClientRect()
  const attrW = parseFloat(svg.getAttribute('width')) || 0
  const attrH = parseFloat(svg.getAttribute('height')) || 0

  const x = Math.min(0, bb.x)
  const y = Math.min(0, bb.y)
  const w = Math.ceil(Math.max(bb.x + bb.width, rect.width, attrW) - x)
  const h = Math.ceil(Math.max(bb.y + bb.height, rect.height, attrH) - y)

  // Shrink the export if a full-length score would overflow the canvas.
  let scale = EXPORT_SCALE
  scale = Math.min(scale, MAX_CANVAS_DIM / w, MAX_CANVAS_DIM / h)
  scale = Math.min(scale, Math.sqrt(MAX_CANVAS_AREA / (w * h)))
  scale = Math.max(scale, 0.1)

  // Work on a copy with explicit dimensions: the live element is sized by CSS,
  // which an Image() built from the serialised markup cannot see.
  const clone = svg.cloneNode(true)
  clone.setAttribute('width', w)
  clone.setAttribute('height', h)
  clone.setAttribute('viewBox', `0 0 ${w} ${h}`)
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')

  const markup = new XMLSerializer().serializeToString(clone)
  const url = URL.createObjectURL(new Blob([markup], { type: 'image/svg+xml;charset=utf-8' }))

  try {
    const img = new Image()
    await new Promise((resolve, reject) => {
      img.onload = resolve
      img.onerror = () => reject(new Error('Η παρτιτούρα δεν μπόρεσε να μετατραπεί σε εικόνα'))
      img.src = url
    })

    const canvas = document.createElement('canvas')
    canvas.width = Math.round(w * scale)
    canvas.height = Math.round(h * scale)
    const ctx = canvas.getContext('2d')
    // OSMD draws in black with no background of its own; without this the
    // transparent areas would render black in most image viewers.
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    const blob = await new Promise(res => canvas.toBlob(res, 'image/png'))
    if (!blob) throw new Error('Η δημιουργία της εικόνας απέτυχε')
    return { blob, width: canvas.width, height: canvas.height, scale }
  } finally {
    URL.revokeObjectURL(url)
  }
}

function saveBlob(blob, name) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export default function ScoreView({ musicxml, tempoBpm, pulseStrength, metrical, filename }) {
  const hostRef = useRef(null)
  const osmdRef = useRef(null)
  const [error, setError] = useState(null)
  const [ready, setReady] = useState(false)
  const [exporting, setExporting] = useState(false)
  // A full-length transcription runs to hundreds of notes, so it is engraved
  // small by default and the reader zooms in rather than scrolling a page
  // that is many thousands of pixels tall.
  const [zoom, setZoom] = useState(0.55)

  useEffect(() => {
    if (!musicxml || !hostRef.current) return
    let cancelled = false

    ;(async () => {
      try {
        setReady(false)
        setError(null)
        // Loaded on demand: OSMD is large and only needed once a score exists.
        const { OpenSheetMusicDisplay } = await import('opensheetmusicdisplay')
        if (cancelled) return

        if (!osmdRef.current) {
          osmdRef.current = new OpenSheetMusicDisplay(hostRef.current, {
            autoResize: true,
            // Heading: file name on top, recognised instrument beneath it.
            // The staff label is turned off because it would only repeat the
            // instrument name that already appears in the subtitle.
            drawTitle: true,
            drawSubtitle: true,
            drawComposer: false,
            drawPartNames: false,
            drawingParameters: 'compact',
          })
        }
        await osmdRef.current.load(musicxml)
        if (cancelled) return
        osmdRef.current.zoom = zoom
        osmdRef.current.render()
        setReady(true)
      } catch (e) {
        if (!cancelled) setError(e?.message || String(e))
      }
    })()

    return () => { cancelled = true }
    // `zoom` is deliberately not a dependency: it is only read to set the
    // initial scale. Re-parsing the whole score on every zoom step would be
    // wasteful, so the effect below re-renders at the new scale instead.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [musicxml])

  // Re-render at the new scale without re-parsing the file.
  useEffect(() => {
    if (!ready || !osmdRef.current) return
    osmdRef.current.zoom = zoom
    osmdRef.current.render()
  }, [zoom, ready])

  if (!musicxml) return null

  const baseName = (filename || 'transcription').replace(/\.[^.]+$/, '')

  const downloadXml = () => {
    saveBlob(
      new Blob([musicxml], { type: 'application/vnd.recordare.musicxml+xml' }),
      `${baseName}.musicxml`,
    )
  }

  const downloadPng = async () => {
    const svg = hostRef.current?.querySelector('svg')
    if (!svg) return
    setExporting(true)
    setError(null)
    try {
      const { blob } = await svgToPngBlob(svg)
      saveBlob(blob, `${baseName}.png`)
    } catch (e) {
      setError(e?.message || String(e))
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className={styles.wrap}>
      <div className={styles.meta}>
        <span className={metrical ? styles.pillMetric : styles.pillFree}>
          {metrical ? 'Σταθερός παλμός' : 'Ελεύθερος ρυθμός'}
        </span>
        <span className={styles.metaItem}>≈ {Math.round(tempoBpm)} BPM</span>
        <span className={styles.metaItem}>ισχύς παλμού {pulseStrength?.toFixed(2)}</span>

        <div className={styles.actions}>
          <div className={styles.zoom}>
            <button
              type="button"
              onClick={() => setZoom(z => ZOOM_STEPS[Math.max(0, ZOOM_STEPS.indexOf(z) - 1)])}
              disabled={zoom === ZOOM_STEPS[0]}
              aria-label="Σμίκρυνση"
            >−</button>
            <span className={styles.zoomVal}>{Math.round(zoom * 100)}%</span>
            <button
              type="button"
              onClick={() => setZoom(z => ZOOM_STEPS[Math.min(ZOOM_STEPS.length - 1, ZOOM_STEPS.indexOf(z) + 1)])}
              disabled={zoom === ZOOM_STEPS[ZOOM_STEPS.length - 1]}
              aria-label="Μεγέθυνση"
            >+</button>
          </div>

          <button type="button" className={styles.dl} onClick={downloadXml}>
            Λήψη MusicXML
          </button>

          <button
            type="button"
            className={styles.dl}
            onClick={downloadPng}
            disabled={!ready || exporting}
          >
            {exporting ? 'Εξαγωγή…' : 'Λήψη PNG'}
          </button>
        </div>
      </div>

      <p className={styles.freeNote}>
        {metrical
          ? 'Εντοπίστηκε σταθερός παλμός και οι διάρκειες μετρήθηκαν ως προς αυτόν. Η '
            + 'παρτιτούρα δεν φέρει ένδειξη μέτρου: το σύστημα αναγνωρίζει ότι υπάρχει '
            + 'παλμός, όχι πόσοι χτύποι συγκροτούν το μέτρο — κρίσιμο σε ρεπερτόριο με '
            + '7/8 και 9/8.'
          : 'Δεν εντοπίστηκε σταθερός παλμός, οπότε η μεταγραφή αποδίδεται χωρίς μέτρο '
            + 'και οι διάρκειες είναι σχετικές. Αυτό είναι το αναμενόμενο για βυζαντινό '
            + 'μέλος και ταξίμι.'}
      </p>

      {error && <div className={styles.err}>Η παρτιτούρα δεν μπόρεσε να αποδοθεί: {error}</div>}
      {!ready && !error && <div className={styles.loading}>Απόδοση παρτιτούρας…</div>}

      <div ref={hostRef} className={styles.host} />
    </div>
  )
}
