import { useState, useRef, useEffect } from 'react'
import styles from './Uploader.module.css'

const LOADING_STAGES = [
  'Separating sources…',
  'Isolating vocal stem…',
  'Running pitch analysis…',
  'Detecting mode & tonic…',
  'Computing microtonal deviations…',
  'Finalising results…',
]

export default function Uploader({ onAnalyze, loading }) {
  const [dragOver, setDragOver]       = useState(false)
  const [file, setFile]               = useState(null)
  const [stageIdx, setStageIdx]       = useState(0)
  const inputRef                      = useRef()

  // Cycle through loading stage messages while waiting. The counter is reset
  // on teardown rather than at the top of the effect: resetting in the body
  // would set state during render and trigger a cascading re-render.
  useEffect(() => {
    if (!loading) return
    const id = setInterval(() => {
      setStageIdx(i => (i + 1) % LOADING_STAGES.length)
    }, 8000)
    return () => {
      clearInterval(id)
      setStageIdx(0)
    }
  }, [loading])

  const accept = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac']

  function handleFile(f) {
    if (!f) return
    setFile(f)
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  function handleSubmit() {
    if (file) onAnalyze(file)
  }

  return (
    <div className={styles.wrapper}>
      <div
        className={`${styles.dropzone} ${dragOver ? styles.dragOver : ''} ${file ? styles.hasFile : ''}`}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept.join(',')}
          hidden
          onChange={e => handleFile(e.target.files[0])}
        />
        {file ? (
          <>
            <div className={styles.fileIcon}>♪</div>
            <p className={styles.fileName}>{file.name}</p>
            <p className={styles.fileMeta}>{(file.size / 1024).toFixed(1)} KB</p>
            <p className={styles.hint}>Click to change file</p>
          </>
        ) : (
          <>
            <div className={styles.uploadIcon}>↑</div>
            <p className={styles.dropText}>Drop an audio file here</p>
            <p className={styles.hint}>MP3, WAV, FLAC, OGG, M4A, AAC</p>
          </>
        )}
      </div>

      <button
        className={styles.analyzeBtn}
        onClick={handleSubmit}
        disabled={!file || loading}
      >
        {loading ? <span className={styles.spinner} /> : null}
        {loading ? LOADING_STAGES[stageIdx] : 'Analyze Recording'}
      </button>
      {loading && (
        <p className={styles.loadingHint}>
          Source separation takes ~2–4 min on CPU for a full-length recording — please wait
        </p>
      )}
    </div>
  )
}
