import { useState } from 'react'
import Uploader from './components/Uploader'
import ResultsPanel from './components/ResultsPanel'
import TraditionSelector from './components/TraditionSelector'
import styles from './App.module.css'

const API = ''

export default function App() {
  const [loading, setLoading]     = useState(false)
  const [result, setResult]       = useState(null)
  const [error, setError]         = useState(null)
  const [tradition, setTradition] = useState('byzantine')

  function handleTraditionChange(t) {
    setTradition(t)
    setResult(null)
    setError(null)
  }

  async function handleAnalyze(file) {
    setLoading(true)
    setError(null)
    setResult(null)

    const form = new FormData()
    form.append('file', file)
    form.append('tradition', tradition)

    try {
      const res = await fetch(`${API}/api/analyze`, { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Analysis failed')
      }
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <div className={styles.titleGroup}>
            <span className={styles.subtitle}>Thesis Project - Music Information Retrieval</span>
            <h1 className={styles.title}>Byzantine & Eastern Music Analysis</h1>
          </div>
        </div>
      </header>

      <main className={styles.main}>
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Select Tradition</h2>
          <p className={styles.sectionDesc}>
            Choose the musical tradition before uploading — this restricts mode
            detection to the correct scale family and avoids cross-tradition confusion.
          </p>
          <TraditionSelector value={tradition} onChange={handleTraditionChange} />
        </section>

        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Upload a Recording</h2>
          <p className={styles.sectionDesc}>
            Upload an audio file to extract the modal structure, pitch statistics,
            and microtonal deviations for the selected tradition.
          </p>
          <Uploader onAnalyze={handleAnalyze} loading={loading} />
        </section>

        {error && (
          <div className={styles.errorBox}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <section className={styles.section}>
            <h2 className={styles.sectionTitle}>Analysis Results</h2>
            <p className={styles.sectionDesc}>
              File: <code>{result.filename}</code>
            </p>
            <ResultsPanel data={result} />
          </section>
        )}
      </main>

      <footer className={styles.footer}>
        <span>MIR Thesis — Byzantine &amp; Eastern Music · Stack: FastAPI · React · librosa · CREPE · Demucs · YAMNet · music21</span>
      </footer>
    </div>
  )
}
