import React, {useState} from 'react'
import axios from 'axios'
import SearchBar from './components/SearchBar'
import ResultCard from './components/ResultCard'

export default function App(){
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleAnalyze(text){
    setLoading(true)
    setError(null)
    setResult(null)
    try{
      const resp = await axios.post('/analyze', { symptom_text: text })
      setResult(resp.data)
    }catch(e){
      const errorMsg = e.response?.data?.detail || e.message || 'Analysis failed'
      setError(errorMsg)
    }finally{
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header>
        <h1>Medical Symptom Analyzer</h1>
        <p className="tagline">AI-powered symptom analysis with department recommendation and medical guidance</p>
      </header>

      <main>
        <SearchBar onAnalyze={handleAnalyze} loading={loading} />

        {loading && (
          <div className="loading" role="status" aria-live="polite">
            Analyzing your symptoms… this may take up to 90 seconds
          </div>
        )}
        
        {error && (
          <div className="error" role="alert" aria-live="assertive">
            <strong>Error:</strong> {error}
          </div>
        )}
        
        {result && <ResultCard result={result} />}

        <section className="examples">
          <h3>Try example symptoms</h3>
          <div className="chips">
            <button 
              onClick={() => handleAnalyze("I feel tightness in my chest and I'm having trouble breathing")}
              disabled={loading}
              aria-label="Example: chest tightness and breathing difficulty"
            >
              Chest tightness
            </button>
            <button 
              onClick={() => handleAnalyze("I am feeling like fainting and dizzy")}
              disabled={loading}
              aria-label="Example: fainting and dizziness"
            >
              Fainting & dizziness
            </button>
            <button 
              onClick={() => handleAnalyze("I have a severe headache that won't go away")}
              disabled={loading}
              aria-label="Example: severe persistent headache"
            >
              Severe headache
            </button>
            <button 
              onClick={() => handleAnalyze("I'm experiencing persistent fatigue and weakness")}
              disabled={loading}
              aria-label="Example: fatigue and weakness"
            >
              Fatigue & weakness
            </button>
          </div>
        </section>
      </main>

      <footer>
        <small>
          <strong>Disclaimer:</strong> This tool provides informational guidance only and does not replace professional medical advice. 
          Always consult a healthcare provider for medical concerns.
        </small>
      </footer>
    </div>
  )
}
