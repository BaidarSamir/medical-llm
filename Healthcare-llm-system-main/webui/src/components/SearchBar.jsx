import React, {useState} from 'react'

export default function SearchBar({onAnalyze, loading}){
  const [text, setText] = useState('')

  function handleSubmit(){
    if(text.trim().length >= 5){
      onAnalyze(text)
    }
  }

  function handleKeyDown(e){
    if(e.key === 'Enter' && e.ctrlKey){
      handleSubmit()
    }
  }

  return (
    <div className="searchbar">
      <textarea
        placeholder="Describe your symptoms (e.g., 'I feel tightness in my chest and trouble breathing')"
        value={text}
        onChange={(e)=>setText(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={4}
        aria-label="Symptom description input"
        aria-describedby="symptom-hint"
      />
      <small id="symptom-hint" style={{display:'block', marginTop:'4px', color:'var(--muted)'}}>
        Tip: Press Ctrl+Enter to analyze quickly
      </small>
      <div className="controls">
        <button 
          disabled={loading || text.trim().length < 5} 
          onClick={handleSubmit}
          aria-label="Analyze symptoms"
        >
          {loading ? 'Analyzing...' : 'Analyze'}
        </button>
        <button 
          onClick={()=>{ setText('') }} 
          className="secondary"
          disabled={loading}
          aria-label="Clear input"
        >
          Clear
        </button>
      </div>
    </div>
  )
}
