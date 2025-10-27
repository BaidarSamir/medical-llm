import React, {useState} from 'react'

export default function ResultCard({result}){
  const [copied, setCopied] = useState(false)

  async function handleCopy(){
    await navigator.clipboard.writeText(JSON.stringify(result, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="result-card" role="article" aria-label="Analysis result">
      <div className="result-header">
        <span className="badge" aria-label={`Department: ${result.department}`}>
          {result.department}
        </span>
        <div className="actions">
          <button onClick={handleCopy} aria-label="Copy result as JSON">
            {copied ? '✓ Copied!' : 'Copy JSON'}
          </button>
          <a 
            href={`data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(result, null, 2))}`} 
            download="analysis.json"
            aria-label="Download result as JSON file"
          >
            Download
          </a>
        </div>
      </div>
      <div className="result-body">
        <pre>{result.description}</pre>
      </div>
      {copied && <div className="toast">Copied to clipboard!</div>}
    </div>
  )
}
