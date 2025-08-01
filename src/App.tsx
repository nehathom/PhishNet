import { useState } from 'react'
import './App.css'

function App() {
  const [url, setUrl] = useState('')
  const [result, setResult] = useState<string | null>(null)
  const [explanation, setExplanation] = useState<string | null>(null)  // <-- add state for explanation


  const handleSubmit = async () => {
    if (!url) return alert("Please enter a URL");
    
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url }),
      });

      const data = await response.json();
      setResult(`Prediction: ${data.prediction} | Is Phishing: ${data.is_phishing}`);
      setExplanation(data.explanation ?? "No explanation available.");

    } catch (error) {
      console.error('Error:', error);
      setResult(`An error occurred while connecting to the server. ${error}`);

    }
  };

  return (
    <div className="App">
      <h1>PhishNet</h1>
      <div style={{ marginBottom: '1rem' }}>
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Enter URL here"
          style={{ padding: '0.5rem', width: '300px' }}
        />
        <button onClick={handleSubmit} style={{ marginLeft: '1rem', padding: '0.5rem 1rem' }}>
          Check URL
        </button>
      </div>
      {result && <p>{result}</p>}
      {explanation && (
        <div>
          <h3>Explanation:</h3>
          <p>{explanation}</p>
        </div>
      )}
    </div>
  );
}

export default App;

