import React, { useState } from 'react';

export default function UrlInput() {
  const [url, setUrl] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    alert(`URL entered: ${url}`);
    // Later you will send this URL to your FastAPI backend
  };

  return (
    <form onSubmit={handleSubmit}>
      <label htmlFor="urlInput">Enter URL:</label>
      <input
        type="url"
        id="urlInput"
        placeholder="https://example.com"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        required
        style={{ width: '300px', marginRight: '10px' }}
      />
      <button type="submit">Check URL</button>
    </form>
  );
}
