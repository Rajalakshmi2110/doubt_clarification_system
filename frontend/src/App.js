import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const validateQuestion = async () => {
    if (!question.trim()) return;
    
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post('/api/validate', {
        question: question
      });
      setResult(response.data);
    } catch (err) {
      setError('Error processing question: ' + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status?.toUpperCase()) {
      case 'VALID': return '#28a745';
      case 'WARNING': return '#ffc107';
      case 'REJECTED': return '#dc3545';
      default: return '#6c757d';
    }
  };

  const getValidationIcon = (method) => {
    return method === 'FLAN-T5' ? '🤖' : '📊';
  };

  const getStatusIcon = (status) => {
    switch (status?.toUpperCase()) {
      case 'VALID': return '✅';
      case 'WARNING': return '⚠️';
      case 'REJECTED': return '❌';
      default: return '❓';
    }
  };

  return (
    <div className="App">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">🎓</div>
            <div className="logo-text">
              <h1>Academic Doubt Clarification</h1>
              <span className="subtitle">Computer Networks Question Validation</span>
            </div>
          </div>
          <div className="status-badge">
            <span className="status-dot"></span>
            AI System Online
          </div>
        </div>
      </header>

      <main className="main-content">
        <div className="validation-panel">
          <div className="panel-header">
            <h2>Question Validation</h2>
            <p>Enter your Computer Networks question for intelligent validation</p>
          </div>
          
          <div className="input-section">
            <div className="input-wrapper">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g., How does TCP three-way handshake work?"
                className="question-input"
                rows="3"
                onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), validateQuestion())}
              />
              <button 
                onClick={validateQuestion} 
                disabled={loading || !question.trim()}
                className="validate-btn"
              >
                {loading ? (
                  <><span className="spinner"></span> Analyzing...</>
                ) : (
                  <>🔍 Validate Question</>
                )}
              </button>
            </div>
          </div>

          {error && (
            <div className="alert alert-error">
              <span className="alert-icon">⚠️</span>
              {error}
            </div>
          )}

          {result && (
            <div className="results-container">
              <div className="result-header">
                <div className="status-display">
                  <span className="status-icon">{getStatusIcon(result.status)}</span>
                  <span className="status-text" style={{ color: getStatusColor(result.status) }}>
                    {result.status}
                  </span>
                  <span className="confidence-score">Score: {result.final_score?.toFixed(2)}</span>
                </div>
                <div className="method-indicator">
                  {getValidationIcon(result.validation_method)} {result.validation_method}
                </div>
              </div>

              <div className="results-grid">
                <div className="result-card">
                  <h3>📝 Question Analysis</h3>
                  <div className="analysis-row">
                    <span className="label">Original:</span>
                    <span className="value">{result.original_question}</span>
                  </div>
                  <div className="analysis-row">
                    <span className="label">Processed:</span>
                    <span className="value">{result.corrected_question}</span>
                  </div>
                  {result.corrections_applied && result.corrections_applied.length > 0 && (
                    <div className="analysis-row">
                      <span className="label">Corrections:</span>
                      <span className="value corrections">{result.corrections_applied.join(', ')}</span>
                    </div>
                  )}
                </div>

                <div className="result-card">
                  <h3>🔬 Technical Analysis</h3>
                  {result.relevance_details && (
                    <>
                      <div className="analysis-row">
                        <span className="label">Method:</span>
                        <span className="value">{result.relevance_details.method}</span>
                      </div>
                      {result.relevance_details.reason && (
                        <div className="analysis-row">
                          <span className="label">Analysis:</span>
                          <span className="value">{result.relevance_details.reason}</span>
                        </div>
                      )}
                    </>
                  )}
                  
                  {result.component_scores && (
                    <div className="scores-section">
                      <h4>Component Scores</h4>
                      <div className="score-bar">
                        <span>Semantic Quality</span>
                        <div className="progress-bar">
                          <div className="progress" style={{width: `${result.component_scores.semantic_sanity * 100}%`}}></div>
                        </div>
                        <span>{result.component_scores.semantic_sanity?.toFixed(2)}</span>
                      </div>
                      <div className="score-bar">
                        <span>Syllabus Relevance</span>
                        <div className="progress-bar">
                          <div className="progress" style={{width: `${result.component_scores.syllabus_relevance * 100}%`}}></div>
                        </div>
                        <span>{result.component_scores.syllabus_relevance?.toFixed(2)}</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {result.feedback && (
                <div className="feedback-panel">
                  <h3>💡 System Feedback</h3>
                  <div className="feedback-grid">
                    {result.feedback.strengths && result.feedback.strengths.length > 0 && (
                      <div className="feedback-section strengths">
                        <h4>✅ Strengths</h4>
                        <ul>
                          {result.feedback.strengths.map((strength, idx) => (
                            <li key={idx}>{strength}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {result.feedback.issues && result.feedback.issues.length > 0 && (
                      <div className="feedback-section issues">
                        <h4>⚠️ Issues</h4>
                        <ul>
                          {result.feedback.issues.map((issue, idx) => (
                            <li key={idx}>{issue}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {result.feedback.suggestions && result.feedback.suggestions.filter(s => s).length > 0 && (
                      <div className="feedback-section suggestions">
                        <h4>💡 Suggestions</h4>
                        <ul>
                          {result.feedback.suggestions.filter(s => s).map((suggestion, idx) => (
                            <li key={idx}>{suggestion}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;