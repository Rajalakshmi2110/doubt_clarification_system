# Academic Doubt Clarification System - React Frontend

Modern React-based web interface for the Academic Doubt Clarification System.

## Features

- **Live Question Validation**: Real-time validation of Computer Networks questions
- **Intelligent Analysis**: ML-based validation using FLAN-T5 or embedding similarity
- **Visual Feedback**: Color-coded status indicators and detailed analysis
- **Responsive Design**: Works on desktop and mobile devices
- **Debug Information**: Shows validation method and detailed reasoning

## Quick Start

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Start Backend API
```bash
# From project root
python api_server.py
```
Backend runs on: http://localhost:8000

### 3. Start Frontend
```bash
# From frontend directory
npm start
```
Frontend runs on: http://localhost:3000

## Validation States

- **🟢 VALID**: Question is well-formed and clearly within syllabus scope
- **🟡 WARNING**: Question is networking-related but weakly grounded or advanced  
- **🔴 REJECTED**: Question is non-networking or technically incorrect

## Validation Methods

- **🤖 FLAN-T5**: Contextual AI analysis using fine-tuned language model
- **📊 Embedding-based**: Semantic similarity matching with dataset

## API Endpoints

- `POST /api/validate`: Validate a question
- `GET /api/health`: Check system health

## Technology Stack

- **Frontend**: React 18, Axios
- **Backend**: Flask, Python
- **ML Models**: FLAN-T5, SentenceTransformers
- **Styling**: CSS Grid, Flexbox

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── App.js          # Main React component
│   ├── App.css         # Styling
│   └── index.js        # Entry point
└── package.json        # Dependencies
```