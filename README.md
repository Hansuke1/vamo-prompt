# 🍍 Vamo Business Analysis Tool

A powerful web application that analyzes business data using multiple AI models (OpenAI GPT-4o-mini, Grok-2-1212, and Gemini 2.0 Flash) to provide comprehensive business insights.

## Features

- **Multi-Model AI Analysis**: Compare responses from OpenAI, Grok, and Gemini simultaneously
- **Prompt Testing**: Switch between different prompt versions or create custom prompts
- **Quick Examples**: Preset data for testing (Flapico, Zogo)
- **Structured Analysis**: Detailed business analysis with valuation, pros/cons, competitors
- **Beautiful UI**: Modern, responsive design with real-time results

## Setup

### Prerequisites

- Python 3.8+
- Flask
- API keys for OpenAI, Grok (X.AI), and Google Gemini

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd @faptest
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
FLASK_SECRET_KEY=your-secret-key-here
OPENAI_API_KEY=your-openai-api-key
GROK_API_KEY=your-grok-api-key
GEMINI_API_KEY=your-gemini-api-key
```

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:3002`

## Usage

1. **Select a Prompt**: Choose from preset prompts or create a custom one
2. **Enter Company Data**: 
   - Company name and tagline
   - Website markdown content
3. **Analyze**: Click "Analyze Business" to get insights from all three AI models
4. **Compare Results**: View side-by-side analysis from different models

## API Endpoints

- `GET /` - Main application interface
- `GET /api/prompts` - Get available analysis prompts
- `POST /api/analyze-business` - Analyze business data

## Technologies Used

- **Backend**: Flask (Python)
- **AI Models**: OpenAI GPT-4o-mini, Grok-2-1212, Gemini 2.0 Flash
- **Frontend**: HTML, CSS, JavaScript
- **Environment**: python-dotenv

## Project Structure

```
@faptest/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (not in repo)
├── .gitignore           # Git ignore rules
├── templates/           # HTML templates
├── venv/               # Virtual environment
└── README.md           # This file
```

## Deployment

### Vercel Deployment

This project is configured for easy deployment on Vercel:

1. Push to GitHub
2. Connect your GitHub repo to Vercel
3. Set environment variables in Vercel dashboard
4. Deploy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details 