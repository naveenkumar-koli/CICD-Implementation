from fastapi import FastAPI, Request, Form, File, UploadFile, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager

# Add scheduler imports
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
import atexit
import threading

# Your existing imports
import pandas as pd
import pickle
from typing import List, Optional
from pydantic import BaseModel
import os
from pathlib import Path
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from fastapi.middleware.cors import CORSMiddleware
import math
from datetime import datetime
from fastapi.encoders import jsonable_encoder
import json
from fastapi.responses import FileResponse
import io
import uuid
import shutil
import string

# Import the model trainer and utils
from app.model_trainer import ModelTrainer
from app.utils import EnhancedSentimentAnalyzer, preprocess_text
from fastapi import HTTPException
import sys

# Initialize NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize enhanced sentiment analyzer and model trainer
sentiment_analyzer = EnhancedSentimentAnalyzer()
model_trainer = ModelTrainer()

# ====================== SIMPLIFIED SCHEDULER SETUP ======================

# Disable local scheduler in CI/CD environment
if os.getenv('GITHUB_ACTIONS') != 'true':
    # Your existing scheduler code
    scheduler = BackgroundScheduler(timezone='Asia/Kolkata')
    scheduler_enabled = True
    print("[SCHEDULER] Local/Production environment - scheduler enabled")
else:
    scheduler = None
    scheduler_enabled = False
    print("[SCHEDULER] Running in GitHub Actions - local scheduler disabled")

# Global variables for model reloading
model_lock = threading.Lock()
model_reload_needed = False

# Global variables for application
single_analysis_log = []
analysis_history = {}
processed_data = None


def validate_english_text(text: str) -> dict:
    """
    Validate that text contains US English language with numbers allowed,
    but no mathematical operations or other languages.
    """
    # Check if text is empty
    if not text.strip():
        return {"valid": False, "message": "Text cannot be empty"}

    # Check length
    if len(text) > 1000:
        return {"valid": False, "message": "Text cannot exceed 1000 characters"}

    # Check for mathematical expressions (not just numbers)
    # Patterns that indicate calculations rather than descriptive numbers
    math_expression_patterns = [
        r'\d\s*[\+\-\*\/\=\%]\s*\d',  # e.g., "5 + 3", "10-2"
        r'\d+\s*=\s*\d+',  # e.g., "5 = 5"
        r'\(\s*\d+\s*\)',  # e.g., "(5)" in mathematical context
        r'\d+\s*\%',  # e.g., "50%" (but we'll allow this in context)
        r'\b\d+\s*(?:times|plus|minus|divided by|multiplied by)\s*\d+\b',  # English math terms
    ]

    for pattern in math_expression_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return {"valid": False, "message": "Text cannot contain mathematical operations"}

    # Check for standalone mathematical operators without context
    standalone_math_ops = re.findall(r'[\+\-\*\/\=\%]', text)
    if standalone_math_ops and len(text.strip()) < 20:  # Short text with math ops
        return {"valid": False, "message": "Text appears to contain mathematical operations"}

    # Check for non-English characters (using Unicode ranges for basic Latin)
    # This allows: A-Z, a-z, numbers, spaces, punctuation, and common symbols
    allowed_chars = set(
        string.ascii_letters + string.digits + string.whitespace + string.punctuation + "°$€£¥¢@#&_–—'\"")

    for char in text:
        if char not in allowed_chars:
            # Allow some common accented characters used in English loanwords
            if char in 'áéíóúàèìòùäëïöüâêîôûñç':
                continue
            return {"valid": False, "message": f"Text contains non-English characters or symbols: '{char}'"}

    # Check for excessive punctuation or symbols
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    if punctuation_count > len(text) * 0.3:  # More than 30% punctuation
        return {"valid": False, "message": "Text contains too many symbols or punctuation"}

    # Check for code-like patterns
    code_patterns = [
        r'\b(def|class|import|from|return|if|else|for|while|try|except|print)\b',
        r'\b(function|var|let|const|console\.log|document\.)\b',
        r'\b(select|from|where|insert|update|delete)\b',
        r'\\[a-zA-Z]',  # Backslash followed by letter
        r'\.(py|js|java|cpp|html|css|sql)$'  # File extensions
    ]

    for pattern in code_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return {"valid": False, "message": "Text appears to contain code or technical syntax"}

    # Check for URL patterns
    url_pattern = r'https?://|www\.|\.(?:com|org|net|edu|gov)'
    if re.search(url_pattern, text, re.IGNORECASE):
        return {"valid": False, "message": "Text cannot contain URLs or website references"}

    # Check for excessive capitalization (might indicate shouting or code)
    upper_count = sum(1 for char in text if char.isupper())
    if upper_count > len(text) * 0.5:  # More than 50% uppercase
        return {"valid": False, "message": "Text contains excessive capitalization"}

    # Additional check: Ensure text has meaningful English content
    # Count alphabetic characters vs numbers
    alpha_count = sum(1 for char in text if char.isalpha())
    digit_count = sum(1 for char in text if char.isdigit())
    if digit_count > alpha_count:  # More numbers than letters
        return {"valid": False, "message": "Text appears to be numeric rather than descriptive English"}

    return {"valid": True, "message": "Text is valid US English"}


def scheduled_model_training():
    """Scheduled task to retrain the model every 3 days"""
    global model, vectorizer, encoder, model_reload_needed
    try:
        print(f"[SCHEDULER] Starting scheduled model retraining at {datetime.now()}")

        # Train new model
        training_result = model_trainer.train_model(generate_report=True)

        print(f"[SCHEDULER] Model retraining completed successfully!")
        print(f"[SCHEDULER] New Accuracy: {training_result['accuracy']:.4f}")
        print(f"[SCHEDULER] New F1 Score: {training_result['cv_f1_score']:.4f}")

        # Set flag to reload models in main thread
        with model_lock:
            model_reload_needed = True

        # Cleanup old backups
        model_trainer.cleanup_old_backups()

        print(f"[SCHEDULER] Scheduled retraining completed successfully at {datetime.now()}")

        # Log the training event
        log_training_event(training_result, "scheduled")

    except Exception as e:
        print(f"[SCHEDULER] Error during scheduled model training: {str(e)}")
        log_training_event({"error": str(e)}, "scheduled_failed")


def log_training_event(result, event_type):
    """Log training events to a file"""
    try:
        log_file = os.path.join(model_trainer.models_dir, "training_log.json")
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "result": result,
            "github_actions": os.getenv('GITHUB_ACTIONS') == 'true'
        }

        # Read existing logs
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)

        # Add new log entry
        logs.append(log_entry)

        # Keep only last 50 entries
        logs = logs[-50:]

        # Write back to file
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    except Exception as e:
        print(f"Error logging training event: {e}")


def reload_models_if_needed():
    """Reload models if retraining has occurred"""
    global model, vectorizer, encoder, model_reload_needed
    with model_lock:
        if model_reload_needed:
            try:
                # Reload models
                model, vectorizer, encoder = load_models()
                model_reload_needed = False
                print("[SCHEDULER] Models reloaded successfully after retraining")
            except Exception as e:
                print(f"[SCHEDULER] Error reloading models: {str(e)}")


# Initialize scheduler if enabled
if scheduler_enabled:
    try:
        # Schedule the retraining task to run every 3 days
        scheduler.add_job(
            func=scheduled_model_training,
            trigger=IntervalTrigger(days=3),
            id='model_retraining',
            name='Retrain sentiment model every 3 days',
            replace_existing=True,
            max_instances=1
        )

        # Start the scheduler
        scheduler.start()
        print(f"[SCHEDULER] Scheduler started successfully")
        print(f"[SCHEDULER] Next retraining scheduled for: {scheduler.get_job('model_retraining').next_run_time}")

        # Ensure scheduler shuts down when the application stops
        atexit.register(lambda: scheduler.shutdown() if scheduler.running else None)

    except Exception as e:
        print(f"[SCHEDULER] Error starting scheduler: {str(e)}")


# Lifespan manager for FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[STARTUP] Application starting up...")
    print(f"[STARTUP] GitHub Actions environment: {os.getenv('GITHUB_ACTIONS') == 'true'}")
    print(f"[STARTUP] Scheduler enabled: {scheduler_enabled}")

    if scheduler_enabled:
        print(f"[STARTUP] Scheduler running: {scheduler.running}")
        # Print next scheduled runs
        jobs = scheduler.get_jobs()
        for job in jobs:
            print(f"[STARTUP] Job '{job.name}' next run: {job.next_run_time}")
    else:
        print("[STARTUP] No local scheduler - using GitHub Actions for training")

    yield

    # Shutdown
    print("[SHUTDOWN] Application shutting down...")
    if scheduler_enabled and scheduler.running:
        scheduler.shutdown()
        print("[SHUTDOWN] Scheduler stopped")


# Initialize FastAPI with lifespan
app = FastAPI(
    title="Sentiment Analysis with Auto-Retraining",
    description="Sentiment Analysis API with automated model retraining every 3 days",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== FIXED PATH RESOLUTION ======================
# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Template and static directories based on your project structure
template_dir = os.path.join(current_dir, "templates")
static_dir = os.path.join(current_dir, "static")

# Create directories if they don't exist
os.makedirs(template_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

print(f"[STARTUP] Current directory: {current_dir}")
print(f"[STARTUP] Template directory: {template_dir}")
print(f"[STARTUP] Static directory: {static_dir}")
print(f"[STARTUP] Template exists: {os.path.exists(os.path.join(template_dir, 'index.html'))}")
print(f"[STARTUP] Static CSS exists: {os.path.exists(os.path.join(static_dir, 'styles.css'))}")

# Mount static files and templates with correct paths
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=template_dir)


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    method: str


class SentimentEditRequest(BaseModel):
    original_sentiment: str
    new_sentiment: str
    confidence: float
    text: str
    reason: str = ""


def get_base_dir():
    """Return correct base directory for models, handling exe (_MEIPASS) case."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running inside a PyInstaller bundle
        return sys._MEIPASS
    else:
        # Running normally (source code)
        return os.path.dirname(os.path.abspath(__file__))


def load_models():
    """Load the trained models"""
    base_dir = get_base_dir()
    model_path = os.path.join(base_dir, "models", "sentiment_model_5000.pkl")
    vectorizer_path = os.path.join(base_dir, "models", "tfidf_vectorizer_5000.pkl")
    encoder_path = os.path.join(base_dir, "models", "label_encoder_5000.pkl")

    if not all(os.path.exists(p) for p in [model_path, vectorizer_path, encoder_path]):
        missing = [p for p in [model_path, vectorizer_path, encoder_path] if not os.path.exists(p)]
        raise FileNotFoundError(f"Model files missing: {missing}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)

    return model, vectorizer, encoder


try:
    model, vectorizer, encoder = load_models()
    print("[STARTUP] Models loaded successfully")
except Exception as e:
    print(f"[STARTUP] Error loading models: {str(e)}")
    model = None
    vectorizer = None
    encoder = None


def predict_sentiment_enhanced(text: str) -> SentimentResponse:
    """Enhanced sentiment prediction combining model and keywords"""
    # Check if models need reloading (only if scheduler is enabled)
    if scheduler_enabled:
        reload_models_if_needed()

    if not all([model, vectorizer, encoder]):
        raise ValueError("Models not loaded properly")

    # Get model prediction
    cleaned_text = sentiment_analyzer.preprocess_text(text)
    X = vectorizer.transform([cleaned_text])
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    model_sentiment = encoder.inverse_transform([pred])[0]
    model_confidence = np.max(proba)

    # Combine with keyword analysis
    final_sentiment, final_confidence = sentiment_analyzer.combine_model_and_keywords(
        text, model_sentiment, model_confidence
    )

    # Determine method used
    keyword_sentiment, keyword_confidence = sentiment_analyzer.analyze_keywords_with_context(text)
    mixed_sentiment, _ = sentiment_analyzer.analyze_mixed_sentiment(text)

    if mixed_sentiment:
        method = "mixed"
    elif keyword_sentiment and final_sentiment == keyword_sentiment and final_sentiment != model_sentiment:
        method = "keywords"
    elif final_sentiment == model_sentiment and keyword_sentiment and final_sentiment == keyword_sentiment:
        method = "combined"
    else:
        method = "model"

    return SentimentResponse(
        sentiment=final_sentiment,
        confidence=final_confidence,
        method=method
    )


def get_sentiment_emoji(sentiment):
    emoji_map = {
        'positive': '😊',
        'negative': '😞',
        'neutral': '😐'
    }
    return emoji_map.get(sentiment.lower(), '😐')


def get_confidence_color(confidence):
    if confidence >= 0.8:
        return 'success'
    elif confidence >= 0.6:
        return 'warning'
    else:
        return 'danger'


def calculate_statistics(df):
    """Calculate all statistics for the dashboard"""
    companies = sorted(df['Company Name'].unique().tolist())
    company_stats = df.groupby('Company Name')['Sentiment'].value_counts().unstack(fill_value=0)

    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment not in company_stats.columns:
            company_stats[sentiment] = 0

    company_stats['Total'] = company_stats.sum(axis=1)

    overall_stats = {
        'total_pos': int(df[df['Sentiment'] == 'positive'].shape[0]),
        'total_neg': int(df[df['Sentiment'] == 'negative'].shape[0]),
        'total_neu': int(df[df['Sentiment'] == 'neutral'].shape[0]),
        'total_records': len(df)
    }

    # Prepare pie chart data for overall
    sentiment_pie_data = {
        'labels': ['Positive', 'Negative', 'Neutral'],
        'counts': [overall_stats['total_pos'], overall_stats['total_neg'], overall_stats['total_neu']]
    }

    # Prepare pie chart data per company
    company_pie_data = {}
    for company in companies:
        comp_sent = df[df['Company Name'] == company]['Sentiment'].value_counts()
        company_pie_data[company] = {
            'positive': int(comp_sent.get('positive', 0)),
            'negative': int(comp_sent.get('negative', 0)),
            'neutral': int(comp_sent.get('neutral', 0))
        }

    return companies, company_stats, overall_stats, sentiment_pie_data, company_pie_data


def calculate_company_sentiment_analysis(df, company_name):
    """Calculate detailed sentiment analysis for a specific company"""
    company_data = df[df['Company Name'] == company_name].copy()

    if company_data.empty:
        return None

    # Calculate confidence-weighted sentiment scores
    sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
    total_weighted_confidence = 0

    for _, row in company_data.iterrows():
        sentiment = row['Sentiment']
        confidence = row['Confidence']
        sentiment_scores[sentiment] += confidence
        total_weighted_confidence += confidence

    # Normalize scores to percentages
    if total_weighted_confidence > 0:
        for sentiment in sentiment_scores:
            sentiment_scores[sentiment] = (sentiment_scores[sentiment] / total_weighted_confidence) * 100

    # Determine overall sentiment based on highest weighted score
    overall_sentiment = max(sentiment_scores, key=sentiment_scores.get)
    overall_confidence = sentiment_scores[overall_sentiment]

    # Prepare data for line chart (sentiment distribution over confidence ranges)
    confidence_ranges = ['0-50%', '51-70%', '71-90%', '91-100%']
    sentiment_distribution = {
        'positive': [0, 0, 0, 0],
        'negative': [0, 0, 0, 0],
        'neutral': [0, 0, 0, 0]
    }

    for _, row in company_data.iterrows():
        confidence = row['Confidence'] * 100
        sentiment = row['Sentiment']

        if confidence <= 50:
            sentiment_distribution[sentiment][0] += 1
        elif confidence <= 70:
            sentiment_distribution[sentiment][1] += 1
        elif confidence <= 90:
            sentiment_distribution[sentiment][2] += 1
        else:
            sentiment_distribution[sentiment][3] += 1

    return {
        'company_name': company_name,
        'overall_sentiment': overall_sentiment,
        'overall_confidence': round(overall_confidence, 1),
        'sentiment_scores': sentiment_scores,
        'total_remarks': len(company_data),
        'confidence_ranges': confidence_ranges,
        'sentiment_distribution': sentiment_distribution,
        'raw_data': company_data[['Opportunity Name', 'Remarks', 'Sentiment', 'Confidence']].to_dict('records')
    }


# ====================== HTML TEMPLATE ROUTES ======================

@app.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Main dashboard page"""
    global processed_data
    context = {
        "request": request,
        "processed_data": processed_data,
        "scheduler_enabled": scheduler_enabled,
        "github_actions": os.getenv('GITHUB_ACTIONS') == 'true'
    }

    if processed_data is not None:
        companies, company_stats, overall_stats, sentiment_pie_data, company_pie_data = calculate_statistics(
            processed_data)
        context.update({
            "companies": companies,
            "company_stats": company_stats.to_dict('index'),
            "overall_stats": overall_stats,
            "sentiment_pie_data": sentiment_pie_data,
            "company_pie_data": company_pie_data,
            "data": processed_data.head(10).to_dict('records') if len(processed_data) > 0 else [],
            "total_records": len(processed_data)
        })

    return templates.TemplateResponse("index.html", context)


@app.get("/analyze_page", response_class=HTMLResponse)
async def analyze_page(request: Request):
    """Single text analysis page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "show_single_analysis": True,
        "scheduler_enabled": scheduler_enabled,
        "github_actions": os.getenv('GITHUB_ACTIONS') == 'true'
    })


@app.get("/batch_page", response_class=HTMLResponse)
async def batch_page(request: Request):
    """Batch analysis page"""
    global processed_data
    context = {
        "request": request,
        "show_batch_analysis": True,
        "scheduler_enabled": scheduler_enabled,
        "github_actions": os.getenv('GITHUB_ACTIONS') == 'true'
    }

    if processed_data is not None:
        companies, company_stats, overall_stats, sentiment_pie_data, company_pie_data = calculate_statistics(
            processed_data)
        context.update({
            "companies": companies,
            "company_stats": company_stats.to_dict('index'),
            "overall_stats": overall_stats,
            "sentiment_pie_data": sentiment_pie_data,
            "company_pie_data": company_pie_data,
            "data": processed_data.head(10).to_dict('records') if len(processed_data) > 0 else [],
            "total_records": len(processed_data)
        })

    return templates.TemplateResponse("index.html", context)


@app.get("/training_page", response_class=HTMLResponse)
async def training_page(request: Request):
    """Model training management page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "show_training": True,
        "scheduler_enabled": scheduler_enabled,
        "github_actions": os.getenv('GITHUB_ACTIONS') == 'true'
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_redirect(request: Request):
    """Redirect to main dashboard"""
    return await dashboard_page(request)


# ====================== API ENDPOINTS (JSON RESPONSES) ======================

@app.get("/api/status")
async def api_status():
    """API status endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Sentiment Analysis API with Auto-Retraining",
            "status": "running",
            "environment": {
                "github_actions": os.getenv('GITHUB_ACTIONS') == 'true',
                "scheduler_enabled": scheduler_enabled
            },
            "scheduler_running": scheduler.running if scheduler_enabled else False,
            "version": "1.0",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.post("/analyze")
async def analyze_text_api(text: str = Form(...)):
    """Single text analysis API endpoint"""
    try:
        # Validate input text
        validation = validate_english_text(text)
        if not validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "message": validation["message"],
                    "error": "Validation failed"
                }
            )

        result = predict_sentiment_enhanced(text)

        # Convert SentimentResponse object to dictionary
        result_dict = {
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "method": result.method
        }

        # Generate session ID for single analysis
        session_id = str(uuid.uuid4())

        # Return JSON response
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": result_dict,
                "input_text": text,
                "session_id": session_id,
                "message": "Analysis completed successfully",
                "validation": validation
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error analyzing text: {str(e)}",
                "error": str(e)
            }
        )


@app.post("/analyze_single", response_class=HTMLResponse)
async def analyze_single_text(request: Request, text: str = Form(...)):
    """Single text analysis with HTML response"""
    try:
        # Validate input text
        validation = validate_english_text(text)
        if not validation["valid"]:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "show_single_analysis": True,
                "error": validation["message"],
                "scheduler_enabled": scheduler_enabled,
                "github_actions": os.getenv('GITHUB_ACTIONS') == 'true'
            })

        result = predict_sentiment_enhanced(text)
        session_id = str(uuid.uuid4())

        return templates.TemplateResponse("index.html", {
            "request": request,
            "show_single_analysis": True,
            "single_result": result,
            "input_text": text,
            "session_id": session_id,
            "success_message": "Analysis completed successfully!",
            "scheduler_enabled": scheduler_enabled,
            "github_actions": os.getenv('GITHUB_ACTIONS') == 'true'
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "show_single_analysis": True,
            "error": f"Error analyzing text: {str(e)}",
            "scheduler_enabled": scheduler_enabled,
            "github_actions": os.getenv('GITHUB_ACTIONS') == 'true'
        })


@app.post("/batch_analyze")
async def batch_analyze_api(file: UploadFile = File(...)):
    """Batch analysis API endpoint"""
    global processed_data
    try:
        if not all([model, vectorizer, encoder]):
            raise ValueError("Models not loaded properly")

        # Read file
        df = pd.read_excel(file.file) if file.filename.endswith('.xlsx') else pd.read_csv(file.file)

        required_cols = ['Company Name', 'Opportunity Name', 'Remarks']
        if not all(col in df.columns for col in required_cols):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": f"File must contain these columns: {', '.join(required_cols)}"
                }
            )

        # Enhanced sentiment analysis for each row
        sentiments = []
        confidences = []
        methods = []

        for _, row in df.iterrows():
            result = predict_sentiment_enhanced(row['Remarks'])
            sentiments.append(result.sentiment)
            confidences.append(result.confidence)
            methods.append(result.method)

        df['Sentiment'] = sentiments
        df['Confidence'] = confidences
        df['Method'] = methods
        df['Emoji'] = df['Sentiment'].apply(get_sentiment_emoji)
        df['Confidence_Color'] = df['Confidence'].apply(get_confidence_color)

        # Store processed data globally
        processed_data = df

        # Calculate all statistics
        companies, company_stats, overall_stats, sentiment_pie_data, company_pie_data = calculate_statistics(df)

        # Convert DataFrames to JSON serializable formats
        company_stats_dict = company_stats.to_dict('index') if hasattr(company_stats, 'to_dict') else company_stats
        sentiment_pie_data_dict = sentiment_pie_data.to_dict() if hasattr(sentiment_pie_data,
                                                                          'to_dict') else sentiment_pie_data
        company_pie_data_dict = company_pie_data.to_dict() if hasattr(company_pie_data, 'to_dict') else company_pie_data

        # Return JSON response
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Batch analysis completed successfully",
                "data": {
                    "total_records": len(df),
                    "sample_data": df.head(10).to_dict('records'),
                    "overall_stats": overall_stats,
                    "company_stats": company_stats_dict,
                    "sentiment_distribution": sentiment_pie_data_dict,
                    "company_distribution": company_pie_data_dict,
                    "companies": companies
                },
                "metadata": {
                    "file_name": file.filename,
                    "file_size": len(df),
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error processing file: {str(e)}",
                "error": str(e)
            }
        )


@app.post("/batch_upload", response_class=HTMLResponse)
async def batch_upload(request: Request, file: UploadFile = File(...)):
    """Batch file upload with HTML response"""
    global processed_data
    try:
        if not all([model, vectorizer, encoder]):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "show_batch_analysis": True,
                "error": "Models not loaded properly",
                "scheduler_enabled": scheduler_enabled,
                "github_actions": os.getenv('GITHUB_ACTIONS') == 'true'
            })

        # Read file
        df = pd.read_excel(file.file) if file.filename.endswith('.xlsx') else pd.read_csv(file.file)

        required_cols = ['Company Name', 'Opportunity Name', 'Remarks']
        if not all(col in df.columns for col in required_cols):
            return templates.TemplateResponse("index.html", {
                "request": request,
                "show_batch_analysis": True,
                "error": f"File must contain these columns: {', '.join(required_cols)}",
                "scheduler_enabled": scheduler_enabled,
                "github_actions": os.getenv('GITHUB_ACTIONS') == 'true'
            })

        # Enhanced sentiment analysis for each row
        sentiments = []
        confidences = []
        methods = []

        for _, row in df.iterrows():
            result = predict_sentiment_enhanced(row['Remarks'])
            sentiments.append(result.sentiment)
            confidences.append(result.confidence)
            methods.append(result.method)

        df['Sentiment'] = sentiments
        df['Confidence'] = confidences
        df['Method'] = methods
        df['Emoji'] = df['Sentiment'].apply(get_sentiment_emoji)
        df['Confidence_Color'] = df['Confidence'].apply(get_confidence_color)

        # Store processed data globally
        processed_data = df

        # Calculate all statistics
        companies, company_stats, overall_stats, sentiment_pie_data, company_pie_data = calculate_statistics(df)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "show_batch_analysis": True,
            "companies": companies,
            "company_stats": company_stats.to_dict('index'),
            "overall_stats": overall_stats,
            "sentiment_pie_data": sentiment_pie_data,
            "company_pie_data": company_pie_data,
            "data": df.head(10).to_dict('records') if len(df) > 0 else [],
            "total_records": len(df),
            "success_message": f"Successfully analyzed {len(df)} records from {file.filename}",
            "scheduler_enabled": scheduler_enabled,
            "github_actions": os.getenv('GITHUB_ACTIONS') == 'true'
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "show_batch_analysis": True,
            "error": f"Error processing file: {str(e)}",
            "scheduler_enabled": scheduler_enabled,
            "github_actions": os.getenv('GITHUB_ACTIONS') == 'true'
        })


@app.post("/add_remark")
async def add_remark(company: str = Form(...), opportunity: str = Form(...), remark: str = Form(...)):
    global processed_data
    if processed_data is None:
        return JSONResponse(status_code=400,
                            content={"success": False, "message": "No data available. Please upload a file first."})

    try:
        # Create new row as dataframe
        new_row = pd.DataFrame({
            'Company Name': [company],
            'Opportunity Name': [opportunity],
            'Remarks': [remark]
        })

        # Enhanced sentiment analysis
        result = predict_sentiment_enhanced(remark)
        new_row['Sentiment'] = result.sentiment
        new_row['Confidence'] = result.confidence
        new_row['Method'] = result.method
        new_row['Emoji'] = get_sentiment_emoji(result.sentiment)
        new_row['Confidence_Color'] = get_confidence_color(result.confidence)

        # Append new row to processed data
        processed_data = pd.concat([processed_data, new_row], ignore_index=True)

        # Calculate updated statistics
        companies, company_stats, overall_stats, sentiment_pie_data, company_pie_data = calculate_statistics(
            processed_data)

        # Get current page data (first page by default after adding)
        page = 1
        per_page = 10
        total_pages = math.ceil(len(processed_data) / per_page)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        current_page_data = processed_data.iloc[start_idx:end_idx].to_dict('records')

        # Return comprehensive response with ALL updated data
        return JSONResponse(status_code=200, content={
            "success": True,
            "message": "Remark added successfully!",
            "new_remark": {
                'Company Name': company,
                'Opportunity Name': opportunity,
                'Remarks': remark,
                'Sentiment': result.sentiment,
                'Confidence': result.confidence,
                'Method': result.method,
                'Emoji': get_sentiment_emoji(result.sentiment),
                'Confidence_Color': get_confidence_color(result.confidence)
            },
            "updated_stats": {
                "companies": companies,
                "company_stats": company_stats.to_dict('index'),
                "overall_stats": overall_stats,
                "sentiment_pie_data": sentiment_pie_data,
                "company_pie_data": company_pie_data,
                "total_records": len(processed_data)
            },
            "updated_table": {
                "data": current_page_data,
                "total_records": len(processed_data),
                "page": page,
                "total_pages": total_pages
            }
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Error adding remark: {str(e)}"})


@app.get("/get_current_data")
async def get_current_data(
        page: int = Query(1),
        company: Optional[str] = Query(""),
        search: Optional[str] = Query(""),
        sentiment: Optional[str] = Query("")
):
    global processed_data
    if processed_data is None:
        return JSONResponse(status_code=400, content={"success": False, "message": "No data available"})

    try:
        df = processed_data.copy()
        original_df = processed_data.copy()

        # Apply filters
        if company and company != "all":
            df = df[df['Company Name'] == company]

        if search:
            search_mask = (
                    df['Company Name'].str.contains(search, case=False, na=False) |
                    df['Opportunity Name'].str.contains(search, case=False, na=False) |
                    df['Remarks'].str.contains(search, case=False, na=False)
            )
            df = df[search_mask]

        if sentiment and sentiment != "all":
            df = df[df['Sentiment'] == sentiment]

        # Pagination
        per_page = 10
        total_pages = math.ceil(len(df) / per_page) if len(df) > 0 else 1
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_data = df.iloc[start_idx:end_idx].to_dict('records')

        # Calculate statistics for all data
        companies, company_stats, overall_stats, sentiment_pie_data, company_pie_data = calculate_statistics(
            original_df)

        return JSONResponse(status_code=200, content={
            "success": True,
            "data": paginated_data,
            "total_records": len(df),
            "page": page,
            "total_pages": total_pages,
            "companies": companies,
            "company_stats": company_stats.to_dict('index'),
            "overall_stats": overall_stats,
            "sentiment_pie_data": sentiment_pie_data,
            "company_pie_data": company_pie_data
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Error retrieving data: {str(e)}"})


@app.get("/export_data")
async def export_data(company: Optional[str] = Query(""), sentiment: Optional[str] = Query("")):
    global processed_data
    if processed_data is None:
        return JSONResponse(
            status_code=400,
            content={"message": "No data available"}
        )

    try:
        df = processed_data.copy()

        # Apply filters
        if company and company != "all":
            df = df[df['Company Name'] == company]

        if sentiment and sentiment != "all":
            df = df[df['Sentiment'] == sentiment]

        # Export to CSV with method information
        export_df = df[['Company Name', 'Opportunity Name', 'Remarks', 'Sentiment', 'Confidence', 'Method']].copy()
        csv_data = export_df.to_csv(index=False)

        return JSONResponse({
            "data": csv_data,
            "filename": f"sentiment_analysis_{company}_{sentiment}.csv"
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error exporting data: {str(e)}"}
        )


@app.get("/company_sentiment_analysis/{company_name}")
async def get_company_sentiment_analysis(company_name: str):
    global processed_data
    if processed_data is None:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "No data available"}
        )

    try:
        analysis = calculate_company_sentiment_analysis(processed_data, company_name)
        if analysis is None:
            return JSONResponse(
                status_code=404,
                content={"success": False, "message": "Company not found"}
            )

        return JSONResponse(
            status_code=200,
            content={"success": True, "data": analysis}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error analyzing company sentiment: {str(e)}"}
        )


# ====================== SINGLE ANALYSIS MANAGEMENT ENDPOINTS ======================

@app.post("/edit_single_sentiment")
async def edit_single_sentiment(
        session_id: str = Form(...),
        original_sentiment: str = Form(...),
        new_sentiment: str = Form(...),
        confidence: float = Form(...),
        text: str = Form(...),
        reason: str = Form("")
):
    """Edit sentiment for single analysis and log the change"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "session_id": session_id,
            "text": text,
            "original_sentiment": original_sentiment,
            "new_sentiment": new_sentiment,
            "confidence": confidence,
            "reason": reason,
            "edited_by": "user",
            "action": "sentiment_edit"
        }

        single_analysis_log.append(log_entry)
        if session_id not in analysis_history:
            analysis_history[session_id] = []
        analysis_history[session_id].append(log_entry)

        return JSONResponse({
            "success": True,
            "message": "Sentiment updated successfully!",
            "updated_sentiment": new_sentiment,
            "confidence": confidence,
            "timestamp": timestamp
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error updating sentiment: {str(e)}"}
        )


@app.post("/save_single_analysis")
async def save_single_analysis(
        session_id: str = Form(...),
        text: str = Form(...),
        sentiment: str = Form(...),
        confidence: float = Form(...),
        method: str = Form(...),
        notes: str = Form(""),
        tags: str = Form("")
):
    """Save single analysis result with notes and tags"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_entry = {
            "timestamp": timestamp,
            "session_id": session_id,
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "method": method,
            "notes": notes,
            "tags": tags.split(",") if tags else [],
            "action": "analysis_save"
        }

        single_analysis_log.append(save_entry)
        if session_id not in analysis_history:
            analysis_history[session_id] = []
        analysis_history[session_id].append(save_entry)

        return JSONResponse({
            "success": True,
            "message": "Analysis saved successfully!",
            "saved_data": save_entry
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error saving analysis: {str(e)}"}
        )


@app.get("/download_single_analysis_log")
async def download_single_analysis_log(
        format: str = Query("excel", regex="^(excel|csv)$"),
        session_id: Optional[str] = Query(None)
):
    """Download analysis log as Excel or CSV"""
    try:
        if session_id:
            data = analysis_history.get(session_id, [])
            filename_suffix = f"_session_{session_id}"
        else:
            data = single_analysis_log
            filename_suffix = "_all_sessions"

        if not data:
            return JSONResponse(
                status_code=404,
                content={"message": "No analysis data found"}
            )

        df = pd.DataFrame(data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "excel":
            filename = f"single_analysis_log{filename_suffix}_{timestamp}.xlsx"
            filepath = f"temp_{filename}"
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Analysis_Log', index=False)
                if len(df) > 0:
                    summary_data = {
                        'Total Analyses': [len(df)],
                        'Positive Sentiments': [len(df[df.get('sentiment', '') == 'positive'])],
                        'Negative Sentiments': [len(df[df.get('sentiment', '') == 'negative'])],
                        'Neutral Sentiments': [len(df[df.get('sentiment', '') == 'neutral'])],
                        'Total Edits': [len(df[df.get('action', '') == 'sentiment_edit'])]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)

            return FileResponse(
                filepath,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                filename=filename
            )

        else:  # CSV format
            filename = f"single_analysis_log{filename_suffix}_{timestamp}.csv"
            csv_data = df.to_csv(index=False)
            return JSONResponse({
                "data": csv_data,
                "filename": filename,
                "content_type": "text/csv"
            })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error generating download: {str(e)}"}
        )


@app.get("/get_analysis_history/{session_id}")
async def get_analysis_history(session_id: str):
    """Get analysis history for a specific session"""
    try:
        history = analysis_history.get(session_id, [])
        return JSONResponse({
            "success": True,
            "history": history,
            "total_entries": len(history)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error retrieving history: {str(e)}"}
        )


@app.delete("/clear_analysis_log")
async def clear_analysis_log(session_id: Optional[str] = Query(None)):
    """Clear analysis log (all or specific session)"""
    try:
        global single_analysis_log, analysis_history

        if session_id:
            if session_id in analysis_history:
                del analysis_history[session_id]
                single_analysis_log = [entry for entry in single_analysis_log
                                       if entry.get("session_id") != session_id]
                message = f"Session {session_id} cleared successfully"
            else:
                return JSONResponse(
                    status_code=404,
                    content={"success": False, "message": "Session not found"}
                )
        else:
            single_analysis_log = []
            analysis_history = {}
            message = "All analysis logs cleared successfully"

        return JSONResponse({
            "success": True,
            "message": message
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error clearing logs: {str(e)}"}
        )


# ====================== SCHEDULER MANAGEMENT ENDPOINTS ======================

@app.post("/manual_retrain")
async def manual_retrain():
    """Manually trigger model retraining"""
    try:
        print("[MANUAL] Starting manual model retraining...")

        training_result = model_trainer.train_model(generate_report=True)

        # Reload models
        global model, vectorizer, encoder
        model, vectorizer, encoder = load_models()

        # Log the training event
        log_training_event(training_result, "manual")

        print(f"[MANUAL] Manual retraining completed successfully!")

        return JSONResponse({
            "success": True,
            "message": "Model retrained successfully",
            "result": training_result,
            "github_actions": os.getenv('GITHUB_ACTIONS') == 'true'
        })

    except Exception as e:
        log_training_event({"error": str(e)}, "manual_failed")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error retraining model: {str(e)}"}
        )


@app.get("/training_status")
async def get_training_status():
    """Get the current training/scheduler status"""
    try:
        # Get next scheduled job
        next_run = None
        job_details = []

        if scheduler_enabled:
            jobs = scheduler.get_jobs()
            for job in jobs:
                job_info = {
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger)
                }
                job_details.append(job_info)
                if job.id == 'model_retraining':
                    next_run = job_info["next_run_time"]

        # Load training metadata
        last_training = model_trainer.get_training_status()

        # Load training logs
        training_logs = []
        log_file = os.path.join(model_trainer.models_dir, "training_log.json")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                training_logs = json.load(f)[-10:]  # Last 10 logs

        # Validate models
        models_valid = model_trainer.validate_models()

        return JSONResponse({
            "success": True,
            "environment": {
                "github_actions": os.getenv('GITHUB_ACTIONS') == 'true',
                "scheduler_enabled": scheduler_enabled
            },
            "scheduler_running": scheduler.running if scheduler_enabled else False,
            "next_scheduled_training": next_run,
            "last_training": last_training,
            "jobs_count": len(job_details),
            "job_details": job_details,
            "models_valid": models_valid,
            "recent_training_logs": training_logs
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error getting training status: {str(e)}"}
        )


@app.post("/upload_training_data")
async def upload_training_data(file: UploadFile = File(...)):
    """Upload new training data for the next scheduled training"""
    try:
        # Save uploaded file to data directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if file.filename.endswith('.xlsx'):
            file_path = os.path.join(model_trainer.data_dir, f"training_data_{timestamp}.xlsx")
            df = pd.read_excel(file.file)
        else:
            file_path = os.path.join(model_trainer.data_dir, f"training_data_{timestamp}.csv")
            df = pd.read_csv(file.file)

        required_cols = ['Remarks', 'Sentiment']
        if not all(col in df.columns for col in required_cols):
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": f"File must contain columns: {', '.join(required_cols)}"}
            )

        # Save the file
        if file.filename.endswith('.xlsx'):
            df.to_excel(file_path, index=False)
        else:
            df.to_csv(file_path, index=False)

        print(f"[DATA UPLOAD] New training data uploaded: {file_path}")

        return JSONResponse({
            "success": True,
            "message": f"Training data uploaded successfully. {len(df)} records added.",
            "file_path": file_path,
            "records_count": len(df),
            "github_actions": os.getenv('GITHUB_ACTIONS') == 'true'
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error uploading training data: {str(e)}"}
        )


# ====================== SCHEDULER CONTROL ENDPOINTS (Only if scheduler enabled) ======================

@app.post("/pause_scheduler")
async def pause_scheduler():
    """Pause the scheduler"""
    try:
        if not scheduler_enabled:
            return JSONResponse({
                "success": False,
                "message": "Scheduler is not enabled - running in GitHub Actions mode"
            })

        if scheduler.running:
            scheduler.pause()
            print("[SCHEDULER] Scheduler paused by user")
            return JSONResponse({
                "success": True,
                "message": "Scheduler paused successfully",
                "scheduler_running": scheduler.running
            })
        else:
            return JSONResponse({
                "success": False,
                "message": "Scheduler is already paused"
            })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error pausing scheduler: {str(e)}"}
        )


@app.post("/resume_scheduler")
async def resume_scheduler():
    """Resume the scheduler"""
    try:
        if not scheduler_enabled:
            return JSONResponse({
                "success": False,
                "message": "Scheduler is not enabled - running in GitHub Actions mode"
            })

        if not scheduler.running:
            scheduler.resume()
            print("[SCHEDULER] Scheduler resumed by user")
            return JSONResponse({
                "success": True,
                "message": "Scheduler resumed successfully",
                "scheduler_running": scheduler.running
            })
        else:
            return JSONResponse({
                "success": False,
                "message": "Scheduler is already running"
            })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error resuming scheduler: {str(e)}"}
        )


@app.post("/update_schedule")
async def update_schedule(interval_days: int = Form(...)):
    """Update the training schedule interval"""
    try:
        if not scheduler_enabled:
            return JSONResponse({
                "success": False,
                "message": "Scheduler is not enabled - running in GitHub Actions mode"
            })

        # Remove existing job
        scheduler.remove_job('model_retraining')

        # Add new job with updated interval
        scheduler.add_job(
            func=scheduled_model_training,
            trigger=IntervalTrigger(days=interval_days),
            id='model_retraining',
            name=f'Retrain sentiment model every {interval_days} days',
            replace_existing=True,
            max_instances=1
        )

        print(f"[SCHEDULER] Schedule updated to every {interval_days} days")

        return JSONResponse({
            "success": True,
            "message": f"Schedule updated to every {interval_days} days",
            "interval_days": interval_days,
            "next_run": scheduler.get_job('model_retraining').next_run_time.isoformat()
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error updating schedule: {str(e)}"}
        )


@app.get("/backup_models")
async def backup_models():
    """Create a manual backup of current models"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create manual backup
        model_files = [
            "sentiment_model_5000.pkl",
            "tfidf_vectorizer_5000.pkl",
            "label_encoder_5000.pkl"
        ]

        backup_count = 0
        for filename in model_files:
            filepath = os.path.join(model_trainer.models_dir, filename)
            if os.path.exists(filepath):
                backup_name = f"{filename.split('.')[0]}_manual_backup_{timestamp}.pkl"
                backup_path = os.path.join(model_trainer.models_dir, backup_name)
                shutil.copy2(filepath, backup_path)
                backup_count += 1

        return JSONResponse({
            "success": True,
            "message": f"Manual backup created successfully. {backup_count} files backed up.",
            "backup_timestamp": timestamp
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error creating backup: {str(e)}"}
        )


@app.get("/scheduler_logs")
async def get_scheduler_logs(limit: int = Query(20)):
    """Get scheduler training logs"""
    try:
        log_file = os.path.join(model_trainer.models_dir, "training_log.json")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)[-limit:]
        else:
            logs = []

        return JSONResponse({
            "success": True,
            "logs": logs,
            "total_logs": len(logs)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error retrieving logs: {str(e)}"}
        )


@app.post("/test_scheduler")
async def test_scheduler():
    """Test scheduler functionality (runs training immediately for testing)"""
    try:
        if not scheduler_enabled:
            return JSONResponse({
                "success": False,
                "message": "Scheduler is not enabled - running in GitHub Actions mode"
            })

        print("[TEST] Testing scheduler functionality...")

        # Add a one-time job that runs in 5 seconds
        from datetime import datetime, timedelta
        run_time = datetime.now() + timedelta(seconds=5)

        scheduler.add_job(
            func=scheduled_model_training,
            trigger='date',
            run_date=run_time,
            id='test_training',
            name='Test model training',
            replace_existing=True
        )

        return JSONResponse({
            "success": True,
            "message": "Test training scheduled to run in 5 seconds",
            "run_time": run_time.isoformat()
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Error scheduling test: {str(e)}"}
        )


# ====================== CLI TRAINING FLAG SUPPORT ======================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        # Training mode for GitHub Actions
        print("[CLI] Starting training mode for GitHub Actions...")
        try:
            training_result = model_trainer.train_model(generate_report=True)
            print(f"[CLI] Training completed successfully!")
            print(f"[CLI] Accuracy: {training_result['accuracy']:.4f}")
            print(f"[CLI] F1 Score: {training_result['cv_f1_score']:.4f}")
            print(f"[CLI] Model saved to: {model_trainer.models_dir}")
            sys.exit(0)
        except Exception as e:
            print(f"[CLI] Training failed: {str(e)}")
            sys.exit(1)
    else:
        # Normal FastAPI server mode
        import uvicorn

        print("[STARTUP] Starting FastAPI application with HTML UI...")
        print(f"[STARTUP] GitHub Actions mode: {os.getenv('GITHUB_ACTIONS') == 'true'}")
        print(f"[STARTUP] Scheduler enabled: {scheduler_enabled}")
        print(f"[STARTUP] UI available at: http://127.0.0.1:8001")
        uvicorn.run(app, host="127.0.0.1", port=8001)
