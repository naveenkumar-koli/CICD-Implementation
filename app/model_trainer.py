import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for scheduled tasks
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec
import pickle
import warnings
import re
from collections import Counter
from textblob import TextBlob
import os
import json
from pathlib import Path
from datetime import datetime
import logging

# MLflow + DagsHub tracking
try:
    import mlflow
    import mlflow.sklearn
    import dagshub
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configure logging for scheduled tasks
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)


class ModelTrainer:
    def __init__(self):
        # Set up paths relative to the app directory
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, "models")
        self.data_dir = os.path.join(self.base_dir, "data")
        self.reports_dir = os.path.join(self.base_dir, "reports")

        # Ensure directories exist
        Path(self.models_dir).mkdir(exist_ok=True)
        Path(self.data_dir).mkdir(exist_ok=True)
        Path(self.reports_dir).mkdir(exist_ok=True)

        # Initialize components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Set style for plots
        plt.style.use('default')
        sns.set_palette("husl")

    def load_training_data(self):
        """Load all available training data from the data directory"""
        logger.info("Loading training data...")
        data_files = []

        # Look for CSV and Excel files in the data directory
        for file_path in Path(self.data_dir).glob("*.csv"):
            data_files.append(file_path)
        for file_path in Path(self.data_dir).glob("*.xlsx"):
            data_files.append(file_path)

        if not data_files:
            # If no files in data directory, try the hardcoded path as fallback
            fallback_path = r'C:\Users\Admin\PycharmProjects\Sentiment_with_scheduler\data\Sales Final Data (5000).csv'
            if os.path.exists(fallback_path):
                logger.info(f"Using fallback data file: {fallback_path}")
                return pd.read_csv(fallback_path, encoding_errors='ignore')
            else:
                raise FileNotFoundError(f"No training data found in {self.data_dir}")

        # Combine all data files
        all_data = []
        for file_path in data_files:
            try:
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path, encoding_errors='ignore')
                else:
                    df = pd.read_excel(file_path)

                # Ensure required columns exist
                if 'Remarks' in df.columns and 'Sentiment' in df.columns:
                    all_data.append(df[['Remarks', 'Sentiment']])
                    logger.info(f"Loaded {len(df)} records from {file_path.name}")
                else:
                    logger.warning(f"Skipping {file_path.name} - missing required columns")

            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {str(e)}")

        if not all_data:
            raise ValueError("No valid training data found")

        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined dataset shape: {combined_df.shape}")

        return combined_df

    def preprocess_text(self, text):
        """Enhanced text preprocessing function"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.!\?]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)

        return ' '.join(processed_tokens)

    def clean_and_prepare_data(self, data):
        """Clean and prepare the dataset"""
        logger.info("Cleaning and preparing data...")

        # Display basic info
        logger.info(f"Dataset shape: {data.shape}")
        logger.info(f"Columns: {list(data.columns)}")

        # Check for missing values
        missing_data = data.isnull().sum()
        missing_dict = {col: int(val) for col, val in missing_data.items() if val > 0}
        logger.info(f"Missing values: {missing_dict}")

        # Remove rows with missing remarks or sentiment
        data_clean = data.dropna(subset=['Remarks', 'Sentiment']).copy()
        logger.info(f"Cleaned dataset shape: {data_clean.shape}")

        # Map variations to standard labels
        sentiment_mapping = {
            'positive': 'positive',
            'negative': 'negative',
            'neutral': 'neutral',
            'pos': 'positive',
            'neg': 'negative'
        }

        data_clean['Sentiment'] = data_clean['Sentiment'].str.lower().map(sentiment_mapping)
        data_clean = data_clean.dropna(subset=['Sentiment'])
        logger.info(f"Final dataset shape: {data_clean.shape}")

        # Add text analysis columns
        data_clean['remark_length'] = data_clean['Remarks'].str.len()
        data_clean['word_count'] = data_clean['Remarks'].str.split().str.len()

        # Apply preprocessing
        logger.info("Preprocessing text data...")
        data_clean['processed_remarks'] = data_clean['Remarks'].apply(self.preprocess_text)

        # Remove empty processed text
        data_clean = data_clean[data_clean['processed_remarks'].str.len() > 0]

        return data_clean

    def generate_analysis_report(self, data_clean, save_plots=True):
        """Generate analysis report with visualizations"""
        logger.info("Generating analysis report...")

        try:
            if save_plots:
                # Sentiment distribution
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                sentiment_counts = data_clean['Sentiment'].value_counts()
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                sentiment_counts.plot(kind='bar', color=colors)
                plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
                plt.xlabel('Sentiment')
                plt.ylabel('Count')
                plt.xticks(rotation=45)

                # Add percentage labels on bars
                total_records = len(data_clean)
                for i, v in enumerate(sentiment_counts.values):
                    percentage = (v / total_records) * 100
                    plt.text(i, v + 10, f'{percentage:.1f}%', ha='center', fontweight='bold')

                plt.subplot(1, 3, 2)
                sns.boxplot(data=data_clean, x='Sentiment', y='word_count')
                plt.title('Word Count by Sentiment', fontweight='bold')

                plt.subplot(1, 3, 3)
                plt.hist(data_clean['word_count'], bins=30, alpha=0.7, color='skyblue')
                plt.title('Distribution of Word Count', fontweight='bold')
                plt.xlabel('Word Count')
                plt.ylabel('Frequency')

                plt.tight_layout()
                report_path = os.path.join(self.reports_dir,
                                           f'sentiment_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                plt.savefig(report_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Analysis report saved: {report_path}")
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")

        # Print statistics - FIXED: Use proper len() on DataFrames
        stats = {}
        for sentiment in data_clean['Sentiment'].unique():
            subset = data_clean[data_clean['Sentiment'] == sentiment]
            subset_length = subset.shape[0]  # Use shape[0] instead of len()
            stats[sentiment] = {
                'count': int(subset_length),
                'avg_word_count': float(subset['word_count'].mean()),
                'avg_char_length': float(subset['remark_length'].mean())
            }
            logger.info(f"{sentiment.upper()} Sentiment: {subset_length} records, "
                        f"avg words: {subset['word_count'].mean():.1f}, "
                        f"avg chars: {subset['remark_length'].mean():.1f}")

        return stats

    def train_model(self, generate_report=False):
        """Train the sentiment analysis model with MLflow + DagsHub tracking"""
        # ── Setup MLflow tracking via DagsHub ──────────────────────────────
        if MLFLOW_AVAILABLE:
            dagshub_user  = os.getenv('DAGSHUB_USER', '')
            dagshub_token = os.getenv('DAGSHUB_TOKEN', '')
            dagshub_repo  = os.getenv('DAGSHUB_REPO', 'CICD-Implementation')
            if dagshub_user and dagshub_token:
                tracking_uri = (
                    f"https://dagshub.com/{dagshub_user}/{dagshub_repo}.mlflow"
                )
                mlflow.set_tracking_uri(tracking_uri)
                os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_user
                os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
                logger.info(f"MLflow tracking → {tracking_uri}")
            mlflow.set_experiment("sales-sentiment-analysis")

        try:
            logger.info("Starting model training...")

            # Load and clean data
            data = self.load_training_data()
            data_clean = self.clean_and_prepare_data(data)

            # Generate analysis report if requested
            stats = None
            if generate_report:
                stats = self.generate_analysis_report(data_clean)

            # Feature Engineering
            logger.info("Creating TF-IDF features...")
            tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                stop_words='english',
                min_df=2,
                max_df=0.95
            )

            X_tfidf = tfidf_vectorizer.fit_transform(data_clean['processed_remarks'])
            logger.info(f"TF-IDF shape: {X_tfidf.shape}")

            # Label encoding
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(data_clean['Sentiment'])
            logger.info(f"Classes: {label_encoder.classes_}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf, y, test_size=0.2, random_state=42, stratify=y
            )

            logger.info(f"Training set size: {X_train.shape}")
            logger.info(f"Test set size: {X_test.shape}")

            # Hyperparameter tuning
            logger.info("Performing hyperparameter tuning...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000, 2000]
            }

            lr_model = LogisticRegression(random_state=42, class_weight='balanced')
            grid_search = GridSearchCV(
                lr_model, param_grid, cv=5,
                scoring='f1_weighted', n_jobs=-1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

            # Model evaluation
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Test Accuracy: {accuracy:.4f}")

            # Cross-validation scores
            cv_scores = cross_val_score(best_model, X_tfidf, y, cv=5, scoring='f1_weighted')
            logger.info(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

            # Classification report
            class_names = label_encoder.classes_
            report = classification_report(y_test, y_pred, target_names=class_names)
            logger.info(f"Classification Report:\n{report}")

            # Train Word2Vec if enough data
            word2vec_model = None
            try:
                sentences = [remark.split() for remark in data_clean['processed_remarks']]
                if len(sentences) > 100:
                    logger.info("Training Word2Vec model...")
                    word2vec_model = Word2Vec(
                        sentences=sentences,
                        vector_size=100,
                        window=5,
                        min_count=2,
                        workers=4,
                        epochs=10
                    )
                    logger.info(f"Word2Vec vocabulary size: {len(word2vec_model.wv.key_to_index)}")
            except Exception as e:
                logger.error(f"Error training Word2Vec: {str(e)}")

            # Train LSA
            lsa_model = None
            lsa_vectorizer = None
            try:
                logger.info("Training LSA model...")
                lsa_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                X_lsa = lsa_vectorizer.fit_transform(data_clean['processed_remarks'])
                n_components = min(50, X_lsa.shape[1] - 1)
                lsa_model = TruncatedSVD(n_components=n_components, random_state=42)
                X_lsa_transformed = lsa_model.fit_transform(X_lsa)
                logger.info(f"LSA components: {n_components}")
                logger.info(f"Explained variance ratio sum: {lsa_model.explained_variance_ratio_.sum():.3f}")
            except Exception as e:
                logger.error(f"Error training LSA: {str(e)}")

            # Save the models
            self.save_models(best_model, tfidf_vectorizer, label_encoder, accuracy,
                             word2vec_model, lsa_model, lsa_vectorizer)

            training_result = {
                'timestamp': datetime.now().isoformat(),
                'accuracy': float(accuracy),
                'cv_f1_score': float(cv_scores.mean()),
                'cv_f1_std': float(cv_scores.std()),
                'best_params': grid_search.best_params_,
                'training_samples': int(X_train.shape[0]),
                'test_samples': int(X_test.shape[0]),
                'total_samples': int(data_clean.shape[0]),
                'classes': [str(cls) for cls in class_names],
                'tfidf_features': int(X_tfidf.shape[1]),
                'data_stats': stats if stats else {}
            }

            metadata_path = os.path.join(self.models_dir, "training_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(training_result, f, indent=2)

            # ── Log everything to MLflow / DagsHub ────────────────────────
            if MLFLOW_AVAILABLE:
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with mlflow.start_run(run_name=run_name):
                    # Parameters
                    mlflow.log_param("max_features", 5000)
                    mlflow.log_param("ngram_range", "(1,3)")
                    mlflow.log_param("model_type", "LogisticRegression")
                    mlflow.log_param("test_size", 0.2)
                    mlflow.log_param("cv_folds", 5)
                    mlflow.log_params({
                        f"best_{k}": v for k, v in grid_search.best_params_.items()
                    })
                    # Metrics
                    mlflow.log_metric("accuracy", float(accuracy))
                    mlflow.log_metric("cv_f1_weighted", float(cv_scores.mean()))
                    mlflow.log_metric("cv_f1_std", float(cv_scores.std()))
                    mlflow.log_metric("training_samples", int(X_train.shape[0]))
                    mlflow.log_metric("total_samples", int(data_clean.shape[0]))
                    # Model artifacts
                    mlflow.sklearn.log_model(best_model, "sentiment_model")
                    mlflow.log_artifact(
                        os.path.join(self.models_dir, "tfidf_vectorizer_5000.pkl"),
                        artifact_path="artifacts"
                    )
                    mlflow.log_artifact(
                        os.path.join(self.models_dir, "label_encoder_5000.pkl"),
                        artifact_path="artifacts"
                    )
                    mlflow.log_artifact(metadata_path, artifact_path="artifacts")
                    logger.info(f"[MLFLOW] Run logged: {run_name}")

            logger.info("Model training completed successfully!")
            return training_result

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def save_models(self, model, vectorizer, label_encoder, accuracy, word2vec_model=None, lsa_model=None,
                    lsa_vectorizer=None):
        """Save trained models to pickle files with backup"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create backup of existing models
        model_files = {
            "sentiment_model_5000.pkl": model,
            "tfidf_vectorizer_5000.pkl": vectorizer,
            "label_encoder_5000.pkl": label_encoder
        }

        # Add optional models
        if word2vec_model is not None:
            model_files["word2vec_sales_5000.model"] = word2vec_model
        if lsa_model is not None:
            model_files["lsa_model_5000.pkl"] = lsa_model
        if lsa_vectorizer is not None:
            model_files["lsa_vectorizer.pkl"] = lsa_vectorizer

        # Create backups and save new models
        for filename, model_obj in model_files.items():
            filepath = os.path.join(self.models_dir, filename)

            # Create backup if file exists
            if os.path.exists(filepath):
                backup_name = f"{filename.split('.')[0]}_backup_{timestamp}.{filename.split('.')[1]}"
                backup_path = os.path.join(self.models_dir, backup_name)
                try:
                    os.rename(filepath, backup_path)
                    logger.info(f"Created backup: {backup_name}")
                except Exception as e:
                    logger.error(f"Error creating backup for {filename}: {str(e)}")

            # Save new model
            try:
                if filename.endswith('.model'):  # Word2Vec model
                    model_obj.save(filepath)
                else:  # Pickle files
                    with open(filepath, 'wb') as f:
                        pickle.dump(model_obj, f)
                logger.info(f"Saved: {filename}")
            except Exception as e:
                logger.error(f"Error saving {filename}: {str(e)}")

        logger.info("All models saved successfully")

    def cleanup_old_backups(self, keep_last_n=5):
        """Clean up old backup files, keeping only the last N backups"""
        try:
            backup_files = {}
            for file in os.listdir(self.models_dir):
                if '_backup_' in file:
                    # Extract base name and timestamp
                    parts = file.split('_backup_')
                    if len(parts) == 2:
                        base_name = parts[0]
                        timestamp_part = parts[1]
                        if base_name not in backup_files:
                            backup_files[base_name] = []
                        backup_files[base_name].append((timestamp_part, file))

            # Clean up each model's backups
            for base_name, files in backup_files.items():
                # Sort by timestamp (newest first)
                files.sort(key=lambda x: x[0], reverse=True)

                # Remove old backups
                if len(files) > keep_last_n:
                    files_to_remove = files[keep_last_n:]
                    for _, filename in files_to_remove:
                        file_path = os.path.join(self.models_dir, filename)
                        try:
                            os.remove(file_path)
                            logger.info(f"Removed old backup: {filename}")
                        except Exception as e:
                            logger.error(f"Error removing {filename}: {str(e)}")

        except Exception as e:
            logger.error(f"Error cleaning up backups: {str(e)}")

    def get_training_status(self):
        """Get the current training status and metadata"""
        try:
            metadata_path = os.path.join(self.models_dir, "training_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata
            else:
                return None
        except Exception as e:
            logger.error(f"Error reading training metadata: {str(e)}")
            return None

    def validate_models(self):
        """Validate that all required models exist and can be loaded"""
        required_files = [
            "sentiment_model_5000.pkl",
            "tfidf_vectorizer_5000.pkl",
            "label_encoder_5000.pkl"
        ]

        missing_files = []
        for filename in required_files:
            filepath = os.path.join(self.models_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)

        if missing_files:
            logger.error(f"Missing model files: {missing_files}")
            return False

        # Try to load the models
        try:
            with open(os.path.join(self.models_dir, "sentiment_model_5000.pkl"), 'rb') as f:
                pickle.load(f)
            with open(os.path.join(self.models_dir, "tfidf_vectorizer_5000.pkl"), 'rb') as f:
                pickle.load(f)
            with open(os.path.join(self.models_dir, "label_encoder_5000.pkl"), 'rb') as f:
                pickle.load(f)
            logger.info("All models validated successfully")
            return True
        except Exception as e:
            logger.error(f"Error validating models: {str(e)}")
            return False


# Convenience functions for backward compatibility
def preprocess_text(text):
    """Backward compatibility function"""
    trainer = ModelTrainer()
    return trainer.preprocess_text(text)


# Main execution for testing
if __name__ == "__main__":
    trainer = ModelTrainer()
    try:
        result = trainer.train_model(generate_report=True)
        print("Training completed successfully!")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"F1 Score: {result['cv_f1_score']:.4f}")
    except Exception as e:
        print(f"Training failed: {str(e)}")
