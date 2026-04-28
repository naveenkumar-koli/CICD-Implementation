import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')


class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Keyword sets for sentiment analysis
        self.positive_keywords = {
            'good', 'best', 'better', 'excellent', 'great', 'amazing', 'fantastic',
            'wonderful', 'perfect', 'outstanding', 'brilliant', 'superb', 'awesome',
            'love', 'like', 'satisfied', 'happy', 'pleased', 'impressed', 'positive',
            'favorable', 'beneficial', 'valuable', 'useful', 'helpful', 'effective',
            'successful', 'promising', 'optimistic', 'confident', 'excited',
            'interested', 'keen', 'enthusiastic', 'recommend', 'approve', 'accept',
            'yes', 'definitely', 'absolutely', 'certainly', 'agree', 'willing', 'won'
        }

        self.negative_keywords = {
            'bad', 'worst', 'worse', 'terrible', 'awful', 'horrible', 'disappointing',
            'not interested', 'no reply', 'no response', 'declined', 'rejected',
            'refused', 'denied', 'cancelled', 'cancel', 'drop', 'withdraw',
            'negative', 'poor', 'unsatisfied', 'unhappy', 'disappointed',
            'frustrated', 'angry', 'annoyed', 'concerned', 'worried', 'doubt',
            'problem', 'issue', 'complaint', 'difficult', 'challenge', 'obstacle',
            'impossible', 'unrealistic', 'expensive', 'costly', 'budget constraints',
            'not feasible', 'not viable', 'not suitable', 'not qualified',
            'no', 'never', 'waste', 'useless', 'pointless', 'fail', 'failure', 'loss'
        }

        self.neutral_keywords = {
            'get back', 'reach you', 'call back', 'follow up', 'soon', 'later',
            'think about', 'consider', 'review', 'evaluate', 'discuss', 'meeting',
            'schedule', 'arrange', 'plan', 'waiting', 'pending', 'processing',
            'under review', 'in progress', 'maybe', 'perhaps', 'possibly',
            'might', 'could', 'would', 'should', 'neutral', 'okay', 'fine',
            'average', 'moderate', 'standard', 'normal', 'typical', 'usual',
            'next week', 'next month', 'next quarter', 'timeline', 'deadline',
            'paperwork', 'documentation', 'approval needed', 'manager decision',
            'information', 'details', 'clarification', 'question', 'inquiry'
        }

        # Negation words that can flip sentiment
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere',
            'barely', 'hardly', 'scarcely', 'seldom', 'rarely', 'cannot', "can't",
            "won't", "wouldn't", "shouldn't", "couldn't", "doesn't", "don't",
            "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
            "hadn't", "but", "however", "although", "though", "yet", "except"
        }

    def preprocess_text(self, text):
        """Standard text preprocessing"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        return ' '.join(words)

    def analyze_keywords_with_context(self, text):
        """Analyze sentiment using keywords with context awareness"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        # Count keyword occurrences
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        # Check for multi-word phrases first
        for phrase in ['not interested', 'no reply', 'no response', 'budget constraints',
                       'not feasible', 'not viable', 'not suitable', 'not qualified',
                       'get back', 'reach you', 'call back', 'follow up',
                       'think about', 'under review', 'in progress', 'approval needed']:
            if phrase in text_lower:
                if phrase in self.negative_keywords:
                    negative_count += 2  # Give more weight to phrases
                elif phrase in self.neutral_keywords:
                    neutral_count += 2
                elif phrase in self.positive_keywords:
                    positive_count += 2

        # Analyze individual words with negation context
        for i, word in enumerate(words):
            # Check for negation in previous 3 words
            negated = False
            if i > 0:
                prev_words = words[max(0, i - 3):i]
                if any(neg_word in prev_words for neg_word in self.negation_words):
                    negated = True

            # Check current word sentiment
            if word in self.positive_keywords:
                if negated:
                    negative_count += 1
                else:
                    positive_count += 1
            elif word in self.negative_keywords:
                if negated:
                    positive_count += 1
                else:
                    negative_count += 1
            elif word in self.neutral_keywords:
                neutral_count += 1

        # Determine sentiment based on keyword analysis
        total_sentiment_words = positive_count + negative_count + neutral_count

        if total_sentiment_words == 0:
            return None, 0.0  # No keywords found

        pos_ratio = positive_count / total_sentiment_words
        neg_ratio = negative_count / total_sentiment_words
        neu_ratio = neutral_count / total_sentiment_words

        # Determine sentiment with confidence
        if pos_ratio > neg_ratio and pos_ratio > neu_ratio:
            sentiment = 'positive'
            confidence = pos_ratio
        elif neg_ratio > pos_ratio and neg_ratio > neu_ratio:
            sentiment = 'negative'
            confidence = neg_ratio
        else:
            sentiment = 'neutral'
            confidence = neu_ratio

        # Boost confidence if there's a clear winner
        if confidence > 0.6:
            confidence = min(0.95, confidence + 0.2)
        elif confidence > 0.4:
            confidence = min(0.85, confidence + 0.1)

        return sentiment, confidence

    def analyze_mixed_sentiment(self, text):
        """Handle mixed sentiment scenarios like 'good response but not interested'"""
        text_lower = text.lower()

        # Look for contrasting patterns
        contrast_patterns = [
            (r'(good|great|excellent|positive|interested).*but.*(not interested|declined|rejected|no)', 'negative'),
            (r'(interested|good).*but.*(expensive|costly|budget)', 'negative'),
            (r'(not good|bad|poor).*but.*(interested|consider|maybe)', 'neutral'),
            (r'(positive|good).*however.*(not|no|declined)', 'negative'),
            (r'(like|love).*but.*(cannot|can\'t|won\'t)', 'negative'),
        ]

        for pattern, result_sentiment in contrast_patterns:
            if re.search(pattern, text_lower):
                return result_sentiment, 0.85

        return None, 0.0

    def combine_model_and_keywords(self, text, model_sentiment, model_confidence):
        """Combine model prediction with keyword analysis"""

        # First check for mixed sentiment patterns
        mixed_sentiment, mixed_confidence = self.analyze_mixed_sentiment(text)
        if mixed_sentiment:
            return mixed_sentiment, mixed_confidence

        # Get keyword-based sentiment
        keyword_sentiment, keyword_confidence = self.analyze_keywords_with_context(text)

        # If no keywords found, trust the model
        if keyword_sentiment is None:
            return model_sentiment, model_confidence

        # If model confidence is very high and matches keywords, trust it
        if model_confidence > 0.8 and model_sentiment == keyword_sentiment:
            return model_sentiment, min(0.95, (model_confidence + keyword_confidence) / 2 + 0.1)

        # If keyword confidence is very high, prioritize keywords
        if keyword_confidence > 0.8:
            return keyword_sentiment, keyword_confidence

        # If model confidence is low, trust keywords more
        if model_confidence < 0.6:
            return keyword_sentiment, keyword_confidence

        # If they disagree, use weighted combination
        if model_sentiment != keyword_sentiment:
            # Give slight preference to keywords for business context
            if keyword_confidence > 0.5:
                return keyword_sentiment, keyword_confidence
            else:
                return model_sentiment, model_confidence * 0.8  # Reduce confidence when disagreeing

        # If they agree, combine confidences
        combined_confidence = (model_confidence + keyword_confidence) / 2
        return model_sentiment, min(0.95, combined_confidence)


def preprocess_text(text):
    """Backward compatibility function"""
    analyzer = EnhancedSentimentAnalyzer()
    return analyzer.preprocess_text(text)
