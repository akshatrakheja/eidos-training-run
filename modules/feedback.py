import json
from datetime import datetime
from textblob import TextBlob  # For sentiment analysis
import os


class FeedbackPipeline:
    def __init__(self, storage_path="feedback_data.json"):
        self.storage_path = storage_path

    def collect_feedback(self, interaction_id, assistant_question, user_response, user_rating=None, engagement_duration=None):
        """
        Collects user interaction and feedback data.
        """
        feedback_entry = {
            "interaction_id": interaction_id,
            "timestamp": datetime.now().isoformat(),
            "assistant_question": assistant_question,
            "user_response": user_response,
            "feedback": {
                "user_rating": user_rating,
                "engagement_duration": engagement_duration
            }
        }
        self.store_feedback_locally(feedback_entry)
        return feedback_entry

    def analyze_feedback(self, user_response, engagement_duration):
        """
        Analyze user feedback for sentiment and engagement.
        """
        sentiment = TextBlob(user_response).sentiment.polarity
        sentiment_label = "positive" if sentiment > 0 else "neutral" if sentiment == 0 else "negative"
        score = 1 if engagement_duration and engagement_duration > 5 else 0.5  # Example scoring logic

        return {"sentiment": sentiment_label, "engagement_score": score}

    def store_feedback_locally(self, feedback_entry):
        """
        Save feedback data locally in a JSON file.
        """
        if not os.path.exists(self.storage_path):
            with open(self.storage_path, 'w') as file:
                json.dump([], file)

        with open(self.storage_path, 'r+') as file:
            data = json.load(file)
            data.append(feedback_entry)
            file.seek(0)
            json.dump(data, file, indent=4)

    def generate_suggestions(self):
        """
        Generate suggestions for improving the assistant's questions.
        """
        if not os.path.exists(self.storage_path):
            return []

        with open(self.storage_path, 'r') as file:
            feedback_data = json.load(file)

        poorly_rated = [item for item in feedback_data if item["feedback"]["user_rating"] and item["feedback"]["user_rating"] <= 2]
        high_engagement = [item for item in feedback_data if item["feedback"]["engagement_duration"] and item["feedback"]["engagement_duration"] > 10]

        # Example suggestions
        suggestions = []
        for entry in poorly_rated:
            suggestions.append(f"Rephrase question: '{entry['assistant_question']}'")

        for entry in high_engagement:
            suggestions.append(f"Focus on questions related to: '{entry['assistant_question']}' (high engagement)")

        return suggestions