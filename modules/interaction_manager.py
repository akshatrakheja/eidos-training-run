import openai
from feedback import FeedbackPipeline


class InteractionManager:
    def __init__(self, model_name, feedback_pipeline):
        self.model_name = model_name
        self.feedback_pipeline = feedback_pipeline

    def interact_with_user(self, user_input):
        """
        Interacts with the user by querying the fine-tuned model and collecting feedback.
        """
        # Query the fine-tuned model
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a moderator for the user to explore the question 'what wins an election'."},
                {"role": "user", "content": user_input}
            ]
        )

        # Parse the assistant's response
        assistant_response = response["choices"][0]["message"]["content"]

        # Simulate user feedback collection (e.g., rating or engagement)
        user_rating = int(input("Rate the assistant's response (1-5): "))
        engagement_duration = float(input("How long did you spend thinking about the response? (in seconds): "))

        # Log feedback
        interaction_id = f"interaction_{len(open(self.feedback_pipeline.storage_path).readlines()) + 1}"
        feedback_entry = self.feedback_pipeline.collect_feedback(
            interaction_id,
            assistant_question=assistant_response,
            user_response=user_input,
            user_rating=user_rating,
            engagement_duration=engagement_duration
        )

        # Return the assistant's response and collected feedback
        return assistant_response, feedback_entry


if __name__ == "__main__":
    # Instantiate the feedback pipeline
    feedback_pipeline = FeedbackPipeline()

    # Instantiate the interaction manager with the fine-tuned model name
    interaction_manager = InteractionManager(model_name="gpt-4o-mini", feedback_pipeline=feedback_pipeline)

    # Simulate user interaction
    print("Welcome to the system. Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        assistant_response, feedback_entry = interaction_manager.interact_with_user(user_input)
        print(f"Assistant: {assistant_response}")