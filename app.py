# app.py
import streamlit as st
from transformers import pipeline
import random
import numpy as np

# Sentiment detection
sentiment_pipeline = pipeline("sentiment-analysis")
def get_sentiment(text):
    return sentiment_pipeline(text)[0]['label'].lower()

# Environment & Q-Agent setup
class SentimentEnvironment:
    def __init__(self):
        self.responses = {
            'positive': ["That's wonderful!", "Glad to hear that!", "Awesome! 😊"],
            'negative': ["I'm here for you.", "That sounds tough. 😞", "Let me help you with that."],
            'neutral': ["Okay.", "I see.", "Let me know how I can assist."]
        }

    def step(self, action_idx, sentiment):
        response = self.responses[sentiment][action_idx]
        expected_sentiment = get_sentiment(response)
        reward = 1 if expected_sentiment == sentiment else 0
        return response, reward

class QLearningAgent:
    def __init__(self, sentiments, num_actions=3, lr=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {s: np.zeros(num_actions) for s in sentiments}
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward):
        best_next = np.max(self.q_table[state])
        self.q_table[state][action] += self.lr * (reward + self.gamma * best_next - self.q_table[state][action])

# Train RL agent
env = SentimentEnvironment()
agent = QLearningAgent(['positive', 'negative', 'neutral'])

for ep in range(300):
    for sentiment in ['positive', 'negative', 'neutral']:
        action = agent.select_action(sentiment)
        _, reward = env.step(action, sentiment)
        agent.update(sentiment, action, reward)

# Inference wrapper
def chat_response(user_input):
    sentiment = get_sentiment(user_input)
    action_idx = agent.select_action(sentiment)
    return env.responses[sentiment][action_idx]

# Streamlit UI
st.title("🤖 RL-based Sentiment Dialogue System")

user_input = st.text_input("You:")
if user_input:
    bot_response = chat_response(user_input)
    st.markdown(f"**Bot:** {bot_response}")
