from app import app, load_model

# Load model during the cold start of the serverless function
load_model()

# This exposes the app for Vercel's Python runtime
# Vercel will look for the variable named 'app'
handler = app
