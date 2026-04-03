"""
================================================================================
FYP: Finding the Most Accurate LLM for Sentiment Analysis
================================================================================
Author: Umar Raza
University of Hertfordshire - BSc Computer Science

This script:
1. Compares 3 LLMs on emotion detection using GoEmotions dataset
2. Prints accuracy comparison
3. Lets you chat with the most accurate model
================================================================================
"""

import os
import csv
import random
import time
import json
import requests

# =============================================================================
# API KEYS
# =============================================================================
GROQ_API_KEY = ""
OPENROUTER_TRINITY_KEY = ""
OPENROUTER_DEEPSEEK_KEY = ""

# =============================================================================
# THREE MODELS TO COMPARE
# =============================================================================
MODELS = [
    {
        "name": "LLaMA 3.1 8B",
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "api_key": GROQ_API_KEY,
    },
    {
        "name": "GLM-4.5 Air",
        "provider": "openrouter",
        "model_id": "z-ai/glm-4.5-air:free",
        "api_key": OPENROUTER_DEEPSEEK_KEY,
    },
    {
        "name": "Trinity Large",
        "provider": "openrouter",
        "model_id": "arcee-ai/trinity-large-preview:free",
        "api_key": OPENROUTER_TRINITY_KEY,
    },
]

# =============================================================================
# SETTINGS
# =============================================================================
NUM_SAMPLES = 100          # Number of test samples
DELAY_BETWEEN_CALLS = 1.0  # Seconds between API calls

# 28 emotions from GoEmotions dataset
EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

# =============================================================================
# API CALL FUNCTIONS
# =============================================================================
def call_groq(prompt, model_id, api_key, temperature=0, max_tokens=10):
    """Call Groq API (LLaMA)."""
    from groq import Groq
    client = Groq(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def call_openrouter(prompt, model_id, api_key, temperature=0, max_tokens=10):
    """Call OpenRouter API."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/umarraza/fyp",
        "X-Title": "FYP Sentiment Analysis"
    }
    
    data = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"API Error {response.status_code}: {response.text}")


def call_llm(provider, model_id, api_key, prompt, temperature=0, max_tokens=10):
    """Call the appropriate API based on provider."""
    if provider == "groq":
        return call_groq(prompt, model_id, api_key, temperature, max_tokens)
    elif provider == "openrouter":
        return call_openrouter(prompt, model_id, api_key, temperature, max_tokens)
    else:
        raise Exception(f"Unknown provider: {provider}")

# =============================================================================
# LOAD TEST DATA
# =============================================================================
def load_test_data(filepath, num_samples):
    """Load test samples from GoEmotions dataset."""
    examples = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            
            text = row[0]
            try:
                labels = [int(x) for x in row[1].split(",")]
                if len(labels) == 1:  # Single-label only
                    examples.append((text, labels[0]))
            except:
                continue
    
    # Random sample
    random.seed(42)
    if len(examples) > num_samples:
        examples = random.sample(examples, num_samples)
    
    return examples

# =============================================================================
# PREDICT EMOTION
# =============================================================================
def predict_emotion(provider, model_id, api_key, text):
    """Ask the LLM to predict the emotion."""
    
    prompt = f"""Classify the emotion in this text. Reply with ONLY one word - the emotion.

Emotions: {", ".join(EMOTIONS)}

Text: "{text}"

Emotion:"""

    try:
        result = call_llm(provider, model_id, api_key, prompt, temperature=0, max_tokens=10)
        result = result.lower().strip(".,!?\"'").split()[0] if result else ""
        
        # Match to valid emotion
        for emo in EMOTIONS:
            if result == emo or result.startswith(emo):
                return emo
        return None
        
    except Exception as e:
        print(f"\n      [API Error: {str(e)[:100]}]")
        return None

# =============================================================================
# EVALUATE ONE MODEL
# =============================================================================
def evaluate_model(model, test_data):
    """Test a model on all samples and return accuracy."""
    
    print(f"\n   Testing {model['name']}...")
    
    correct = 0
    total = 0
    errors = 0
    
    for i, (text, true_label) in enumerate(test_data):
        true_emotion = EMOTIONS[true_label]
        
        predicted = predict_emotion(model["provider"], model["model_id"], model["api_key"], text)
        
        if predicted is None:
            errors += 1
        elif predicted == true_emotion:
            correct += 1
        total += 1
        
        # Progress update
        if (i + 1) % 20 == 0 or (i + 1) == len(test_data):
            acc = correct / (total - errors) * 100 if (total - errors) > 0 else 0
            print(f"      Progress: {i+1}/{len(test_data)} | Accuracy: {acc:.1f}% | Errors: {errors}")
        
        time.sleep(DELAY_BETWEEN_CALLS)
    
    valid = total - errors
    accuracy = correct / valid * 100 if valid > 0 else 0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors
    }

# =============================================================================
# RUN COMPARISON
# =============================================================================
def run_comparison(test_data):
    """Compare all models."""
    
    print("\n" + "="*60)
    print("RUNNING SENTIMENT ANALYSIS COMPARISON")
    print(f"Testing {len(MODELS)} models on {len(test_data)} samples")
    print("="*60)
    
    results = []
    
    for model in MODELS:
        stats = evaluate_model(model, test_data)
        results.append({
            "name": model["name"],
            "provider": model["provider"],
            "model_id": model["model_id"],
            "api_key": model["api_key"],
            "accuracy": stats["accuracy"],
            "correct": stats["correct"],
            "total": stats["total"],
            "errors": stats["errors"]
        })
        print(f"   ✓ {model['name']}: {stats['accuracy']:.2f}%")
    
    return results

# =============================================================================
# PRINT RESULTS
# =============================================================================
def print_results(results):
    """Print comparison table and identify winner."""
    
    # Sort by accuracy
    results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS ACCURACY COMPARISON")
    print("="*60)
    print(f"\n{'Rank':<6}{'Model':<25}{'Accuracy':<15}")
    print("-"*46)
    
    for i, r in enumerate(results, 1):
        marker = "🏆" if i == 1 else "  "
        valid = r['total'] - r['errors']
        print(f"{marker} {i:<4}{r['name']:<25}{r['accuracy']:.2f}% ({r['correct']}/{valid})")
    
    print("-"*46)
    
    winner = results[0]
    print(f"\n🏆 WINNER: {winner['name']} with {winner['accuracy']:.2f}% accuracy")
    
    return winner

# =============================================================================
# CHATBOT WITH WINNER
# =============================================================================
def chat_with_winner(winner):
    """Chat with the winning model."""
    
    print("\n" + "="*60)
    print(f"CHATTING WITH {winner['name'].upper()}")
    print(f"This model achieved {winner['accuracy']:.2f}% accuracy")
    print("="*60)
    print("\nType your message. Type 'quit' to exit.\n")
    
    provider = winner["provider"]
    model_id = winner["model_id"]
    api_key = winner["api_key"]
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye! 👋")
            break
        
        # Detect emotion
        print("   [Analyzing...]", end="\r")
        emotion = predict_emotion(provider, model_id, api_key, user_input)
        if not emotion:
            emotion = "neutral"
        print(f"   [Detected: {emotion}]" + " "*20)
        
        # Generate response
        prompt = f"""You are a supportive, empathetic AI assistant.

The user said: "{user_input}"
Their detected emotion: {emotion}

Respond naturally and empathetically in 2-3 sentences."""

        try:
            response = call_llm(provider, model_id, api_key, prompt, temperature=0.7, max_tokens=150)
            print(f"\n🤖 {response}\n")
        except Exception as e:
            print(f"\n🤖 Error: {e}\n")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*60)
    print("FYP: FINDING THE MOST ACCURATE LLM FOR SENTIMENT ANALYSIS")
    print("By Umar Raza - University of Hertfordshire")
    print("="*60)
    
    # Test API connections
    print("\n[1/4] Testing API connections...")
    
    try:
        call_groq("Say OK", "llama-3.1-8b-instant", GROQ_API_KEY, max_tokens=5)
        print("   ✓ Groq API (LLaMA) connected")
    except Exception as e:
        print(f"   ✗ Groq failed: {e}")
    
    try:
        call_openrouter("Say OK", "z-ai/glm-4.5-air:free", OPENROUTER_DEEPSEEK_KEY, max_tokens=5)
        print("   ✓ OpenRouter API (GLM-4.5 Air) connected")
    except Exception as e:
        print(f"   ✗ GLM-4.5 Air failed: {e}")
    
    try:
        call_openrouter("Say OK", "arcee-ai/trinity-large-preview:free", OPENROUTER_TRINITY_KEY, max_tokens=5)
        print("   ✓ OpenRouter API (Trinity) connected")
    except Exception as e:
        print(f"   ✗ Trinity failed: {e}")
    
    # Load data
    print("\n[2/4] Loading test data...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(script_dir, "data", "test.tsv")
    
    if not os.path.exists(test_file):
        test_file = "data/test.tsv"
    
    if not os.path.exists(test_file):
        print(f"   ✗ Test file not found!")
        print("   Make sure 'data/test.tsv' exists")
        return
    
    test_data = load_test_data(test_file, NUM_SAMPLES)
    print(f"   ✓ Loaded {len(test_data)} test samples")
    
    # Compare models
    print("\n[3/4] Comparing models (this takes a few minutes)...")
    results = run_comparison(test_data)
    
    # Show results
    print("\n[4/4] Results...")
    winner = print_results(results)
    
    # Save results
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n💾 Results saved to comparison_results.json")
    
    # Chat option
    print("\n" + "-"*60)
    choice = input("\nChat with the winning model? (y/n): ").strip().lower()
    
    if choice in ["y", "yes", ""]:
        chat_with_winner(winner)

if __name__ == "__main__":
    main()
