import gensim.downloader as api
import cohere

co = cohere.Client("")

print("Loading Word2Vec model (Google News)...")
model = api.load("word2vec-google-news-300")
print("Model loaded successfully.\n")

def get_similar_words(seed_word, top_n=5):
    try:
        similar = model.most_similar(seed_word, topn=top_n)
        return [word for word, score in similar]
    except KeyError:
        print(f"'{seed_word}' not found in Word2Vec vocabulary.")
        return []

def generate_paragraph_with_cohere(seed_word, similar_words):
    word_str = ", ".join(similar_words)
    prompt = (
        f"Write a creative and meaningful paragraph using the word '{seed_word}', "
        f"and these related words as a starting point: {word_str}. "
        f"Make sure the paragraph is coherent, with clear sentence structure, and stays on topic."
    )
    try:
        response = co.generate(
            model='command', 
            prompt=prompt,
            max_tokens=150,
            temperature=0.8
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"Cohere API error: {str(e)}"

def main():
    seed_word = input("Enter a seed word: ").strip().lower()
    similar_words = get_similar_words(seed_word)

    if not similar_words:
        print("Could not find similar words. Try a different seed word.")
        return

    print(f"Similar words found: {', '.join(similar_words)}\n")

    paragraph = generate_paragraph_with_cohere(seed_word, similar_words)
    print("Generated Paragraph:\n")
    print(paragraph)

if __name__ == "__main__":
    main()