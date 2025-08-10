import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def clean_text(text):
   
    text = text.lower()  # Lowercasing

    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation

    words = text.split()  # Tokenization

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return " ".join(words)

if __name__ == '__main__':
    # Example usage:
    sample_text = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
    cleaned_text = clean_text(sample_text)
    print(f"Original: {sample_text}")
    print(f"Cleaned: {cleaned_text}")