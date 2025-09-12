from bs4 import BeautifulSoup
import re
from collections import Counter

# 1. Load local HTML file
with open("page.html", "r", encoding="utf-8") as f:
    html = f.read()

# 2. Extract visible text
soup = BeautifulSoup(html, "html.parser")

# Remove script and style elements
for script in soup(["script", "style"]):
    script.decompose()

text = soup.get_text(separator=" ")

# 3. Keep only letters (a-z, A-Z), everything else -> whitespace
text = re.sub(r"[^a-zA-Z]", " ", text)

# 4. Convert to lowercase
text = text.lower()

# 5. Split into words
words = text.split()

# 6. Count word frequencies
counter = Counter(words)

# 7. Print top 20 words
for word, freq in counter.most_common(20):
    print(freq, word)
