import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def is_relevant(question, answer, page_text):
    query = question + " " + answer

    emb1 = model.encode(query, convert_to_tensor=True)
    emb2 = model.encode(page_text[:2000], convert_to_tensor=True)  # truncate

    score = util.cos_sim(emb1, emb2).item()

    return score > 0.3   # tune threshold

def verify_source(url, question, answer):
    try:
        r = requests.get(url, timeout=5)

        # Step 1: status
        if r.status_code != 200:
            return "invalid_url"

        # Step 2: content check
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text().lower()

        if "does not exist" in text or "404" in text:
            return "fake_page"

        # Step 3: relevance (simple version)
        # if len(text) < 200:
        #     return "low_content"

        # return "is_relevant" if is_relevant(question, answer) else "is_not_relevant"
        if not is_relevant(question, answer) : return "is_not_relevant" 
    

        # (optional: plug in semantic model here)



        return "valid"

    except:
        return "error"