import requests
from bs4 import BeautifulSoup
import os

def fetch_article_text(url, timeout=15):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text

def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")

    #remove script, style, ads
    for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "aside"]):
        tag.decompose()

    # title
    title = ""
    title_tag = soup.find("h1", class_="title_detail")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # --- Lead
    lead = ""
    lead_tag = soup.find(["div", "p", "span"], class_="lead_post_detail")
    if lead_tag:
        lead = lead_tag.get_text(" ", strip=True)

    #main
    main = ""
    main_tag = soup.find("div", class_="fck_detail")
    if main_tag:
        #remove img
        for img in main_tag.find_all(["img", "figure", "picture", "svg"]):
            img.decompose()
        for bad in main_tag.select("[class*='related'], [class*='advert'], [class*='ads'], [id*='related']"):
            bad.decompose()

        paragraphs = [
            p.get_text(" ", strip=True)
            for p in main_tag.find_all(["p", "h2", "h3", "li", "span"])
            if p.get_text(strip=True)
        ]
        main = "\n\n".join(paragraphs)

    #merge
    parts = [p for p in [title, lead, main] if p]
    return "\n\n".join(parts).strip()

def save_text_to_file(text, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return os.path.abspath(filename)

if __name__ == "__main__":
    print("URL: ")
    count = 1
    while True:
        url = input(f"\nLink #{count}: ").strip()
        html = fetch_article_text(url)
        text = extract_text_from_html(html)
        if not text:
            continue
        filename = f"sport{count}.txt"
        path = save_text_to_file(text, filename)
        print(f"Saved: {path}")
        count += 1

