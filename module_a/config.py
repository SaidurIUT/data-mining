# config.py

BASE_URL = "https://books.toscrape.com/catalogue"
CSS_SELECTOR = "article.product_pod"

REQUIRED_KEYS = [
    "title",
    "price",
    "rating",
    "availability",
]
