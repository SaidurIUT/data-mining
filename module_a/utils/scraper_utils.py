# utils/scraper_utils.py
import json
import os
import asyncio
from typing import List, Set, Tuple

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMExtractionStrategy,
)

from models.book import Book
from utils.data_utils import is_complete_book, is_duplicate_book


# -------------------------------
# Browser configuration
# -------------------------------
def get_browser_config() -> BrowserConfig:
    return BrowserConfig(
        browser_type="chromium",
        headless=False,
        verbose=True,
    )


# -------------------------------
# LLM extraction strategy
# -------------------------------
def get_llm_strategy() -> LLMExtractionStrategy:
    return LLMExtractionStrategy(
        provider="groq/llama-3.1-8b-instant",
        api_token=os.getenv("GROQ_API_KEY"),
        schema=Book.model_json_schema(),
        extraction_type="schema",
        instruction=(
            "Extract all book objects with the following fields:\n"
            "- title (book title)\n"
            "- price (including currency symbol)\n"
            "- rating (One, Two, Three, Four, Five)\n"
            "- availability (e.g. 'In stock')"
        ),
        input_format="markdown",

        # 🔽 CRITICAL: keep token usage low
        max_input_tokens=2500,
        max_tokens=1200,
        temperature=0,

        # 🔽 Disable LiteLLM retries (we handle them ourselves)
        retry_attempts=0,

        verbose=True,
    )


# -------------------------------
# Page fetch + LLM extraction
# -------------------------------
async def fetch_and_process_page(
    crawler: AsyncWebCrawler,
    page_number: int,
    base_url: str,
    css_selector: str,
    llm_strategy: LLMExtractionStrategy,
    session_id: str,
    required_keys: List[str],
    seen_titles: Set[str],
) -> Tuple[List[dict], bool]:

    url = f"{base_url}/page-{page_number}.html"
    print(f"Loading page {page_number}: {url}")

    result = None

    # 🔁 HARD retry with proper wait (Groq-aware)
    for attempt in range(2):
        try:
            result = await crawler.arun(
                url=url,
                config=CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    extraction_strategy=llm_strategy,
                    css_selector=css_selector,
                    session_id=session_id,
                ),
            )
            break

        except Exception as e:
            if "RateLimit" in str(e):
                wait_time = 16  # ⬅️ MUST be > Groq suggested wait
                print(f"Rate limit hit. Sleeping {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise e

    if result is None:
        print(f"Skipping page {page_number} due to repeated rate limits.")
        return [], False

    if not (result.success and result.extracted_content):
        print(f"No content on page {page_number}. Stopping.")
        return [], False

    try:
        extracted_data = json.loads(result.extracted_content)
    except json.JSONDecodeError:
        print(f"Invalid JSON returned by LLM on page {page_number}.")
        return [], False

    if not extracted_data:
        return [], False

    complete_books = []

    for book in extracted_data:
        if isinstance(book, dict) and book.get("error") is False:
            book.pop("error", None)

        if not is_complete_book(book, required_keys):
            continue

        if is_duplicate_book(book["title"], seen_titles):
            continue

        seen_titles.add(book["title"])
        complete_books.append(book)

    print(f"Extracted {len(complete_books)} books from page {page_number}")
    return complete_books, False
