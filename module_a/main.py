# main.py
import asyncio
from crawl4ai import AsyncWebCrawler
from dotenv import load_dotenv

from config import BASE_URL, CSS_SELECTOR, REQUIRED_KEYS
from utils.data_utils import save_books_to_csv
from utils.scraper_utils import (
    fetch_and_process_page,
    get_browser_config,
    get_llm_strategy,
)

load_dotenv()


async def crawl_books():
    browser_config = get_browser_config()
    llm_strategy = get_llm_strategy()
    session_id = "book_crawl_session"

    page_number = 1
    all_books = []
    seen_titles = set()

    async with AsyncWebCrawler(config=browser_config) as crawler:
        while True:
            books, _ = await fetch_and_process_page(
                crawler,
                page_number,
                BASE_URL,
                CSS_SELECTOR,
                llm_strategy,
                session_id,
                REQUIRED_KEYS,
                seen_titles,
            )

            if not books:
                print("No more books found. Ending crawl.")
                break

            all_books.extend(books)
            page_number += 1
            await asyncio.sleep(18)

    if all_books:
        save_books_to_csv(all_books, "books.csv")
    else:
        print("No books scraped.")

    llm_strategy.show_usage()


async def main():
    await crawl_books()


if __name__ == "__main__":
    asyncio.run(main())
