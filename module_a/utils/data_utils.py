# utils/data_utils.py

import csv
from models.book import Book


def is_duplicate_book(title: str, seen_titles: set) -> bool:
    return title in seen_titles


def is_complete_book(book: dict, required_keys: list) -> bool:
    return all(key in book for key in required_keys)


def save_books_to_csv(books: list, filename: str):
    if not books:
        print("No books to save.")
        return

    fieldnames = Book.model_fields.keys()

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(books)

    print(f"Saved {len(books)} books to '{filename}'.")
