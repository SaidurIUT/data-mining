# models/book.py

from pydantic import BaseModel

class Book(BaseModel):
    title: str
    price: str
    rating: str
    availability: str
