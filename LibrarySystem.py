# The purpose of this code is to design an online interface for library management.
# This is a customer facing interface that allows users to execute various actions such as checkout
# books, return books, reserve books, and receive information about books.

class Book:
    def __init__(self, title="", author="", isbn=0):
        """Defines a book with attributes title, author, ISBN, status, and reservation status."""
        self.title = title
        self.author = author
        self.ISBN = isbn
        self.status = "available"
        self.reserved = False
        self.reserved_by = 'N/A'

    def __str__(self):
        """Returns the title, author, ISBN, and availability status of the book."""
        return print(f'Title: {self.title}\nAuthor: {self.author}\nISBN: {self.ISBN}\nStatus: {self.status}')

    def checkout(self):
        """Allows a user to check out the book."""
        if self.status == "available":
            self.status = "borrowed"
            print(f"{self.title} has been checked out.")
        elif self.status == "reserved":
            self.status = "borrowed"
            self.reserved = False
            self.reserved_by = 'N/A'
            print(f"{self.title} has been checked out.")
        else:
            print(f"{self.title} is borrowed and is not available for checkout.")

    def return_book(self):
        """Marks a book as available."""
        if self.status == "borrowed" and self.reserved == True:
            print(f"To {self.reserved_by}, {self.title} has been returned and is available for checkout.")
            self.status = "reserved"
        elif self.status == "borrowed" and self.reserved == False:
            self.status = "available"
            print(f"{self.title} has been successfully returned.")
        elif self.status == "available":
            print(f"{self.title} is available and cannot be returned.")


class LibraryUser:
    """Defines a library user who can borrow and return books."""

    def __init__(self, name="", user_id=0):
        """Initializes the library user with attributes name(String) and user_id(int)."""
        self.name = name
        self.user_id = user_id
        self.books_borrowed = []

    def borrow_book(self, book):
        """Will add the book to self.books_borrowed and apply book.checkout() to make the book borrowed."""
        if book.status == 'available':
            self.books_borrowed.append(book)
            book.checkout()
        elif book.status == 'reserved' and self.name == book.reserved_by:
            book.checkout()
            self.books_borrowed.append(book)
        elif book.status == 'reserved' and self.name != book.reserved_by:
            print(f"{book.title} is reserved by another user and not available for checkout at this time.")
        else:
            print(f"{book.title} is already borrowed by another user.")

    def return_book(self, book):
        """Will remove the book from self.books_borrowed and apply book.return_book() to make the book available."""
        if book.status == 'borrowed' and book in self.books_borrowed:
            book.return_book()
            self.books_borrowed.pop(self.books_borrowed.index(book))
        elif book.status == 'borrowed' and book not in self.books_borrowed:
            print(f"{self.name} does not have this book borrowed.")
        else:
            book.return_book()

    def reserve_book(self, book):
        """Allows a library user to reserve a book that is currently borrowed."""
        if book.status == "borrowed" and book.reserved == False:
            book.reserved = True
            book.reserved_by = self.name
            print(f"{book.title} is now reserved by {self.name}")
        elif book.status == "borrowed" and book.reserved == True:
            print(f"{book.title} is already reserved by another library user.")
        else:
            print(f"{book.title} is available for checkout instead of reservation.")

    def cancel_reservation(self, book):
        """Allows a user with a reservation of a book to cancel their reservation."""
        if book.reserved_by == self.name:
            book.reserved = False
            book.reserved_by = 'N/A'
            print(f"{self.name} has canceled their reservation of {book.title}.")
        else:
            print(f"{self.name} does not have this book reserved.")
