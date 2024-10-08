# src/easy_programming_tools/tools.py

import math
from datetime import datetime, timedelta
from collections import Counter

# ---------------------------
# Mathematical Utilities
# ---------------------------

def add_numbers(a, b):
    return a + b

def subtract_numbers(a, b):
    return a - b

def multiply_numbers(a, b):
    return a * b

def divide_numbers(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b

def factorial(n):
    if n < 0:
        raise ValueError("Cannot compute factorial of a negative number.")
    return math.factorial(n)

def fibonacci(n):
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence[:n]

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def gcd(a, b):
    """Returns the greatest common divisor (GCD) of two numbers."""
    return math.gcd(a, b)

def lcm(a, b):
    """Returns the least common multiple (LCM) of two numbers."""
    return abs(a * b) // math.gcd(a, b)

def square(n):
    return n * n

def cube(n):
    return n * n * n

def power(base, exp):
    return base ** exp

def sqrt(n):
    if n < 0:
        raise ValueError("Cannot compute square root of a negative number.")
    return math.sqrt(n)

def celsius_to_fahrenheit(c):
    return (c * 9/5) + 32

def fahrenheit_to_celsius(f):
    return (f - 32) * 5/9

def deg_to_rad(degrees):
    return math.radians(degrees)

def rad_to_deg(radians):
    return math.degrees(radians)

def distance_2d(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# ---------------------------
# String Utilities
# ---------------------------

def reverse_string(s):
    return s[::-1]

def is_palindrome(s):
    return s == s[::-1]

def count_vowels(s):
    return sum(1 for char in s.lower() if char in 'aeiou')

def capitalize_words(s):
    return ' '.join(word.capitalize() for word in s.split())

def count_words(s):
    return len(s.split())

def to_uppercase(s):
    return s.upper()

def to_lowercase(s):
    return s.lower()

def remove_whitespace(s):
    return s.strip()

def replace_spaces(s, replacement="_"):
    return s.replace(" ", replacement)

def remove_punctuation(s):
    return ''.join(char for char in s if char.isalnum() or char.isspace())

def is_anagram(s1, s2):
    return sorted(s1.replace(" ", "").lower()) == sorted(s2.replace(" ", "").lower())

def count_occurrences(s, sub):
    return s.count(sub)

def find_substring(s, sub):
    return s.find(sub)

def repeat_string(s, n):
    return s * n

def split_into_words(s):
    return s.split()

def join_words(words, delimiter=" "):
    return delimiter.join(words)

# ---------------------------
# Date and Time Utilities
# ---------------------------

def current_time():
    return datetime.now().strftime("%H:%M:%S")

def current_date():
    return datetime.now().strftime("%Y-%m-%d")

def days_between_dates(date1, date2, fmt='%Y-%m-%d'):
    d1 = datetime.strptime(date1, fmt)
    d2 = datetime.strptime(date2, fmt)
    return abs((d2 - d1).days)

def add_days_to_date(date_str, days, fmt='%Y-%m-%d'):
    date_obj = datetime.strptime(date_str, fmt)
    return (date_obj + timedelta(days=days)).strftime(fmt)

def subtract_days_from_date(date_str, days, fmt='%Y-%m-%d'):
    date_obj = datetime.strptime(date_str, fmt)
    return (date_obj - timedelta(days=days)).strftime(fmt)

def is_weekend(date_str, fmt='%Y-%m-%d'):
    date_obj = datetime.strptime(date_str, fmt)
    return date_obj.weekday() >= 5  # 5=Saturday, 6=Sunday

def days_in_month(year, month):
    next_month = month % 12 + 1
    next_month_first_day = datetime(year + (month // 12), next_month, 1)
    return (next_month_first_day - timedelta(days=1)).day

def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def time_until_midnight():
    now = datetime.now()
    midnight = datetime.combine(now + timedelta(days=1), datetime.min.time())
    return (midnight - now).seconds

# ---------------------------
# List Utilities
# ---------------------------

def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

def remove_duplicates(lst):
    return list(dict.fromkeys(lst))

def chunk_list(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]

def find_max(lst):
    if not lst:
        raise ValueError("The list is empty.")
    return max(lst)

def find_min(lst):
    if not lst:
        raise ValueError("The list is empty.")
    return min(lst)

def average(lst):
    if not lst:
        raise ValueError("The list is empty.")
    return sum(lst) / len(lst)

def most_common(lst):
    if not lst:
        raise ValueError("The list is empty.")
    return Counter(lst).most_common(1)[0][0]

def find_median(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    mid = n // 2
    return (sorted_lst[mid] + sorted_lst[~mid]) / 2

def sum_of_elements(lst):
    return sum(lst)

def product_of_elements(lst):
    product = 1
    for num in lst:
        product *= num
    return product
