# easy_functions/utils.py
import datetime
import math
import random
import re
import inspect

# List all functions in the utils file
def list_all_functions():
    functions = [func for func, _ in inspect.getmembers(
        inspect.getmodule(inspect.currentframe()), inspect.isfunction)]
    return functions

# 1. Get the current timestamp
def current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 2. Convert a string to title case
def title_case(string):
    return string.title()

# 3. Check if a number is prime
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

# 4. Calculate factorial of a number
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

# 5. Reverse a string
def reverse_string(s):
    return s[::-1]

# 6. Convert Celsius to Fahrenheit
def celsius_to_fahrenheit(celsius):
    return celsius * 9/5 + 32

# 7. Convert Fahrenheit to Celsius
def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

# 8. Check if a number is even
def is_even(number):
    return number % 2 == 0

# 9. Check if a number is odd
def is_odd(number):
    return number % 2 != 0

# 10. Get the current day of the week
def current_day():
    return datetime.datetime.now().strftime("%A")

# 11. Convert a string to lowercase
def to_lowercase(string):
    return string.lower()

# 12. Convert a string to uppercase
def to_uppercase(string):
    return string.upper()

# 13. Get the length of a string
def string_length(s):
    return len(s)

# 14. Sum of elements in a list
def sum_of_list(lst):
    return sum(lst)

# 15. Find the average of a list
def average_of_list(lst):
    return sum(lst) / len(lst)

# 16. Sort a list in ascending order
def sort_list(lst):
    return sorted(lst)

# 17. Check if a string is a palindrome
def is_palindrome(s):
    return s == reverse_string(s)

# 18. Get unique elements of a list
def unique_elements(lst):
    return list(set(lst))

# 19. Merge two lists
def merge_lists(lst1, lst2):
    return lst1 + lst2

# 20. Remove duplicates from a list
def remove_duplicates(lst):
    return list(set(lst))

# 21. Find the maximum number in a list
def max_in_list(lst):
    return max(lst)

# 22. Find the minimum number in a list
def min_in_list(lst):
    return min(lst)

# 23. Check if a list is empty
def is_list_empty(lst):
    return len(lst) == 0

# 24. Get the square of a number
def square(number):
    return number ** 2

# 25. Get the cube of a number
def cube(number):
    return number ** 3

# 26. Generate a random integer within a range
def random_integer(start, end):
    return random.randint(start, end)

# 27. Calculate the power of a number
def power(base, exp):
    return base ** exp

# 28. Get the day of the month
def day_of_month():
    return datetime.datetime.now().day

# 29. Get the month
def current_month():
    return datetime.datetime.now().month

# 30. Get the year
def current_year():
    return datetime.datetime.now().year

# 31. Find the greatest common divisor (GCD)
def gcd(a, b):
    return math.gcd(a, b)

# 32. Find the least common multiple (LCM)
def lcm(a, b):
    return abs(a * b) // gcd(a, b)

# 33. Calculate simple interest
def simple_interest(principal, rate, time):
    return (principal * rate * time) / 100

# 34. Convert a string to a list of words
def string_to_list(string):
    return string.split()

# 35. Convert a list of words to a string
def list_to_string(lst):
    return " ".join(lst)

# 36. Find the median of a list
def median(lst):
    lst = sorted(lst)
    n = len(lst)
    if n % 2 == 0:
        return (lst[n // 2 - 1] + lst[n // 2]) / 2
    else:
        return lst[n // 2]

# 37. Count occurrences of an element in a list
def count_occurrences(lst, element):
    return lst.count(element)

# 38. Check if a string is a valid email address
def is_valid_email(email):
    regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(regex, email) is not None

# 39. Check if a number is a perfect square
def is_perfect_square(number):
    return number == int(number ** 0.5) ** 2

# 40. Find the square root of a number
def square_root(number):
    return number ** 0.5

# 41. Calculate the area of a circle
def circle_area(radius):
    return math.pi * radius * radius

# 42. Calculate the perimeter of a circle
def circle_perimeter(radius):
    return 2 * math.pi * radius

# 43. Calculate the area of a rectangle
def rectangle_area(length, width):
    return length * width

# 44. Calculate the perimeter of a rectangle
def rectangle_perimeter(length, width):
    return 2 * (length + width)

# 45. Calculate the area of a triangle
def triangle_area(base, height):
    return 0.5 * base * height

# 46. Convert a list of strings to uppercase
def list_to_uppercase(lst):
    return [item.upper() for item in lst]

# 47. Reverse a list
def reverse_list(lst):
    return lst[::-1]

# 48. Check if a list contains a specific element
def contains(lst, element):
    return element in lst

# 49. Remove an element from a list
def remove_element(lst, element):
    if element in lst:
        lst.remove(element)
    return lst

# 50. Calculate the Euclidean distance between two points (x1, y1) and (x2, y2)
def euclidean_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# 51. Get the current time in seconds since epoch
def current_time_seconds():
    return int(datetime.datetime.now().timestamp())

# 52. Generate a random float between two numbers
def random_float(start, end):
    return random.uniform(start, end)

# 53. Create a dictionary from two lists (keys and values)
def create_dict(keys, values):
    return dict(zip(keys, values))

# 54. Convert a list of numbers to their factorials
def list_factorial(lst):
    return [factorial(n) for n in lst]

# 55. Find the mode of a list
from statistics import mode

def mode_of_list(lst):
    return mode(lst)

# 56. Perform matrix multiplication
import numpy as np

def matrix_multiplication(A, B):
    return np.dot(A, B)

# 57. Perform matrix addition
def matrix_addition(A, B):
    return np.add(A, B)

# 58. Find the transpose of a matrix
def transpose_matrix(matrix):
    return np.transpose(matrix)

# 59. Check if a string contains digits only
def is_digit_string(string):
    return string.isdigit()

# 60. Compute the determinant of a 2x2 matrix
def determinant_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

# 61. Calculate the compound interest
def compound_interest(principal, rate, time, n):
    return principal * (1 + rate / (n * 100)) ** (n * time)

# 62. Check if a string is a valid URL
def is_valid_url(url):
    regex = r'^https?://[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    return re.match(regex, url) is not None

# 63. Calculate the cosine of an angle in degrees
def cos_degrees(degrees):
    return math.cos(math.radians(degrees))

# 64. Calculate the sine of an angle in degrees
def sin_degrees(degrees):
    return math.sin(math.radians(degrees))

# 65. Calculate the tangent of an angle in degrees
def tan_degrees(degrees):
    return math.tan(math.radians(degrees))

# 66. Convert a list of integers to their binary representations
def to_binary(lst):
    return [bin(n)[2:] for n in lst]

# 67. Find the harmonic mean of a list
def harmonic_mean(lst):
    return len(lst) / sum(1/x for x in lst)

# 68. Check if a string is a valid IPv4 address
def is_valid_ipv4(ip):
    regex = r'^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    return re.match(regex, ip) is not None

# 69. Generate a list of Fibonacci numbers
def fibonacci(n):
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[-1] + sequence[-2])
    return sequence[:n]

# 70. Calculate the logarithm (base 10) of a number
def log_base_10(number):
    return math.log10(number)

# 71. Generate a list of prime numbers up to n
def primes_up_to_n(n):
    return [i for i in range(2, n+1) if is_prime(i)]

# 72. Perform a quicksort on a list
def quicksort(lst):
    if len(lst) <= 1:
        return lst
    pivot = lst[0]
    less_than_pivot = [x for x in lst[1:] if x < pivot]
    greater_than_pivot = [x for x in lst[1:] if x >= pivot]
    return quicksort(less_than_pivot) + [pivot] + quicksort(greater_than_pivot)

# 73. Get the nth Fibonacci number
def nth_fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

# 74. Generate a random string of a given length
import string

def random_string(length):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

# 75. Compute the Greatest Common Divisor (GCD) of a list of numbers
def gcd_list(lst):
    return math.gcd(*lst)

# 76. Convert a list of numbers to their hexadecimal representation
def to_hexadecimal(lst):
    return [hex(n)[2:] for n in lst]

# 77. Calculate the area of a trapezoid
def trapezoid_area(a, b, h):
    return 0.5 * (a + b) * h

# 78. Check if a number is a power of 2
def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

# 79. Find the missing number in a list of consecutive numbers
def find_missing_number(lst):
    total_sum = sum(range(lst[0], lst[-1] + 1))
    return total_sum - sum(lst)

# 80. Calculate the combination (n choose r)
def combination(n, r):
    return math.comb(n, r)

# 81. Calculate the permutation (nPr)
def permutation(n, r):
    return math.perm(n, r)

# 82. Find the angle between two vectors
def angle_between_vectors(v1, v2):
    dot_product = sum(a*b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a**2 for a in v1))
    magnitude_v2 = math.sqrt(sum(a**2 for a in v2))
    return math.degrees(math.acos(dot_product / (magnitude_v1 * magnitude_v2)))

# 83. Get the nth root of a number
def nth_root(number, n):
    return number ** (1/n)

# 84. Calculate the midpoint of a line segment
def midpoint(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)

# 85. Generate a random list of integers
def random_int_list(size, start, end):
    return [random.randint(start, end) for _ in range(size)]

# 86. Merge two dictionaries
def merge_dicts(d1, d2):
    merged = d1.copy()
    merged.update(d2)
    return merged

# 87. Perform a binary search on a sorted list
def binary_search(lst, target):
    low, high = 0, len(lst) - 1
    while low <= high:
        mid = (low + high) // 2
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

# 88. Find the lowest common ancestor in a binary tree (placeholder)
def lowest_common_ancestor(root, n1, n2):
    # Placeholder, actual implementation depends on binary tree structure
    pass

# 89. Check if a string is a valid palindrome (ignores spaces and case)
def is_valid_palindrome(s):
    s = ''.join(filter(str.isalnum, s)).lower()
    return s == reverse_string(s)

# 90. Compute the area of a regular polygon
def regular_polygon_area(sides, side_length):
    return (sides * side_length ** 2) / (4 * math.tan(math.pi / sides))

# 91. Perform a depth-first search in a graph (placeholder)
def depth_first_search(graph, start):
    # Placeholder, actual implementation depends on graph structure
    pass

# 92. Check if a string is a valid bracket sequence (balanced brackets)
def is_balanced_brackets(s):
    stack = []
    matching_brackets = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in "({[":
            stack.append(char)
        elif char in ")}]":
            if not stack or stack[-1] != matching_brackets[char]:
                return False
            stack.pop()
    return not stack

# 93. Find the longest common subsequence in two strings
def longest_common_subsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# 94. Find the nth triangular number
def triangular_number(n):
    return n * (n + 1) // 2

# 95. Generate Pascal's triangle up to row n
def pascals_triangle(n):
    triangle = [[1]]
    for i in range(1, n):
        row = [1]
        for j in range(1, i):
            row.append(triangle[i-1][j-1] + triangle[i-1][j])
        row.append(1)
        triangle.append(row)
    return triangle

# 96. Calculate the harmonic sum of a list
def harmonic_sum(lst):
    return sum(1/x for x in lst if x != 0)

# 97. Find the sum of digits of a number
def sum_of_digits(number):
    return sum(int(digit) for digit in str(number))

# 98. Check if a string contains all unique characters
def has_unique_chars(string):
    return len(set(string)) == len(string)

# 99. Find the longest substring without repeating characters
def longest_unique_substring(s):
    seen = {}
    start = 0
    max_length = 0

    for i, char in enumerate(s):
        if char in seen and seen[char] >= start:
            start = seen[char] + 1
        seen[char] = i
        max_length = max(max_length, i - start + 1)
    return max_length

# 100. Count the number of words in a string
def word_count(string):
    return len(string.split())
