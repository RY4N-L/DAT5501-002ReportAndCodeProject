## Unit tests using unittest module ##

import unittest
from src.function_example import add_numbers

class TestFunction (unittest.TestCase):
    
    def test_addition(self):
    # Test the addition of positive numbers
        result = add_numbers(3,5)
        self.assertEqual(result, 8)

    def test_addition_negative_numbers(self):
    # Test the addition of negative numbers
        result = add_numbers(-2,7)
        self.assertEqual(result, 5)

    def test_addition_float_numbers(self):
    # Test the addition of floating point numbers to 2 decimal places
        result  = add_numbers(1.5, 2.5)
        self.assertAlmostEqual(result, 4.0, places = 2)

if __name__ == '__main__':
    unittest.main()