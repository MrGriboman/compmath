import random as rn
import numpy as np


possible_number_of_rounds = tuple(range(2, 9))
BLANK = 0
REAL = 1


'''This function generates a sequence of rounds in random order that are to be loaded into the shotgun'''
def generate_rounds():
    n = rn.choice(possible_number_of_rounds)
    n = 3
    n_blanks = rn.choice(range(1, n))
    rounds = [BLANK for i in range(n_blanks)] + [REAL for i in range(n - n_blanks)]
    print(rounds)
    rn.shuffle(rounds)
    return rounds


print(generate_rounds())