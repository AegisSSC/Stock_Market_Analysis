import random

def estimate_pi(number_points):
    inside_circle = 0
    for _ in range (number_points):
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        if x ** 2 + y ** 2 <=1:
            inside_circle += 1
    return  4 * inside_circle/number_points

pi_estimate = estimate_pi(100000)

print("The Estimated Value of Pi is: ", pi_estimate)