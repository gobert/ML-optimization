import numpy as np


def gaussian(x, mu=0.5, sig=0.1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

class TestFunction:
    '''
    A mock information retrieval algorithm (chatbot) which takes 3 independent variables as
    input and returns the recall, the precision and the average questions asked
    0 ≤ a ≤ 1
    0 ≤ b ≤ 7
    0 ≤ c ≤ 5
    '''
    def __init__(self):
        self._min = 0
        self.a_max = 1
        self.b_max = 7
        self.c_max = 5

    def _check_a(self, a):
        if a < self._min or a > self.a_max:
            raise TypeError('a is must be within the bounds 0 and {a_max}'.format(a_max=self.a_max))

    def _check_b(self, b):
        if b < self._min or b > self.b_max:
            raise TypeError('b is must be within the bounds 0 and {b_max}'.format(b_max=self.b_max))

    def _check_c(self, c):
        if c < self._min or c > self.c_max:
            raise TypeError('c is must be within the bounds 0 and {c_max}'.format(c_max=self.c_max))

    def _test_function_a(self, x):
        self._check_a(x)
        (A_r, w_r) = (0.5, 10)
        (A_p, w_p) = (0.5, 3)

        r = A_r*np.sin(x*w_r) + 0.4*gaussian(x, mu=0.6, sig=0.3) + 0.1
        p = A_p*np.sin(x*w_p-0.1) + 0.5
        q = 2*np.sin(x*20) + 3*np.sin(x*5) + 10

        return r, p, q

    def _test_function_b(self, x):
        self._check_b(x)
        A_r = 0.5
        A_p = 0.7

        r = A_r*np.sin(x*0.9) + 0.9*gaussian(x, mu=5, sig=0.6) + 0.3
        p = A_p*np.sin(x*0.4) + 0.1*np.sin(x*5)
        q = 2*np.sin(x) + 2*np.sin(x*2) + 8

        return r, p, q

    def _test_function_c(self, x):
        self._check_c(x)
        (A_r, w_r) = (0.2, 3)
        (A_p, w_p) = (0.3, 6)

        r = A_r*np.sin(x*w_r) + 0.5 + 0.3*gaussian(x, mu=1.5, sig=1)
        p = (A_p/2)*np.sin(x*3 + 1) + (A_p/5)*np.sin(x*20) + A_p*np.sin(x*0.5) + 0.3
        q = 2*np.sin(x*4) - np.sin(x*1) + 7

        return r, p, q

    def evaluate_parameters(self, a, b, c):
        r_a, p_a, q_a = self._test_function_a(a)
        r_b, p_b, q_b = self._test_function_b(b)
        r_c, p_c, q_c = self._test_function_c(c)

        recall = (r_a + r_b + r_c)/3
        precision = (p_a + p_b + p_c)/3
        mean_questions = (q_a + q_b + q_c)/3

        return recall, precision, mean_questions

def fitness_func(r, p, q):
    # insert fitness function logic here.
    return fitness

def main():
    TF = TestFunction()
    # (a, b, c) = some set of parameters
    r, p, q = TF.evaluate_parameters(a, b, c)
    fitness = fitness_func(r, p, q)


if __name__=="__main__":
    main()
