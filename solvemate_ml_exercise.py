import numpy as np
import random
import copy

np.random.seed(112)
izip = zip


def round_up_to_even(f):
    return int(np.ceil(f / 2.) * 2)


def grouped(iterable, n):
    '''
    Group a list so it's iterable by n like:
        for obj1, obj2 in grouped([1,2,3,4]):
    expect iterable % n == 0
    '''
    return izip(*[iter(iterable)]*n)


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

def fitness_func(r, p, q, alpha=1, beta=1, omega=1):
    '''
        The F-score is a weighted harmonic mean
        Here I will adapt the weighted harmonic mean with 3 variables.

        This fitness function will be optimized later: We will look for the
        parameters that maximimize or minimize the fitness function.

        Ideally we want to:
        - maximize precision p
        - maximize recall r
        - minimize number of questions asked p

        1 / x is a decreasing function. So this is the same than:
        - maximizing p, r and 1/q

        We can not have q = 0 then, which mean we have to ask at least one
        question. That make sense to me.
    '''
    Q_q = 1 / q
    return (
        (alpha + beta + omega) /
        ( (alpha/r) + (beta/p) + (omega/Q_q) )
    )


class Individual:
    '''
        Represents an individual of a population in a genetic algorithm problem
    '''

    @classmethod
    def create(cls):
        return cls(
            a=np.random.rand() * 1,
            b=np.random.rand() * 7,
            c=np.random.rand() * 5
        )

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.r, self.p, self.q, self.fitness = None, None, None, None

    def evaluate_fitness(self):
        self.__evaluate_rpq__()
        self.fitness = fitness_func(self.r, self.p, self.q)
        return self.fitness

    def __evaluate_rpq__(self):
            self.r, self.p, self.q = TestFunction().evaluate_parameters(
                a=self.a,
                b=self.b,
                c=self.c
            )
            return self.r, self.p, self.q

    @classmethod
    def crossover(cls, individual1, individual2):
        # init children param as copy of parents gen
        child1_params = {
            'a': individual1.a,
            'b': individual1.b,
            'c': individual1.c,
        }
        child2_params = {
            'a': individual2.a,
            'b': individual2.b,
            'c': individual2.c,
        }

        # handle crossover point: gen that will be exchanged
        crossover_point = np.random.randint(4)
        gen_list = ['a', 'b', 'c']
        random.shuffle(gen_list, np.random.rand)
        for idx, gen in enumerate(gen_list):
            if idx < crossover_point:
                child1_params[gen] = getattr(individual2, gen)
                child2_params[gen] = getattr(individual1, gen)

        children1 = cls(**child1_params)
        children2 = cls(**child2_params)

        # Mutation
        if np.random.rand() <= 0.5:
            children1.__mutate__()
        if np.random.rand() <= 0.5:
            children2.__mutate__()

        return children1, children2

    def __mutate__(self):
        mutation_probability = 0.33
        if np.random.rand() < mutation_probability:
            self.a = np.random.rand() * 1
        if np.random.rand() < mutation_probability:
            self.b = np.random.rand() * 7
        if np.random.rand() < mutation_probability:
            self.c = np.random.rand() * 5
        return self


class Population:
    '''
        Represents an ensemble of individuals in a genetic algorithm problem.
    '''
    def __init__(self, individuals):
        self.individuals = individuals

    def __sort_by_fitness__(self):
        return sorted(self.individuals, key=lambda individual: -individual.fitness)

    def top10(self):
        sorted_individuals = self.__sort_by_fitness__()
        size = len(sorted_individuals)
        return sorted_individuals[:round_up_to_even(size/10)]

    def replace_unable_by_fit(self, unable_idx, fit_individual):
        fit_individual.fitness = float('inf')
        unable_individual = self.individuals[unable_idx]
        self.individuals[unable_idx] = fit_individual
        return unable_individual

    def crossover(self):
        fittest_individuals = copy.copy(self.top10())
        random.shuffle(fittest_individuals, np.random.rand)

        idx = 0
        for fit1, fit2 in grouped(fittest_individuals, 2):
            fitter1, fitter2 = Individual.crossover(fit1, fit2)
            self.replace_unable_by_fit(-2*idx-1, fitter1)
            self.replace_unable_by_fit(-2*idx-2, fitter2)
            idx = idx + 1

        return self.individuals

    def evaluate_fitness(self):
        for individual in self.individuals:
            if individual.fitness is None or individual.fitness == float('inf'):
                individual.evaluate_fitness()
        self.individuals = self.__sort_by_fitness__()

    def has_converged(self):
        # does not produce offspring which are significantly different from the previous generation
        return False


def main():
    TF = TestFunction()
    # (a, b, c) = some set of parameters
    # 0 ≤ a ≤ 1
    # 0 ≤ b ≤ 7
    # 0 ≤ c ≤ 5
    # https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3
    population_size = 200
    population = Population(
        [Individual.create() for _ in range(population_size)]
    )
    population.evaluate_fitness()
    termination = False

    i = 0
    while i < 100 and termination is False:
        print('-'*100)
        print('iteration %s' % i)
        # selection & crossover & mutation
        population.crossover()

        # compute fitness
        population.evaluate_fitness()

        print(population.top10()[0].fitness)

        # termination
        i = i + 1
        termination = len({i.fitness for i in population.top10()}) == 1


if __name__=="__main__":
    main()
