import pytest
from solvemate_ml_exercise import Individual
from solvemate_ml_exercise import np


class TestIndividual:
    '''
        Based on the stochastic behavior of the class individual, we will run
        the assertion on 100 individuals and ensure that it works for all of
        them.
    '''

    '''
        .create()
    '''
    @pytest.fixture
    def individuals(self):
        return [Individual.create() for _ in range(100)]

    def test_create_a_bigger_than_0(self, individuals):
        def bigger_than_0(individual):
            return individual.a >= 0
        assert len(list(filter(bigger_than_0, individuals))) == len(individuals)

    def test_create_a_smaller_than_1(self, individuals):
        def smaller_than_1(individual):
            return individual.a <= 1
        assert len(list(filter(smaller_than_1, individuals))) == len(individuals)

    def test_create_b_bigger_than_0(self, individuals):
        def bigger_than_0(individual):
            return individual.b >= 0
        assert len(list(filter(bigger_than_0, individuals))) == len(individuals)

    def test_create_b_smaller_than_7(self, individuals):
        def smaller_than_7(individual):
            return individual.b <= 7
        assert len(list(filter(smaller_than_7, individuals))) == len(individuals)

    def test_create_c_bigger_than_0(self, individuals):
        def bigger_than_0(individual):
            return individual.c >= 0
        assert len(list(filter(bigger_than_0, individuals))) == len(individuals)

    def test_create_c_smaller_than_5(self, individuals):
        def smaller_than_5(individual):
            return individual.c <= 5
        assert len(list(filter(smaller_than_5, individuals))) == len(individuals)

    '''
        .crossover()
    '''
    def test_evaluate_crossover_return(self):
        parent1, parent2 = [Individual.create(), Individual.create()]
        children1, children2 = Individual.create().crossover(parent1, parent2)
        assert type(children1) == Individual
        assert type(children2) == Individual


    '''
        .evaluate_fitness()
    '''
    def test_evaluate_fitness_set_fitness(self):
        individual = Individual.create()
        individual.evaluate_fitness()
        assert type(individual.fitness) == np.float64

    def test_evaluate_fitness_return(self):
        individual = Individual.create()
        assert type(individual.evaluate_fitness()) == np.float64
