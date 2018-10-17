import pytest
from solvemate_ml_exercise import Individual, Population
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


class TestPopulation:
    @pytest.fixture
    def population(self):
        population = Population(
            [Individual.create() for _ in range(100)]
        )
        list(map(lambda individual: individual.evaluate_fitness(), population.individuals))

        return population

    '''
        .replace_unable_by_fit()
    '''
    def test_replace_unable_by_fit(self, population):
        fit_individual = Individual.create()
        population.replace_unable_by_fit(42, fit_individual)
        assert population.individuals.index(fit_individual) == 42

    '''
        .top10()
    '''
    def test_top10_size(self, population):
        top10 = population.top10()

        assert len(top10) == 10

    def test_top10_order(self, population):
        for idx, individual in enumerate(population.individuals):
            individual.fitness = idx + 1

        top10 = population.top10()

        assert(
            list(
                map(lambda individual: individual.fitness, top10[:2])
            ) == [100, 99]
        )

    '''
        .crossover()
    '''
    def test_crossover(self, population):
        def to_fitness(individuals):
            return list(map(lambda individual: individual.fitness, individuals))
        original_fitnesses = to_fitness(population.individuals)
        new_fitnesses = to_fitness(population.crossover())
        assert len(set(original_fitnesses) - set(new_fitnesses)) == 10

    '''
        .evaluate_fitness()
    '''
    def test_evaluate_fitness(self, population):
        for individual in population.individuals:
            individual.fitness = None

        population.evaluate_fitness()

        assert len(list(filter(lambda individual: individual.fitness is None, population.individuals))) == 0
