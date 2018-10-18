# ML Optimization a set of parameters in a mock information retrieval algorithm

In this exercise you will optimize a set of parameters in a mock information retrieval algorithm.


# Run the tets
```
pytest -- tests.py
```

# Run the optimization
```
 python solvemate_ml_exercise.py
```

## Answers to questions

### Task a
To keep it simple, I've applied the same logic to our problem than the f-score: the fitness function is a weighted harmonic mean.

See `fitness_func()` in `solvemate_ml_exercise.py`


### Task b
See `main()` in `solvemate_ml_exercise.py`
There are plenty of ways of tuning parameters. Commonly used are:
* grid search: we need to discretize the 3 parameters and then the complexity will be O(n3) so not efficient.
* randomized search: we can set the number of iterations but then it will be an approximation of the optimum.
* Stochastic Gradient Descent: we are not in a linear case. Some tricks are possible to avoid local maximum but it will anyway converge slower.

That's why I decided to come with a genetic algorithm approach.

### Task c
```
How would you adjust your algorithm to the case that the variables were weights? For example, if ​a, b​ and c​ ​ are models or algorithms which we wish to weight within a pipeline?
```
In this case the problem is different. We could think the problem linear and the loss of a linear function is convex, so we can use Stochastic Gradient Descent in an efficient way.

## Task d
```
How does the complexity of the problem increase in higher dimensions i.e. with more parameters?
```
The problem to optimize a nonlinear function with continuous parameter is NP-hard. Adding parameters to the problems keeps an NP-hard problem.
