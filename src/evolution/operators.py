import random
from copy import deepcopy

from src.evolution.search_space import SEARCH_SPACE
from src.evolution.individual import apply_constraints


def mutate_individual(cfg: dict, mutation_rate: float = 0.2) -> dict:
    child = deepcopy(cfg)

    for key, values in SEARCH_SPACE.items():
        if random.random() < mutation_rate:
            child[key] = random.choice(values)

    return apply_constraints(child)


def crossover_individuals(parent1: dict, parent2: dict) -> dict:
    child = {}
    for key in SEARCH_SPACE.keys():
        child[key] = random.choice([parent1[key], parent2[key]])
    return apply_constraints(child)