import numpy as np
import pandas as pd
import random

# Define synthetic fraud dataset generation with Genetic Algorithm
class GeneticAlgorithmFraudGenerator:
    def __init__(self, population_size, generations, mutation_rate):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            individual = {
                "transaction_amount": random.uniform(10, 1000),  # Amount in USD
                "location": random.choice(["US", "UK", "CA", "IN", "DE", "AU"]),
                "payment_method": random.choice(["Credit Card", "Debit Card", "PayPal", "Bitcoin"]),
                "address_match": random.choice(["Match", "Mismatch"]),
                "items": random.randint(1, 10),
                "device": random.choice(["Desktop", "Mobile", "Tablet"]),
                "proxy_used": random.choice(["Yes", "No"]),
            }
            population.append(individual)
        return population

    def fitness_function(self, individual):
        # Define fitness based on common fraud patterns
        fitness = 0
        if individual["transaction_amount"] > 500:  # High-value transaction
            fitness += 1
        if individual["address_match"] == "Mismatch":  # Billing and shipping mismatch
            fitness += 1
        if individual["proxy_used"] == "Yes":  # Proxy use
            fitness += 1
        if individual["payment_method"] == "Bitcoin":  # Rare payment method
            fitness += 1
        return fitness

    def selection(self, population, fitness_scores):
        # Select individuals based on fitness (roulette wheel selection)
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        return random.choices(population, weights=probabilities, k=self.population_size // 2)

    def crossover(self, parent1, parent2):
        # Combine two parents to create an offspring
        offspring = {}
        for key in parent1:
            offspring[key] = random.choice([parent1[key], parent2[key]])
        return offspring

    def mutate(self, individual):
        # Randomly mutate some attributes based on mutation rate
        if random.random() < self.mutation_rate:
            individual["transaction_amount"] = random.uniform(10, 1000)
        if random.random() < self.mutation_rate:
            individual["address_match"] = random.choice(["Match", "Mismatch"])
        if random.random() < self.mutation_rate:
            individual["proxy_used"] = random.choice(["Yes", "No"])
        return individual

    def evolve_population(self, population):
        fitness_scores = [self.fitness_function(ind) for ind in population]
        selected_individuals = self.selection(population, fitness_scores)
        next_generation = []

        # Perform crossover
        for _ in range(self.population_size):
            parent1, parent2 = random.sample(selected_individuals, 2)
            offspring = self.crossover(parent1, parent2)
            offspring = self.mutate(offspring)
            next_generation.append(offspring)

        return next_generation

    def generate_dataset(self):
        population = self.initialize_population()
        for _ in range(self.generations):
            population = self.evolve_population(population)
        return pd.DataFrame(population)


# Parameters
population_size = 100
generations = 10
mutation_rate = 0.1

# Generate synthetic dataset
ga_fraud_generator = GeneticAlgorithmFraudGenerator(population_size, generations, mutation_rate)
synthetic_data = ga_fraud_generator.generate_dataset()

# Display the synthetic dataset
synthetic_data.head()
