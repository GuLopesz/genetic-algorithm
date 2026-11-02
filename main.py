import random
import math
import pandas as pd
import matplotlib.pyplot as plt

def load_cities(file_path):
    dataframe = pd.read_csv(file_path, decimal=',')
    cities = {}
    for _, row in dataframe.iterrows():
        cities[int(row['cidade'])] = (float(row['x']), float(row['y']))
    return cities

def create_initial_population(population_size, city_ids):
    population = []
    for _ in range(population_size):
        individual = list(city_ids)
        random.shuffle(individual)
        population.append(individual)
    return population

def selection_and_evaluation(population, cities, tournament_size=3):
    evaluated_population = []
    for individual in population:
        total_distance = 0.0
        num_cities = len(individual)
        for i in range(num_cities):
            current_city = cities[individual[i]]
            next_city = cities[individual[(i + 1) % num_cities]]
            distance = math.hypot(next_city[0] - current_city[0], next_city[1] - current_city[1])
            total_distance += distance
        evaluated_population.append((individual, total_distance))
    
    evaluated_population.sort(key=lambda item: item[1])
    best_individual, best_distance = evaluated_population[0]
    
    selected_parents = []
    for _ in range(len(population)):
        competitors = random.sample(evaluated_population, tournament_size)
        winner = min(competitors, key=lambda item: item[1])[0]
        selected_parents.append(list(winner))
        
    return selected_parents, list(best_individual), best_distance

def single_point_crossover(parent1, parent2):
    cut_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:cut_point] + [city for city in parent2 if city not in parent1[:cut_point]]
    child2 = parent2[:cut_point] + [city for city in parent1 if city not in parent2[:cut_point]]
    return child1, child2

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

def plot_best_individual(cities, best_individual, best_distance, generation):
    plt.clf()
    x_coords = [cities[city][0] for city in best_individual] + [cities[best_individual[0]][0]]
    y_coords = [cities[city][1] for city in best_individual] + [cities[best_individual[0]][1]]
    
    plt.plot(x_coords, y_coords, 'b-o', markersize=8, linewidth=1.5)
    for city_id, coords in cities.items():
        plt.annotate(
            str(city_id), 
            (coords[0], coords[1]), 
            textcoords="offset points", 
            xytext=(0, 8), 
            ha='center',
            fontsize=9,
            fontweight='bold'
        )
        
    plt.title(f"Geracao: {generation} | Melhor Distancia: {best_distance:.4f}\nRota: {best_individual}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.pause(0.001)

def run_genetic_algorithm():
    file_path = 'exemplo2.csv'
    population_size = 100
    generations = 2000
    mutation_rate = 0.10
    
    cities = load_cities(file_path)
    city_ids = list(cities.keys())
    population = create_initial_population(population_size, city_ids)
    
    plt.ion()
    plt.figure(figsize=(8, 6))
    
    for generation in range(1, generations + 1):
        selected_parents, best_individual, best_distance = selection_and_evaluation(population, cities)
        
        print(f"Geracao {generation:04d} | Melhor Distancia: {best_distance:.4f} | Melhor Rota: {best_individual}")
        
        if generation % 20 == 0 or generation == 1 or generation == generations:
            plot_best_individual(cities, best_individual, best_distance, generation)
        
        next_population = []
        for i in range(0, population_size, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            child1, child2 = single_point_crossover(parent1, parent2)
            next_population.append(mutate(child1, mutation_rate))
            next_population.append(mutate(child2, mutation_rate))
            
        population = next_population
        
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_genetic_algorithm()