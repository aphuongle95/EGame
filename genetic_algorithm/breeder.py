from game.individuals.dot import Dot

from random import choice, uniform, randint
from copy import copy

import numpy as np

class Breeder:
    def __init__(self, parent):
        self.parent = parent

    def breed(self, population):

        return self.breed_function(population)

    def initialize_population(self, num_individuals, color):
        """
        init function with each individuals having outstanding perception, ability and desire (centric value) in one out of the features. other features value are small random value
        """
        population = []
        for i in range(num_individuals):
            greater_than = 0.5
            less_than = 0.7
            centric_a = round(uniform(greater_than, less_than), 2)
            centric_b = round(uniform(greater_than, less_than), 2)
            centric_c = round(uniform(greater_than, less_than), 2)
            a = np.random.dirichlet(np.ones(5), size=1)[0] * (1-centric_a)#perception
            b = np.random.dirichlet(np.ones(4), size=1)[0] * (1-centric_b)#ability
            c = np.random.dirichlet(np.ones(5), size=1)[0] * (1-centric_c)#desire
            if(i%5==0):
                a = np.insert(a, 4, centric_a)
                b = np.insert(b, 4, centric_b)
                c = np.insert(c, 0, centric_c)
            elif(i%5==1):
                a = np.insert(a, 5, centric_a)
                b = np.insert(b, 0, centric_b)
                c = np.insert(c, 1, centric_c)
            elif(i%5==2):
                a = np.insert(a, 0, centric_a)
                b = np.insert(b, 1, centric_b)
                c = np.insert(c, 2, centric_c)
            elif(i%5==3):
                a = np.insert(a, 1, centric_a)
                b = np.insert(b, 2, centric_b)
                c = np.insert(c, 4, centric_c)
            elif(i%5==4):
                a = np.insert(a, 2, centric_a)
                b = np.insert(b, 3, centric_b)
                c = np.insert(c, 5, centric_c)
            else:
                a = np.random.dirichlet(np.ones(5), size=1)[0]
                b = np.random.dirichlet(np.ones(5), size=1)[0]
                c = np.random.dirichlet(np.ones(5), size=1)[0]

            population.append(self.assign_value(a, b, c, color=(0,139,139)))
        return population

    def assign_value(self, a, b, c, color):
        x = Dot(self.parent, color=color)
        x.perception.predator = a[0]
        x.perception.poison = a[1]
        x.perception.health_potion = a[2]
        x.perception.opponent = a[3] #
        x.perception.food = a[4]
        x.perception.corpse = a[5]

        x.abilities.armor_ability = b[0]
        x.abilities.speed = b[1]
        x.abilities.strength = b[2]
        x.abilities.poison_resistance = b[3]
        x.abilities.toxicity = b[4]

        x.desires.dodge_predators = c[0]
        x.desires.dodge_poison = c[1]
        x.desires.seek_potion = c[2]
        x.desires.seek_opponents = c[3]
        x.desires.seek_food = c[4]
        x.desires.seek_corpse = c[5]
        return x

    def breed_function(self, population):
        """
        breed function with our created crossover, tweak and mutation
        """
        population_cpy = copy(population)
        dead = []
        alive = []
        for individual in population_cpy:
            if individual.dead:
                dead.append(individual)
            else:
                alive.append(individual)

        for _ in range(len(dead)):
            where = choice(alive)._position
            color = alive[0].color

            selected = self.select(population_cpy)
            parent1 = selected[0]
            parent2 = selected[1]
            child1, child2 = self.crossover(copy(parent1), copy(parent2))
            child1 = self.tweak(child1)
            child2 = self.tweak(child2)
            score_child1 = self.assess_individual_fitness(child1)
            score_child2 = self.assess_individual_fitness(child2)
            if score_child1 > score_child2:
                new_individual = Dot(self.parent, color=color, position=where, dna=child1.get_dna())
                print(score_child1)
            else:
                new_individual = Dot(self.parent, color=color, position=where, dna=child2.get_dna())
            population_cpy.append(new_individual)
        for dead_individual in dead:
            population_cpy.remove(dead_individual)
        return population_cpy

    def select(self, population):
        """
        Creates the distribution to be used in the selectParentSUS: We chose not to use a Fitness Proportional Selection because it can have bad performance when a member of the population has a really large fitness in comparison with other members. Using a comb-like ruler, SUS starts from a small random number, and chooses the next candidates from the rest of population remaining, not allowing the fittest members to saturate the candidate space.
        """
        fitness_array = np.empty([len(population)])
        for i in range(len(population)):
            score = self.assess_individual_fitness(population[i])
            fitness_array[i] = score

        for i in range(1, len(fitness_array)):
            fitness_array[i] = fitness_array[i] + fitness_array[i - 1]

        parents = self.selectParentSUS(population, fitness_array, 2)
        return parents

    def assess_individual_fitness(self, individual):
        """
        The objective of the game is to last longer so the time_survived should be included in the fitness function. Eating food is also a factor that makes the individual last longer so we make a multiplication of them both. As our breed_function compares the fitness of the two children generated, if we only consider the time_survived and the food_eaten, both of them would have a fitness score of 0, so that's why we add dna genes that could make a individual has better chances to survive longer.
        """
        statistic = individual.statistic
        dna = individual.get_dna()
        score = (statistic.time_survived * statistic.food_eaten) + dna[0][0] + dna[1][0] + dna[2][2] + dna[2][4] + dna[1][5] + dna[0][5] + dna[0][3]
        return score

    def selectParentSUS(self, population, fitness_array, count):
        """
        Stochastic uniform sampling (The reason of working with this type of selection is commented on the select function)
        """
        individual_indices = []
        offset = uniform(0, fitness_array[-1] / count)
        for _ in range(count):
            index = 0

            while fitness_array[index] < offset:
                index += 1

            offset = offset + fitness_array[-1] / count
            individual_indices.append(population[index])

        return np.array(individual_indices)

    def crossover(self, solution_a, solution_b):
        """
        We just generate a random division k in one point that split the dna array of the parents in two blocks in order to make a crossover of them interchanging the first block of parent_a with the first of parent_b, and the same with the second blocks. We think that making this crossover by blocks is better because it exploites more group of genes that could be working well together and making fitter indivuduals.
        """
        dna_a = solution_a.get_dna()
        dna_b = solution_b.get_dna()
        k = randint(1, len(dna_a)-1)
        #swap with separator at k
        temp = copy(dna_b[0:k])
        dna_b[0:k] = copy(dna_a[0:k])
        dna_a[0:k] = copy(temp)
        solution_a.dna_to_traits(dna_a)
        solution_b.dna_to_traits(dna_b)
        return solution_a, solution_b

    def tweak(self, individual):
        """
        This funtions increases the highest gene value (best_) contained in the perception dna, desires dna and abilities dna adding a random value between 0 and 0,3. Because the dna must always sum 1, the increased value is decreased to the lowest gene value (worst_). We consider than exploiting and making the difference between the highest and the lowest gene value greater, will make the individual fitter.
        """
        dna = individual.get_dna()
        increase = uniform(0, 0.3)

        perc = dna[0]
        des = dna[1]
        abil = dna[2]

        best_perc = np.argmax(perc)
        best_des = np.argmax(des)
        best_abil = np.argmax(abil)

        worst_perc = np.argmin(perc)
        worst_des = np.argmin(des)
        worst_abil = np.argmin(abil)

        perc = self.mutate_dna(dna=perc, increase_value=increase, increase=best_perc, decrease=worst_perc)
        des = self.mutate_dna(dna=des, increase_value=increase, increase=best_des, decrease=worst_des)
        abil = self.mutate_dna(dna=abil, increase_value=increase, increase=best_abil, decrease=worst_abil)

        dna = [perc, des, abil]
        individual.dna_to_traits(dna)
        return individual

    def mutate_dna(self, dna, increase_value, increase, decrease):
        """
        None of the genes can take a value greater than 1. So, in order to make the mutation we have to check first if the highest gene value + the increased value, is over the limit or not. If it does then the highest value becomes 1. We then decrease the real increased value to the lowest gene value in the array.
        """
        if dna[increase] + increase_value > 1:
            increase_value = 1-dna[increase]
            dna[increase] = 1
        else:
            dna[increase] += increase_value

        dna[decrease] -= increase_value
        if dna[decrease] < 0:
            left_over=0-dna[decrease]
            dna[decrease] = 0
            choices = [i for i in range(len(dna))]
            print("before",choices)
            choices.remove(increase)
            choices.remove(decrease)
            print("after",choices)
            decrease2 = choice(choices)
            dna[decrease2]-=left_over
            if dna[decrease2] < 0:
                left_over2=0-dna[decrease2]
                dna[decrease2] = 0
                dna[increase]-=left_over2
        return dna
