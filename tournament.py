import numpy as np
import string
from typing import List, Tuple


class TournamentSelections:
    def __init__(self,N:int,L:int,mu:float, p_c:float = 1.0) -> None:
        self.N=N
        self.L=L
        self.mu=mu
        self.p_c=p_c
        self.Sigma = [*string.ascii_lowercase+" "]
        self.population = self.generate_population()
        
    def generate_population(self) -> List[str]:
        """
        Generates N species (words) of lenght L.

        Returns:
            List[str]: All the words.
        """
        return list(map("".join, np.random.choice(self.Sigma, size=(self.N, self.L))))

    def k_tournament_selection(self, K:int, fitness_values:List[float]) -> str:
        """
        Performs local search in the K most optimal parents.

        Args:
            K (int): Number of parents to consider.
            fitness_values (List[float]): Fitness of the parents.

        Returns:
            str: The most optimal parent.
        """
        idx = np.random.choice(np.arange(self.N), size=K,replace=False)
        parents = np.array(self.population)[idx]    
        parent_fitness = np.array(fitness_values)[idx]

        best_fitness = -1
        if (largest := np.max(parent_fitness)) > best_fitness:
            best_fitness = largest
        best_parent = parents[np.where(parent_fitness==best_fitness)[0][0]]

        return best_parent
            
    def fitness(self,individual:str,target:str) -> float:
        """
        Measures the fitness of a candidate string compared to the output.

        Args:
            individual (str): The cadidate string.
            target (str): The target string.

        Returns:
            float: A number between 0 and 1. 1 means equal, 0 means no match.
        """
        length = max(len(individual),len(target))
        individual += ("."*(length-len(individual)))
        target += ("."*(length-len(target)))
        correct = 0
        for i,j in zip(individual,target):
            correct += i==j
        return correct/length
    

    def crossover(self, a:str, b:str, length:int) -> Tuple[str,str]:
        """
        Performs crossover between two words.

        Args:
            a (str): First word.
            b (str): Second word.
            length (int): The limit at which the crossover can be selected at random.

        Returns:
            Tuple[str,str]: The two words with some crossover
        """
        cross_point = np.random.choice(range(length))

        a1, a2 = a[:cross_point], a[cross_point:]
        b1, b2 = b[:cross_point], b[cross_point:]

        return a1+b2, b1+a2
    

    def mutate(self,individual:str) -> str:
        """
        Stochastically mutate a sample.

        Args:
            individual (str): The string to mutate

        Returns:
            str: The mutated (changed) string based on stochastic choice for words.
        """
        ind_list = np.array(list(individual))
        to_mutate = np.random.random(size=len(individual)) < self.mu
        N_cell_mutate = to_mutate.sum()
        ind_list[to_mutate] = np.random.choice(self.Sigma,size=N_cell_mutate)

        return "".join(ind_list)
    
    def tournament_selection_GA(self, target:str, G: int, K:int) -> Tuple[List[float],List[str]]:
        """
        Performs a tournament K evolution algorithm.

        Args:
            target (str): The target word to consider.
            G (int): The number of generations.
            K (int): The number of parents to consider.

        Returns:
            List[float,float]: The fitness history and the latest generation.
        """
        fitness_history = []
        old_fitness =  [self.fitness(individual=x,target=target) for x in self.population]
        fitness_history.append(max(old_fitness))

        for g in range(G):
            i = 0
            new_generation = []
            while self.N > i:
                if np.random.random() < self.p_c:
                    p1 = self.k_tournament_selection(K=K,fitness_values=old_fitness)
                    p2 = self.k_tournament_selection(K=K,fitness_values=old_fitness)
                    n1,n2 = self.crossover(a=p1,b=p2,length=len(target))
                    n1 = self.mutate(individual=n1)
                    n2 = self.mutate(individual=n2)

                    new_generation.append(n1)
                    new_generation.append(n2)

                    i+=2
                else:
                    p1 = self.k_tournament_selection(K=K,fitness_values=old_fitness)
                    n1 = self.mutate(individual=p1)
                    new_generation.append(n1)
                    i+=1
                
            new_fitness = [self.fitness(individual=x,target=target) for x in new_generation]
            fitness_history.append(max(new_fitness))

            if max(new_fitness) == 1:
                print(f"The target {target} found in {g} generations")
                return(fitness_history,new_generation)
            
            old_fitness = new_fitness
            self.population = new_generation
    
        return fitness_history,new_generation
