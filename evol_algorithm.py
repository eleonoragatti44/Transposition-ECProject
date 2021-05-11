'''
Code inspired by Ernesto Costa's codes for Evolutionary Computation course
'''

from random import random, randint, sample, uniform, randrange
from operator import itemgetter
import numpy as np
import math

'''
UTILS
'''

# return the average of the population
def average_pop(population):
    fitness_vec = [indiv[1] for indiv in population]
    return sum(fitness_vec)/len(fitness_vec)

def read_init_pop(filename):
    pop_init = []
    cromo = []
    file = open(filename, 'r')
    lines = file.readlines()
    for line in lines:
        for i in line.split():
            cromo.append(int(i))
        pop_init.append((cromo,0))
        cromo = []
    return pop_init

def get_data(filename):
    data_raw = np.loadtxt(filename)
    data = data_raw.transpose()
    return data

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

'''
REPRESENTATION
'''

# return the lenght of the cromosome of a problem with a precise domain and precision
def cromo_len(max_domain, prec):
    approx_max_domain = int(max_domain * 10**(prec))
    max_domain_bin = bin(approx_max_domain).lstrip("0b") 
    len_max = len(max_domain_bin)
    return len_max
    
# Convert a float number into a binary string. 
# It multiplies the the float for 10^(precision), cuts the decimal part, and then applies the binary conversion.
def float_to_bin(number, max_domain, prec):
    # Defining the lenght of the cromosome
    approx_max_domain = int(max_domain * 10**(prec))
    max_domain_bin = bin(approx_max_domain).lstrip("0b") 
    len_max = len(max_domain_bin)
    sign = 0
    # Use prec to define max lenght for decimals
    approx_number = int(number * 10**(prec))
    if number < 0:
        sign += 1
        approx_number *= -1
    number_bin_str = bin(approx_number).lstrip("0b") 
    # We add the zeros at the beginning of the bin represented number
    len_numb = len(number_bin_str)
    diff_len = len_max - len_numb
    if diff_len > 0:
        for _ in range(diff_len):
            number_bin_str = '0' + number_bin_str
    if sign == 0:
        number_bin_str = '0' + number_bin_str
    else:
        number_bin_str = '1' + number_bin_str
    return(number_bin_str)

# convert a binary list to float.
def bin_to_float(numb, prec):
    if numb[0] == 0:
        sign = 1
    else:
        sign = -1
    numb = numb[1:]
    numb = ''.join(str(i) for i in numb)
    approx_numb = int(numb,2)/(10**prec)
    return(approx_numb * sign)

# return a list of float correspoding to the coordinate of the binary input element (geno).
def phenotype(geno, dimension, precision):
    pheno = []
    len_mono_cromo = int(len(geno)/dimension)
    for i in range(0, len(geno), len_mono_cromo):
        pheno.append(bin_to_float(geno[i:i+len_mono_cromo], precision))
    return pheno

# return a binary list of the float coordinate input
def genotype(feno, max_domain,precision):
    geno_str = ''
    for i in feno:
        geno_str += float_to_bin(i,max_domain, precision)
    geno = [int(geno_str[i]) for i in range(len(geno_str))]
    return geno
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
'''
STEP 0 - FITNESS
'''

def fit_rastrigin(dim, prec):
    '''
    rastrigin function
    domain = [-5.12, 5.12]
    minimum at (0,....,0)
    '''
    def rastrigin(indiv):
        X = phenotype(indiv, dim, prec)
        n = len(X)
        A = 10
        return A * n + sum([x**2 - A * math.cos(2 * math.pi * x) for x in X])
    return rastrigin

def fit_schwefel(dim, prec):
    '''
    schwefel function
    domain = [-500; 500]
    minimum at (420.9687,...,420.9687)
    '''
    def schwefel(indiv):
        X = phenotype(indiv, dim, prec)
        return sum([-x * math.sin(math.sqrt(math.fabs(x))) for x in X])
    return schwefel


def fit_quartic(dim, prec):
    '''
    quartic function
    domain = [-1.28; 1.28]
    minimum at (0,....,0)
    '''
    def quartic(indiv):
        X = phenotype(indiv, dim, prec)
        return sum([ (i+1) * x for i, x in enumerate(X)]) + random.uniform(0,1)
    return quartic
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

'''
STEP 1 - INITIALIZE POPULATION
'''

# Initialize population
def gera_pop(max_domain, precision, dimension):
    def pop(size_pop):
        return [(gera_indiv(max_domain, precision, dimension),0) for _ in range(size_pop)]
    return pop

def gera_indiv(max_domain, precision, dimension):
    indiv = ''
    for _ in range(dimension):
        # random initialization
        x = np.random.uniform(-max_domain, max_domain)
        # convert to bin
        indiv += float_to_bin(x, max_domain, precision)
    indiv = [int(indiv[i]) for i in range (len(indiv))]
    return indiv

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

'''
STEP 2 - PARENTS SELECTION
'''

# Parents Selection: tournament
def tour_sel(t_size):
    def tournament(pop):
        size_pop= len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = one_tour(pop,t_size)
            mate_pool.append(winner)
        return mate_pool
    return tournament

def one_tour(population,size):
    """Minimization Problem. Deterministic"""
    pool = sample(population, size)
    pool.sort(key=itemgetter(1))
    return pool[0]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

'''
STEP 3 - VARIATION OPERATOR
'''
# Binary mutation
def muta_bin(max_domain,dimension,prec):
    def mutation(indiv, prob_muta):
        # Mutation by gene
        old = indiv[:]
        cromo = indiv[:]
        len_cromo = int(len(indiv)) # cromosome length
        len_mono_cromo = int(len_cromo/dimension) # length of the single coordinate in bit    
        # jump from one mono_cromo to the next one
        # and applied the mutation to the last len_cromo/2 bits
        for j in range(0, len_cromo, len_mono_cromo):
            for i in range(1, round(len_mono_cromo/2)):
                cromo[len_mono_cromo+j-i] = muta_bin_gene(cromo[len_mono_cromo+j-i],prob_muta)
                pheno = phenotype(cromo,dimension,prec)
                # We control that the mutated cromosome does not correspond to a float value out from the domain
                for x in pheno:
                    if x > max_domain: 
                        cromo[i] = old[i]
        return cromo
    return mutation

def muta_bin_gene(gene, prob_muta):
    g = gene
    value = random()
    if value < prob_muta:
        g ^= 1
    return g

# Uniform crossover 
def uniform_cross(prob_cross, max_domain, precision, dimension):
    def crossover(indiv_1, indiv_2):
        value = random()
        if value < prob_cross:
            cromo_1 = indiv_1[0]
            cromo_2 = indiv_2[0]
            f1 = []
            f2 = []
            for i in range(0,len(cromo_1)):
                if random() < 0.5:
                    f1.append(cromo_1[i])
                    f2.append(cromo_2[i])
                else:
                    f1.append(cromo_2[i])
                    f2.append(cromo_1[i])
            pheno1 = phenotype(f1, dimension, precision)  
            pheno2 = phenotype(f2, dimension, precision)
            for i, val in enumerate(pheno1):
                if(val > max_domain):
                    pheno1[i] = max_domain
                if(val < - max_domain):
                    pheno1[i] = - max_domain
            for i, val in enumerate(pheno2):
                if(val > max_domain):
                    pheno2[i] = max_domain
                if(val < - max_domain):
                    pheno2[i] = - max_domain
            f1 = genotype(pheno1, max_domain,precision)
            f2 = genotype(pheno2, max_domain,precision)
            return ((f1,0),(f2,0))
        else:
            return (indiv_1,indiv_2)
    return crossover

# Transposition
def transposition(flank_size):
    def transpose(indiv_1, indiv_2):
        end = False
        start = False
        f1 = indiv_1[0]
        f2 = indiv_2[0]
        j = 0
        # define a random point in indiv_1
        size_indiv = len(f1)
        rnd_index = randrange(flank_size, size_indiv)
        flanking = f1[rnd_index-flank_size:rnd_index]
        # looking for the second flanking sequence in the first indiv
        while end == False:
            # if flanking==slice of indiv_1
            if (f1[ (rnd_index+j)%size_indiv : (rnd_index+j+flank_size)%size_indiv ] == flanking):
                end = True
                end_indiv_1 = (rnd_index+j+flank_size)%size_indiv # final index of the second flanking sequence
            else:
                j += 1
        j = 0
        while start == False and j <= size_indiv-flank_size:
            if (f2[j : j+flank_size] == flanking):
                start = True
                start_indiv_2 = j+flank_size # final index of the flanking sequence in the second indiv
            else:
                j += 1
        if end_indiv_1 == rnd_index or start == False:
            return indiv_1, indiv_2
        # now the transposition happens
        i = 0
        # repeat len(transposone) times
        while (rnd_index+i)%size_indiv != end_indiv_1:
            f1[(rnd_index+i)%size_indiv] = indiv_2[0][(start_indiv_2+i)%size_indiv]
            f2[(start_indiv_2+i)%size_indiv] = indiv_1[0][(rnd_index+i)%size_indiv]
            i += 1
        return ((f1,0),(f2,0))
    return transpose
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        
'''
STEP 4 - SURVIVALS SELECTION
'''

# Survivals Selection: elitism
def sel_survivors_elite(elite):
    def elitism(parents,offspring):
        size = len(parents)
        comp_elite = int(size* elite)
        offspring.sort(key=itemgetter(1))
        parents.sort(key=itemgetter(1))
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population
    return elitism

def best_pop(population):
    population.sort(key=itemgetter(1))
    return population[0]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

'''
EVOLUTIONARY ALGORITHMS
'''

# Return: best at the end - best by generation - average population by generation
def sea_same_pop(numb_generations, population, size_pop, prob_mut, sel_parents, recombination, mutation, sel_survivors, fitness_func):
    population = [(indiv[0], fitness_func(indiv[0])) for indiv in population]
    # For tatistics
    stat = [best_pop(population)[1]]
    stat_aver = [average_pop(population)]
    for j in range(numb_generations):
        print(j, end='\r')
        mate_pool = sel_parents(population)
        # Variation
        # ------ Crossover
        progenitores = []
        for i in  range(0, size_pop-1, 2):
            cromo_1 = mate_pool[i]
            cromo_2 = mate_pool[i+1]
            filhos = recombination(cromo_1, cromo_2)
            progenitores.extend(filhos) 
        # ------ Mutation
        descendentes = []
        for indiv,fit in progenitores:
            novo_indiv = mutation(indiv, prob_mut)
            descendentes.append((novo_indiv, fitness_func(novo_indiv)))
        population = sel_survivors(population, descendentes)
        population = [(indiv[0], fitness_func(indiv[0])) for indiv in population] 
        # Update statistics
        stat.append(best_pop(population)[1])
        stat_aver.append(average_pop(population))
    return best_pop(population), stat, stat_aver

# Run sea_same_pop, store results in 2 files
# file1: best over all + average for every generation
# file2: bast at the end of the run
def run_file(filename, numb_runs, numb_generations, population, size_pop, prob_mut, sel_parents, recombination, mutation,sel_survivors, fitness_func):
    statistics = []
    bea = []
    for i in range(numb_runs):
        print('run', i, end='\r')
        best, stat_best, stat_aver = sea_same_pop(numb_generations, population, size_pop, prob_mut, sel_parents, recombination, mutation, sel_survivors, fitness_func)
        statistics.append(stat_best)
        bea.append(best[1])
    stat_gener = list(zip(*statistics))
    boa = [min(g_i) for g_i in stat_gener] # minimization
    aver_gener =  [sum(g_i)/len(g_i) for g_i in stat_gener]
    stat_for_file = list(zip(*(boa,aver_gener)))
    np.savetxt(filename+'.dat', stat_for_file, delimiter='\t')
    np.savetxt(filename+'_bea.dat', bea, delimiter='\n')
