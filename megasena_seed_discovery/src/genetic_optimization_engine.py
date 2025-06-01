#!/usr/bin/env python3
"""
Motor de Otimização Genética
Implementa busca exaustiva de seeds usando algoritmos genéticos e MCMC
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn.preprocessing import StandardScaler
import json
import random
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas de otimização
try:
    import emcee
    EMCEE_AVAILABLE = True
except ImportError:
    EMCEE_AVAILABLE = False
    print("⚠️ emcee não disponível")

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("⚠️ PyMC não disponível")

class GeneticOptimizationEngine:
    def __init__(self, historical_data, prng_candidates=None):
        self.historical_data = historical_data
        self.prng_candidates = prng_candidates or []
        self.optimization_results = {}
        self.best_solutions = []
        
        # Parâmetros do algoritmo genético
        self.ga_params = {
            'population_size': 100,
            'generations': 500,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elite_size': 10,
            'tournament_size': 5
        }
        
        # Parâmetros MCMC
        self.mcmc_params = {
            'n_walkers': 32,
            'n_steps': 1000,
            'burn_in': 200
        }
    
    def optimize_all_candidates(self):
        """Otimiza todos os candidatos PRNG detectados"""
        print("\n🧬 Iniciando Otimização Genética...")
        
        results = {}
        
        if not self.prng_candidates:
            print("   ⚠️ Nenhum candidato PRNG fornecido, executando busca genérica...")
            results['generic'] = self.generic_genetic_search()
        else:
            # Otimizar cada candidato PRNG
            for i, candidate in enumerate(self.prng_candidates[:5]):  # Top 5
                prng_type = candidate.get('type', f'candidate_{i}')
                print(f"\n   🎯 Otimizando {prng_type}...")
                
                if prng_type == 'LCG':
                    results[prng_type] = self.optimize_lcg_genetic(candidate)
                elif prng_type == 'LFSR':
                    results[prng_type] = self.optimize_lfsr_genetic(candidate)
                elif prng_type == 'Xorshift':
                    results[prng_type] = self.optimize_xorshift_genetic(candidate)
                elif prng_type == 'Mersenne_Twister':
                    results[prng_type] = self.optimize_mt_genetic(candidate)
                else:
                    results[prng_type] = self.optimize_generic_genetic(candidate)
        
        # MCMC para refinamento
        if EMCEE_AVAILABLE:
            print("\n   🔬 Refinamento com MCMC...")
            results['mcmc_refinement'] = self.mcmc_optimization()
        
        # Busca híbrida
        print("\n   🔄 Busca híbrida...")
        results['hybrid_search'] = self.hybrid_optimization()
        
        self.optimization_results = results
        self.extract_best_solutions()
        
        return results
    
    def optimize_lcg_genetic(self, candidate):
        """Otimização genética específica para LCG"""
        print("     Otimizando parâmetros LCG...")
        
        # Parâmetros a otimizar: [seed, a, c, m_index]
        # m_index aponta para valores conhecidos de m
        m_values = [2**31 - 1, 2**32, 2**31, 4294967296]
        
        def fitness_function(individual):
            seed, a, c, m_index = individual
            seed = int(seed) % (2**31)
            a = int(a) % (2**31)
            c = int(c) % (2**31)
            m_index = int(m_index) % len(m_values)
            m = m_values[m_index]
            
            return self.evaluate_lcg_fitness(seed, a, c, m)
        
        # Bounds para parâmetros
        bounds = [
            (1, 2**31 - 1),  # seed
            (1, 2**16),      # a
            (0, 2**16),      # c
            (0, len(m_values) - 1)  # m_index
        ]
        
        # Executar algoritmo genético
        best_individual, best_fitness, evolution = self.genetic_algorithm(
            fitness_function, bounds, maximize=True
        )
        
        # Interpretar resultado
        seed, a, c, m_index = best_individual
        m = m_values[int(m_index) % len(m_values)]
        
        return {
            'best_parameters': {
                'seed': int(seed),
                'a': int(a),
                'c': int(c),
                'm': m
            },
            'fitness': best_fitness,
            'evolution': evolution,
            'validation_score': self.validate_lcg_solution(int(seed), int(a), int(c), m)
        }
    
    def evaluate_lcg_fitness(self, seed, a, c, m):
        """Avalia fitness de parâmetros LCG"""
        try:
            # Gerar sequência LCG
            generated_sequence = self.generate_lcg_sequence(seed, a, c, m, min(500, len(self.historical_data)))
            
            # Converter para range da loteria
            lottery_sequence = self.convert_to_lottery_sums(generated_sequence)
            
            # Sequência real
            real_sequence = [sum(draw['numbers']) for draw in self.historical_data[:len(lottery_sequence)]]
            
            if len(lottery_sequence) != len(real_sequence):
                return 0
            
            # Múltiplas métricas de fitness
            correlation = self.calculate_correlation(real_sequence, lottery_sequence)
            ks_similarity = 1 - stats.ks_2samp(real_sequence, lottery_sequence)[0]
            variance_similarity = 1 - abs(np.var(real_sequence) - np.var(lottery_sequence)) / max(np.var(real_sequence), 1)
            
            # Penalizar parâmetros inválidos
            if a <= 0 or m <= 0 or seed <= 0:
                return 0
            
            # Fitness combinado
            fitness = (0.5 * correlation + 0.3 * ks_similarity + 0.2 * variance_similarity)
            
            return max(0, fitness)
            
        except:
            return 0
    
    def generate_lcg_sequence(self, seed, a, c, m, length):
        """Gera sequência LCG"""
        sequence = []
        x = seed
        
        for _ in range(length):
            x = (a * x + c) % m
            sequence.append(x)
        
        return sequence
    
    def convert_to_lottery_sums(self, sequence):
        """Converte sequência para somas da loteria"""
        if not sequence:
            return []
        
        # Normalizar para range 0-1
        min_val, max_val = min(sequence), max(sequence)
        if max_val == min_val:
            return [200] * len(sequence)
        
        normalized = [(x - min_val) / (max_val - min_val) for x in sequence]
        
        # Mapear para range da Mega Sena (aproximadamente 80-300)
        lottery_sums = [80 + norm * 220 for norm in normalized]
        
        return lottery_sums
    
    def calculate_correlation(self, seq1, seq2):
        """Calcula correlação entre sequências"""
        if len(seq1) != len(seq2) or len(seq1) < 2:
            return 0
        
        try:
            correlation, _ = stats.pearsonr(seq1, seq2)
            return abs(correlation) if not np.isnan(correlation) else 0
        except:
            return 0
    
    def validate_lcg_solution(self, seed, a, c, m):
        """Valida solução LCG"""
        # Gerar sequência mais longa para validação
        long_sequence = self.generate_lcg_sequence(seed, a, c, m, len(self.historical_data))
        lottery_sequence = self.convert_to_lottery_sums(long_sequence)
        
        real_sequence = [sum(draw['numbers']) for draw in self.historical_data]
        
        # Métricas de validação
        validation_metrics = {
            'correlation': self.calculate_correlation(real_sequence, lottery_sequence),
            'mean_difference': abs(np.mean(real_sequence) - np.mean(lottery_sequence)),
            'std_difference': abs(np.std(real_sequence) - np.std(lottery_sequence)),
            'ks_statistic': stats.ks_2samp(real_sequence, lottery_sequence)[0],
            'periodicity_score': self.check_periodicity(lottery_sequence)
        }
        
        # Score de validação combinado
        validation_score = (
            validation_metrics['correlation'] * 0.4 +
            (1 - min(1, validation_metrics['mean_difference'] / 50)) * 0.2 +
            (1 - min(1, validation_metrics['std_difference'] / 50)) * 0.2 +
            (1 - validation_metrics['ks_statistic']) * 0.1 +
            validation_metrics['periodicity_score'] * 0.1
        )
        
        return {
            'metrics': validation_metrics,
            'combined_score': validation_score
        }
    
    def check_periodicity(self, sequence):
        """Verifica periodicidade da sequência"""
        # Procurar por períodos
        max_period = min(1000, len(sequence) // 3)
        
        for period in range(2, max_period):
            if period * 3 > len(sequence):
                break
            
            # Comparar início com repetições
            segment1 = sequence[:period]
            segment2 = sequence[period:2*period]
            segment3 = sequence[2*period:3*period]
            
            if len(segment1) == len(segment2) == len(segment3):
                corr12 = self.calculate_correlation(segment1, segment2)
                corr23 = self.calculate_correlation(segment2, segment3)
                
                if corr12 > 0.9 and corr23 > 0.9:
                    return 1.0  # Periodicidade forte detectada
        
        return 0.0  # Nenhuma periodicidade óbvia
    
    def optimize_lfsr_genetic(self, candidate):
        """Otimização genética para LFSR"""
        print("     Otimizando parâmetros LFSR...")
        
        def fitness_function(individual):
            polynomial, initial_state = individual
            polynomial = int(polynomial) % (2**16)
            initial_state = int(initial_state) % (2**16)
            
            if polynomial <= 0 or initial_state <= 0:
                return 0
            
            return self.evaluate_lfsr_fitness(polynomial, initial_state)
        
        bounds = [
            (1, 2**16 - 1),  # polynomial
            (1, 2**16 - 1)   # initial_state
        ]
        
        best_individual, best_fitness, evolution = self.genetic_algorithm(
            fitness_function, bounds, maximize=True
        )
        
        polynomial, initial_state = best_individual
        
        return {
            'best_parameters': {
                'polynomial': int(polynomial),
                'initial_state': int(initial_state)
            },
            'fitness': best_fitness,
            'evolution': evolution
        }
    
    def evaluate_lfsr_fitness(self, polynomial, initial_state):
        """Avalia fitness LFSR"""
        try:
            # Gerar sequência binária LFSR
            binary_sequence = self.generate_lfsr_sequence(polynomial, initial_state, 3000)
            
            # Converter para números da loteria
            lottery_numbers = self.binary_to_lottery_numbers(binary_sequence)
            
            if len(lottery_numbers) < 100:
                return 0
            
            # Comparar com dados reais
            real_binary = self.historical_to_binary()
            real_numbers = self.binary_to_lottery_numbers(real_binary)
            
            min_length = min(len(lottery_numbers), len(real_numbers))
            
            # Fitness baseado em similaridade binária
            binary_fitness = self.calculate_binary_similarity(
                binary_sequence[:min_length*6], 
                real_binary[:min_length*6]
            )
            
            return binary_fitness
            
        except:
            return 0
    
    def generate_lfsr_sequence(self, polynomial, initial_state, length):
        """Gera sequência LFSR"""
        state = initial_state
        sequence = []
        
        for _ in range(length):
            # XOR feedback baseado no polinômio
            feedback = 0
            temp_poly = polynomial
            temp_state = state
            
            while temp_poly > 0:
                if temp_poly & 1:
                    feedback ^= temp_state & 1
                temp_poly >>= 1
                temp_state >>= 1
            
            # Shift e adicionar feedback
            state = (state >> 1) | (feedback << 15)
            sequence.append(state & 1)
        
        return sequence
    
    def binary_to_lottery_numbers(self, binary_sequence):
        """Converte sequência binária para números da loteria"""
        numbers = []
        
        # Agrupar em blocos de 6 bits (para números 1-60)
        for i in range(0, len(binary_sequence) - 5, 6):
            block = binary_sequence[i:i+6]
            
            # Converter para inteiro
            value = 0
            for j, bit in enumerate(block):
                value += bit * (2 ** (5-j))
            
            # Mapear para range 1-60
            number = (value % 60) + 1
            numbers.append(number)
        
        return numbers
    
    def historical_to_binary(self):
        """Converte dados históricos para binário"""
        binary_sequence = []
        
        for draw_info in self.historical_data[:500]:  # Limitar para performance
            for number in draw_info['numbers']:
                # Converter para binário (6 bits)
                binary = format(number, '06b')
                binary_sequence.extend([int(b) for b in binary])
        
        return binary_sequence
    
    def calculate_binary_similarity(self, seq1, seq2):
        """Calcula similaridade entre sequências binárias"""
        if len(seq1) != len(seq2) or len(seq1) == 0:
            return 0
        
        matches = sum(1 for i in range(len(seq1)) if seq1[i] == seq2[i])
        return matches / len(seq1)
    
    def optimize_xorshift_genetic(self, candidate):
        """Otimização genética para Xorshift"""
        print("     Otimizando parâmetros Xorshift...")
        
        def fitness_function(individual):
            seed, a, b, c = individual
            seed = int(seed) % (2**32)
            a = int(a) % 32
            b = int(b) % 32
            c = int(c) % 32
            
            if seed <= 0 or a <= 0 or b <= 0 or c <= 0:
                return 0
            
            return self.evaluate_xorshift_fitness(seed, a, b, c)
        
        bounds = [
            (1, 2**31),  # seed
            (1, 31),     # a
            (1, 31),     # b
            (1, 31)      # c
        ]
        
        best_individual, best_fitness, evolution = self.genetic_algorithm(
            fitness_function, bounds, maximize=True
        )
        
        return {
            'best_parameters': {
                'seed': int(best_individual[0]),
                'a': int(best_individual[1]),
                'b': int(best_individual[2]),
                'c': int(best_individual[3])
            },
            'fitness': best_fitness,
            'evolution': evolution
        }
    
    def evaluate_xorshift_fitness(self, seed, a, b, c):
        """Avalia fitness Xorshift"""
        try:
            generated_sequence = self.generate_xorshift_sequence(seed, a, b, c, 500)
            lottery_sequence = self.convert_to_lottery_sums(generated_sequence)
            
            real_sequence = [sum(draw['numbers']) for draw in self.historical_data[:len(lottery_sequence)]]
            
            return self.calculate_correlation(real_sequence, lottery_sequence)
            
        except:
            return 0
    
    def generate_xorshift_sequence(self, seed, a, b, c, length):
        """Gera sequência Xorshift"""
        x = seed
        sequence = []
        
        for _ in range(length):
            x ^= x << a
            x ^= x >> b
            x ^= x << c
            x &= 0xFFFFFFFF
            sequence.append(x)
        
        return sequence
    
    def optimize_mt_genetic(self, candidate):
        """Otimização genética para Mersenne Twister (simplificada)"""
        print("     Otimizando estado inicial MT...")
        
        # MT é muito complexo, otimizar apenas estado inicial
        def fitness_function(individual):
            seed = int(individual[0]) % (2**32)
            return self.evaluate_mt_fitness(seed)
        
        bounds = [(1, 2**32 - 1)]
        
        best_individual, best_fitness, evolution = self.genetic_algorithm(
            fitness_function, bounds, maximize=True
        )
        
        return {
            'best_parameters': {
                'seed': int(best_individual[0])
            },
            'fitness': best_fitness,
            'evolution': evolution,
            'note': 'MT optimization simplified - only initial seed'
        }
    
    def evaluate_mt_fitness(self, seed):
        """Avalia fitness MT (usando numpy.random como proxy)"""
        try:
            # Usar numpy.random como aproximação do MT
            np.random.seed(seed % (2**32))
            
            # Gerar sequência
            generated = []
            for _ in range(min(500, len(self.historical_data))):
                # Simular geração de 6 números
                numbers = []
                for _ in range(6):
                    num = (np.random.randint(0, 2**32) % 60) + 1
                    if num not in numbers:
                        numbers.append(num)
                
                if len(numbers) == 6:
                    generated.append(sum(sorted(numbers)))
            
            real_sequence = [sum(draw['numbers']) for draw in self.historical_data[:len(generated)]]
            
            return self.calculate_correlation(real_sequence, generated)
            
        except:
            return 0
    
    def optimize_generic_genetic(self, candidate):
        """Otimização genética genérica"""
        print("     Otimização genérica...")
        
        # Busca de padrão genérico com múltiplos parâmetros
        def fitness_function(individual):
            return self.evaluate_generic_fitness(individual)
        
        # Parâmetros genéricos: [param1, param2, param3, param4]
        bounds = [
            (1, 2**16),
            (0, 2**16),
            (1, 255),
            (1, 255)
        ]
        
        best_individual, best_fitness, evolution = self.genetic_algorithm(
            fitness_function, bounds, maximize=True
        )
        
        return {
            'best_parameters': {
                'param1': int(best_individual[0]),
                'param2': int(best_individual[1]),
                'param3': int(best_individual[2]),
                'param4': int(best_individual[3])
            },
            'fitness': best_fitness,
            'evolution': evolution
        }
    
    def evaluate_generic_fitness(self, individual):
        """Fitness genérico baseado em padrões matemáticos"""
        try:
            p1, p2, p3, p4 = [int(x) for x in individual]
            
            # Gerar sequência usando função matemática genérica
            sequence = []
            x = p1
            
            for i in range(min(500, len(self.historical_data))):
                # Função complexa genérica
                x = ((x * p2) + p3) % (p4 * 1000 + 1)
                x = x ^ (x >> 3)
                x = x ^ (x << 7)
                
                # Mapear para range da loteria
                lottery_sum = (x % 220) + 80
                sequence.append(lottery_sum)
            
            real_sequence = [sum(draw['numbers']) for draw in self.historical_data[:len(sequence)]]
            
            return self.calculate_correlation(real_sequence, sequence)
            
        except:
            return 0
    
    def generic_genetic_search(self):
        """Busca genética genérica sem candidatos específicos"""
        print("     Executando busca genética genérica...")
        
        def fitness_function(individual):
            # Interpretar individual como parâmetros de algoritmo desconhecido
            seed, mult, add, mod_factor, shift1, shift2 = [int(x) for x in individual]
            
            return self.evaluate_unknown_algorithm(seed, mult, add, mod_factor, shift1, shift2)
        
        bounds = [
            (1, 2**20),      # seed
            (1, 2**16),      # multiplier
            (0, 2**16),      # additive
            (1000, 10000),   # modulus factor
            (1, 31),         # shift1
            (1, 31)          # shift2
        ]
        
        best_individual, best_fitness, evolution = self.genetic_algorithm(
            fitness_function, bounds, maximize=True, generations=300
        )
        
        return {
            'algorithm': 'unknown_complex',
            'best_parameters': {
                'seed': int(best_individual[0]),
                'multiplier': int(best_individual[1]),
                'additive': int(best_individual[2]),
                'modulus_factor': int(best_individual[3]),
                'shift1': int(best_individual[4]),
                'shift2': int(best_individual[5])
            },
            'fitness': best_fitness,
            'evolution': evolution
        }
    
    def evaluate_unknown_algorithm(self, seed, mult, add, mod_factor, shift1, shift2):
        """Avalia algoritmo desconhecido"""
        try:
            sequence = []
            state = seed
            
            for _ in range(min(500, len(self.historical_data))):
                # Algoritmo complexo hipotético
                state = (state * mult + add) % (mod_factor * 1000)
                state = state ^ (state >> shift1)
                state = state ^ (state << shift2)
                state = state ^ (state >> (shift1 // 2))
                
                # Converter para soma da loteria
                lottery_sum = (state % 220) + 80
                sequence.append(lottery_sum)
            
            real_sequence = [sum(draw['numbers']) for draw in self.historical_data[:len(sequence)]]
            
            correlation = self.calculate_correlation(real_sequence, sequence)
            
            # Penalizar soluções muito simples ou muito complexas
            complexity_penalty = 0
            if mult == 1 and add == 0:
                complexity_penalty = 0.2
            elif mult > 2**15 or add > 2**15:
                complexity_penalty = 0.1
            
            return max(0, correlation - complexity_penalty)
            
        except:
            return 0
    
    def genetic_algorithm(self, fitness_function, bounds, maximize=True, generations=None):
        """Algoritmo genético genérico"""
        if generations is None:
            generations = self.ga_params['generations']
        
        population_size = self.ga_params['population_size']
        mutation_rate = self.ga_params['mutation_rate']
        crossover_rate = self.ga_params['crossover_rate']
        elite_size = self.ga_params['elite_size']
        
        # Dimensões do problema
        dimensions = len(bounds)
        
        # População inicial
        population = []
        for _ in range(population_size):
            individual = []
            for i in range(dimensions):
                low, high = bounds[i]
                value = random.uniform(low, high)
                individual.append(value)
            population.append(individual)
        
        # Evolução
        evolution_history = []
        best_fitness_ever = -np.inf if maximize else np.inf
        best_individual_ever = None
        
        for generation in range(generations):
            # Avaliar fitness
            fitness_scores = []
            for individual in population:
                fitness = fitness_function(individual)
                fitness_scores.append(fitness)
            
            # Atualizar melhor
            current_best_idx = np.argmax(fitness_scores) if maximize else np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            
            if (maximize and current_best_fitness > best_fitness_ever) or \
               (not maximize and current_best_fitness < best_fitness_ever):
                best_fitness_ever = current_best_fitness
                best_individual_ever = population[current_best_idx][:]
            
            evolution_history.append({
                'generation': generation,
                'best_fitness': current_best_fitness,
                'mean_fitness': np.mean(fitness_scores),
                'std_fitness': np.std(fitness_scores)
            })
            
            # Seleção e reprodução
            new_population = []
            
            # Elitismo
            elite_indices = np.argsort(fitness_scores)
            if maximize:
                elite_indices = elite_indices[-elite_size:]
            else:
                elite_indices = elite_indices[:elite_size]
            
            for idx in elite_indices:
                new_population.append(population[idx][:])
            
            # Reprodução
            while len(new_population) < population_size:
                # Seleção por torneio
                parent1 = self.tournament_selection(population, fitness_scores, maximize)
                parent2 = self.tournament_selection(population, fitness_scores, maximize)
                
                # Crossover
                if random.random() < crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                
                # Mutação
                if random.random() < mutation_rate:
                    child1 = self.mutate(child1, bounds)
                if random.random() < mutation_rate:
                    child2 = self.mutate(child2, bounds)
                
                new_population.extend([child1, child2])
            
            # Limitar população
            population = new_population[:population_size]
            
            # Log progresso
            if generation % 50 == 0:
                print(f"       Gen {generation}: Best fitness = {best_fitness_ever:.6f}")
        
        return best_individual_ever, best_fitness_ever, evolution_history
    
    def tournament_selection(self, population, fitness_scores, maximize):
        """Seleção por torneio"""
        tournament_size = self.ga_params['tournament_size']
        
        # Selecionar indivíduos aleatórios para o torneio
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        # Selecionar o melhor do torneio
        if maximize:
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        else:
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        
        return population[winner_idx][:]
    
    def crossover(self, parent1, parent2):
        """Crossover de ponto único"""
        if len(parent1) != len(parent2) or len(parent1) < 2:
            return parent1[:], parent2[:]
        
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        return child1, child2
    
    def mutate(self, individual, bounds):
        """Mutação gaussiana"""
        mutated = individual[:]
        
        for i in range(len(individual)):
            if random.random() < 0.1:  # Probabilidade de mutação por gene
                low, high = bounds[i]
                
                # Mutação gaussiana
                std_dev = (high - low) * 0.1  # 10% do range como desvio padrão
                mutation = random.gauss(0, std_dev)
                
                new_value = individual[i] + mutation
                new_value = max(low, min(high, new_value))  # Manter dentro dos bounds
                
                mutated[i] = new_value
        
        return mutated
    
    def mcmc_optimization(self):
        """Otimização usando MCMC (Ensemble Sampling)"""
        if not EMCEE_AVAILABLE:
            return {'status': 'emcee_not_available'}
        
        print("     Executando MCMC com emcee...")
        
        # Definir função de log-probabilidade
        def log_probability(params):
            # Mapear parâmetros para LCG (exemplo)
            if len(params) < 4:
                return -np.inf
            
            seed, a, c, m_factor = params
            
            # Priors
            if seed <= 0 or a <= 0 or c < 0 or m_factor <= 0:
                return -np.inf
            
            # Calcular likelihood
            m = int(m_factor * 1000000)
            fitness = self.evaluate_lcg_fitness(int(seed), int(a), int(c), m)
            
            # Converter fitness para log-probability
            log_prob = np.log(fitness + 1e-10)
            
            return log_prob
        
        # Configurar MCMC
        n_dim = 4
        n_walkers = self.mcmc_params['n_walkers']
        n_steps = self.mcmc_params['n_steps']
        
        # Posições iniciais dos walkers
        initial_positions = []
        for _ in range(n_walkers):
            pos = [
                random.uniform(1, 2**20),      # seed
                random.uniform(1, 2**16),      # a
                random.uniform(0, 2**16),      # c
                random.uniform(1, 10)          # m_factor
            ]
            initial_positions.append(pos)
        
        # Executar MCMC
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_probability)
        
        print(f"       Executando {n_steps} passos com {n_walkers} walkers...")
        sampler.run_mcmc(initial_positions, n_steps, progress=True)
        
        # Analisar resultados
        samples = sampler.get_chain(discard=self.mcmc_params['burn_in'], flat=True)
        
        # Encontrar melhor amostra
        log_probs = sampler.get_log_prob(discard=self.mcmc_params['burn_in'], flat=True)
        best_idx = np.argmax(log_probs)
        best_params = samples[best_idx]
        
        # Estatísticas
        param_stats = {}
        param_names = ['seed', 'a', 'c', 'm_factor']
        
        for i, name in enumerate(param_names):
            param_stats[name] = {
                'mean': float(np.mean(samples[:, i])),
                'std': float(np.std(samples[:, i])),
                'median': float(np.median(samples[:, i])),
                'best': float(best_params[i])
            }
        
        return {
            'best_parameters': {
                'seed': int(best_params[0]),
                'a': int(best_params[1]),
                'c': int(best_params[2]),
                'm': int(best_params[3] * 1000000)
            },
            'parameter_statistics': param_stats,
            'acceptance_rate': float(np.mean(sampler.acceptance_fraction)),
            'n_samples': len(samples),
            'autocorrelation_time': sampler.get_autocorr_time(quiet=True).tolist() if len(samples) > 100 else None
        }
    
    def hybrid_optimization(self):
        """Otimização híbrida combinando múltiplas técnicas"""
        print("     Executando otimização híbrida...")
        
        # Combinar resultados de diferentes abordagens
        hybrid_results = {}
        
        # 1. Busca em grade para parâmetros conhecidos
        hybrid_results['grid_search'] = self.grid_search_optimization()
        
        # 2. Otimização diferencial
        hybrid_results['differential_evolution'] = self.differential_evolution_optimization()
        
        # 3. Simulated Annealing
        hybrid_results['simulated_annealing'] = self.simulated_annealing_optimization()
        
        # 4. Combinar melhores resultados
        best_overall = self.combine_hybrid_results(hybrid_results)
        
        return {
            'individual_methods': hybrid_results,
            'best_combined': best_overall
        }
    
    def grid_search_optimization(self):
        """Busca em grade para parâmetros específicos"""
        print("       Executando busca em grade...")
        
        # Grade reduzida para viabilidade computacional
        seed_values = [1, 12345, 123456, 1234567]
        a_values = [1, 16807, 48271, 69621]
        c_values = [0, 1, 12345]
        m_values = [2**31 - 1, 2**32]
        
        best_params = None
        best_fitness = 0
        
        total_combinations = len(seed_values) * len(a_values) * len(c_values) * len(m_values)
        print(f"         Testando {total_combinations} combinações...")
        
        for seed in seed_values:
            for a in a_values:
                for c in c_values:
                    for m in m_values:
                        fitness = self.evaluate_lcg_fitness(seed, a, c, m)
                        
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_params = {'seed': seed, 'a': a, 'c': c, 'm': m}
        
        return {
            'best_parameters': best_params,
            'best_fitness': best_fitness,
            'total_tested': total_combinations
        }
    
    def differential_evolution_optimization(self):
        """Otimização por evolução diferencial"""
        print("       Executando evolução diferencial...")
        
        def objective(params):
            seed, a, c, m_factor = params
            m = int(m_factor * 1000000)
            fitness = self.evaluate_lcg_fitness(int(seed), int(a), int(c), m)
            return -fitness  # Minimizar o negativo da fitness
        
        bounds = [
            (1, 2**20),     # seed
            (1, 2**16),     # a
            (0, 2**16),     # c
            (1, 10)         # m_factor
        ]
        
        try:
            result = optimize.differential_evolution(
                objective, bounds, maxiter=100, seed=42
            )
            
            best_params = result.x
            
            return {
                'best_parameters': {
                    'seed': int(best_params[0]),
                    'a': int(best_params[1]),
                    'c': int(best_params[2]),
                    'm': int(best_params[3] * 1000000)
                },
                'best_fitness': -result.fun,
                'success': result.success,
                'iterations': result.nit
            }
        except Exception as e:
            return {'error': str(e)}
    
    def simulated_annealing_optimization(self):
        """Otimização por Simulated Annealing"""
        print("       Executando Simulated Annealing...")
        
        def objective(params):
            seed, a, c, m_factor = params
            m = int(m_factor * 1000000)
            fitness = self.evaluate_lcg_fitness(int(seed), int(a), int(c), m)
            return -fitness
        
        bounds = [
            (1, 2**20),
            (1, 2**16),
            (0, 2**16),
            (1, 10)
        ]
        
        try:
            result = optimize.basinhopping(
                objective, 
                x0=[12345, 1103515245, 12345, 2.147],  # Ponto inicial
                niter=50,
                T=1.0,
                stepsize=0.5
            )
            
            best_params = result.x
            
            return {
                'best_parameters': {
                    'seed': int(best_params[0]),
                    'a': int(best_params[1]),
                    'c': int(best_params[2]),
                    'm': int(best_params[3] * 1000000)
                },
                'best_fitness': -result.fun,
                'success': result.message,
                'iterations': result.nit
            }
        except Exception as e:
            return {'error': str(e)}
    
    def combine_hybrid_results(self, hybrid_results):
        """Combina resultados híbridos"""
        all_solutions = []
        
        for method, result in hybrid_results.items():
            if isinstance(result, dict) and 'best_parameters' in result and 'best_fitness' in result:
                all_solutions.append({
                    'method': method,
                    'parameters': result['best_parameters'],
                    'fitness': result['best_fitness']
                })
        
        if not all_solutions:
            return {'status': 'no_valid_solutions'}
        
        # Ordenar por fitness
        all_solutions.sort(key=lambda x: x['fitness'], reverse=True)
        
        return {
            'best_solution': all_solutions[0],
            'all_solutions': all_solutions,
            'consensus_analysis': self.analyze_parameter_consensus(all_solutions)
        }
    
    def analyze_parameter_consensus(self, solutions):
        """Analisa consenso entre parâmetros encontrados"""
        if len(solutions) < 2:
            return {'status': 'insufficient_solutions'}
        
        # Coletar parâmetros
        seeds = []
        a_values = []
        c_values = []
        m_values = []
        
        for sol in solutions:
            params = sol['parameters']
            if 'seed' in params:
                seeds.append(params['seed'])
            if 'a' in params:
                a_values.append(params['a'])
            if 'c' in params:
                c_values.append(params['c'])
            if 'm' in params:
                m_values.append(params['m'])
        
        consensus = {}
        
        # Analisar consenso para cada parâmetro
        for param_name, values in [('seed', seeds), ('a', a_values), ('c', c_values), ('m', m_values)]:
            if values:
                consensus[param_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'range': [float(min(values)), float(max(values))],
                    'most_common': max(set(values), key=values.count) if values else None,
                    'agreement_score': 1 - (np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0
                }
        
        return consensus
    
    def extract_best_solutions(self):
        """Extrai melhores soluções de todos os métodos"""
        print("\n   📊 Extraindo melhores soluções...")
        
        all_solutions = []
        
        for method, result in self.optimization_results.items():
            if isinstance(result, dict):
                solution = self.extract_solution_from_result(method, result)
                if solution:
                    all_solutions.append(solution)
        
        # Ordenar por fitness/score
        all_solutions.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        self.best_solutions = all_solutions
        
        print(f"     ✓ {len(all_solutions)} soluções extraídas")
        
        return all_solutions
    
    def extract_solution_from_result(self, method, result):
        """Extrai solução de resultado específico"""
        if 'best_parameters' in result and ('fitness' in result or 'score' in result):
            return {
                'method': method,
                'parameters': result['best_parameters'],
                'score': result.get('fitness', result.get('score', 0)),
                'validation': result.get('validation_score', {}),
                'metadata': {k: v for k, v in result.items() 
                           if k not in ['best_parameters', 'fitness', 'score', 'validation_score']}
            }
        
        elif method == 'hybrid_search' and 'best_combined' in result:
            hybrid_result = result['best_combined']
            if 'best_solution' in hybrid_result:
                best = hybrid_result['best_solution']
                return {
                    'method': f"hybrid_{best['method']}",
                    'parameters': best['parameters'],
                    'score': best['fitness'],
                    'validation': {},
                    'metadata': hybrid_result
                }
        
        return None
    
    def generate_optimization_report(self):
        """Gera relatório de otimização"""
        print("\n📋 Gerando relatório de otimização...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'optimization_type': 'Genetic Algorithm and MCMC',
            'total_draws_analyzed': len(self.historical_data),
            'methods_used': list(self.optimization_results.keys()),
            'ga_parameters': self.ga_params,
            'mcmc_parameters': self.mcmc_params,
            'results': self.optimization_results,
            'best_solutions': self.best_solutions,
            'summary': {
                'total_solutions_found': len(self.best_solutions),
                'best_overall_score': self.best_solutions[0]['score'] if self.best_solutions else 0,
                'best_method': self.best_solutions[0]['method'] if self.best_solutions else None,
                'optimization_quality': self.assess_optimization_quality()
            },
            'recommendations': self.generate_optimization_recommendations()
        }
        
        # Salvar relatório
        report_path = f"../output/genetic_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   ✓ Relatório salvo em: {report_path}")
        
        return report
    
    def assess_optimization_quality(self):
        """Avalia qualidade da otimização"""
        if not self.best_solutions:
            return 0
        
        quality_factors = []
        
        # Diversidade de métodos
        methods = set(sol['method'] for sol in self.best_solutions)
        diversity_score = min(1.0, len(methods) / 5)
        quality_factors.append(diversity_score)
        
        # Score máximo
        max_score = self.best_solutions[0]['score']
        score_quality = min(1.0, max_score * 2)  # Assumindo que score máximo esperado é 0.5
        quality_factors.append(score_quality)
        
        # Consenso entre soluções
        if len(self.best_solutions) > 1:
            top_scores = [sol['score'] for sol in self.best_solutions[:5]]
            score_consistency = 1 - (np.std(top_scores) / np.mean(top_scores)) if np.mean(top_scores) > 0 else 0
            quality_factors.append(score_consistency)
        
        return np.mean(quality_factors)
    
    def generate_optimization_recommendations(self):
        """Gera recomendações de otimização"""
        recommendations = []
        
        if not self.best_solutions:
            recommendations.append("Nenhuma solução satisfatória encontrada. Recomenda-se:")
            recommendations.append("1. Aumentar número de gerações/iterações")
            recommendations.append("2. Expandir espaço de busca")
            recommendations.append("3. Tentar algoritmos alternativos")
            return recommendations
        
        best_score = self.best_solutions[0]['score']
        best_method = self.best_solutions[0]['method']
        
        if best_score > 0.7:
            recommendations.append(f"Excelente resultado obtido com {best_method}!")
            recommendations.append("Recomenda-se validação com dados independentes")
        elif best_score > 0.4:
            recommendations.append(f"Resultado promissor com {best_method}")
            recommendations.append("Recomenda-se refinamento adicional dos parâmetros")
        else:
            recommendations.append("Scores baixos sugerem que:")
            recommendations.append("1. Sistema pode não ser baseado em PRNG padrão")
            recommendations.append("2. Parâmetros de busca precisam ser expandidos")
            recommendations.append("3. Algoritmo pode ser mais complexo que o assumido")
        
        # Recomendações específicas por método
        method_counts = {}
        for sol in self.best_solutions[:10]:
            method = sol['method'].split('_')[0]  # Pegar tipo base
            method_counts[method] = method_counts.get(method, 0) + 1
        
        if method_counts:
            dominant_method = max(method_counts, key=method_counts.get)
            recommendations.append(f"Método {dominant_method} mostrou-se mais promissor")
        
        return recommendations
    
    def visualize_optimization_results(self):
        """Cria visualizações dos resultados de otimização"""
        print("\n📈 Gerando visualizações de otimização...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Evolução do fitness (se disponível)
        ax1 = axes[0, 0]
        
        evolution_data = None
        for method, result in self.optimization_results.items():
            if isinstance(result, dict) and 'evolution' in result:
                evolution_data = result['evolution']
                break
        
        if evolution_data:
            generations = [e['generation'] for e in evolution_data]
            best_fitness = [e['best_fitness'] for e in evolution_data]
            mean_fitness = [e['mean_fitness'] for e in evolution_data]
            
            ax1.plot(generations, best_fitness, label='Melhor Fitness', linewidth=2)
            ax1.plot(generations, mean_fitness, label='Fitness Médio', alpha=0.7)
            ax1.set_title('Evolução do Fitness - Algoritmo Genético')
            ax1.set_xlabel('Geração')
            ax1.set_ylabel('Fitness')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Dados de evolução\nnão disponíveis', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Evolução do Fitness')
        
        # 2. Comparação de métodos
        ax2 = axes[0, 1]
        
        if self.best_solutions:
            methods = [sol['method'] for sol in self.best_solutions[:10]]
            scores = [sol['score'] for sol in self.best_solutions[:10]]
            
            bars = ax2.bar(range(len(methods)), scores, alpha=0.7)
            ax2.set_title('Comparação de Métodos - Top 10')
            ax2.set_xlabel('Método')
            ax2.set_ylabel('Score')
            ax2.set_xticks(range(len(methods)))
            ax2.set_xticklabels(methods, rotation=45, ha='right')
            
            # Adicionar valores nas barras
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Distribuição de scores
        ax3 = axes[1, 0]
        
        if self.best_solutions:
            all_scores = [sol['score'] for sol in self.best_solutions]
            ax3.hist(all_scores, bins=min(20, len(all_scores)), alpha=0.7, density=True)
            ax3.axvline(x=np.mean(all_scores), color='red', linestyle='--', 
                       label=f'Média: {np.mean(all_scores):.3f}')
            ax3.set_title('Distribuição de Scores')
            ax3.set_xlabel('Score')
            ax3.set_ylabel('Densidade')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Parâmetros dos melhores candidatos
        ax4 = axes[1, 1]
        
        if self.best_solutions:
            # Analisar parâmetros mais comuns
            param_analysis = self.analyze_top_parameters()
            
            if param_analysis:
                param_names = list(param_analysis.keys())
                param_values = [param_analysis[name]['mean'] for name in param_names]
                param_stds = [param_analysis[name]['std'] for name in param_names]
                
                bars = ax4.bar(param_names, param_values, yerr=param_stds, 
                              capsize=5, alpha=0.7)
                ax4.set_title('Parâmetros Médios - Top Soluções')
                ax4.set_ylabel('Valor Médio')
                ax4.tick_params(axis='x', rotation=45)
                
                # Escala logarítmica se necessário
                if max(param_values) > 1000:
                    ax4.set_yscale('log')
            else:
                ax4.text(0.5, 0.5, 'Análise de parâmetros\nnão disponível', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Análise de Parâmetros')
        
        plt.tight_layout()
        plt.savefig('../output/genetic_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✓ Visualizações salvas em: genetic_optimization_results.png")
    
    def analyze_top_parameters(self):
        """Analisa parâmetros das melhores soluções"""
        if len(self.best_solutions) < 5:
            return {}
        
        top_solutions = self.best_solutions[:10]
        
        # Coletar todos os parâmetros
        all_params = {}
        
        for sol in top_solutions:
            params = sol['parameters']
            for param_name, value in params.items():
                if isinstance(value, (int, float)):
                    if param_name not in all_params:
                        all_params[param_name] = []
                    all_params[param_name].append(value)
        
        # Calcular estatísticas
        param_stats = {}
        for param_name, values in all_params.items():
            if len(values) > 1:
                param_stats[param_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        return param_stats

# Script de execução
if __name__ == "__main__":
    from seed_discovery_engine import SeedDiscoveryEngine
    from multi_prng_reverse_engineer import MultiPRNGReverseEngineer
    
    print("🧬 OTIMIZAÇÃO GENÉTICA DE SEEDS")
    print("="*70)
    
    # Carregar dados
    engine = SeedDiscoveryEngine()
    data_path = "../data/MegaSena3.xlsx"
    engine.load_megasena_data(data_path)
    
    # Executar engenharia reversa primeiro
    reverse_engineer = MultiPRNGReverseEngineer(engine.historical_data)
    reverse_engineer.reverse_engineer_all_prngs()
    
    # Inicializar otimizador genético
    genetic_optimizer = GeneticOptimizationEngine(
        engine.historical_data, 
        reverse_engineer.prng_candidates
    )
    
    # Executar otimização
    genetic_optimizer.optimize_all_candidates()
    
    # Gerar relatórios
    genetic_optimizer.generate_optimization_report()
    genetic_optimizer.visualize_optimization_results()
    
    print("\n✅ Otimização genética completa!")
    
    if genetic_optimizer.best_solutions:
        best = genetic_optimizer.best_solutions[0]
        print(f"\n🏆 Melhor solução:")
        print(f"   Método: {best['method']}")
        print(f"   Score: {best['score']:.6f}")
        print(f"   Parâmetros: {best['parameters']}")
    else:
        print("\n⚠️ Nenhuma solução satisfatória encontrada")