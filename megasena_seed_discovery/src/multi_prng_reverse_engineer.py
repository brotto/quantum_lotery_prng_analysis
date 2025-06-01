#!/usr/bin/env python3
"""
Engenheiro Reverso Multi-PRNG
Implementa engenharia reversa para múltiplos tipos de geradores pseudoaleatórios
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal, optimize
from scipy.special import factorial
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas de otimização genética
try:
    import pymc as pm
    import emcee
    MCMC_AVAILABLE = True
except ImportError:
    MCMC_AVAILABLE = False
    print("⚠️ Bibliotecas MCMC não disponíveis")

class MultiPRNGReverseEngineer:
    def __init__(self, historical_data):
        self.historical_data = historical_data
        self.prng_models = {}
        self.seed_candidates = {}
        self.analysis_results = {}
        
        # Parâmetros conhecidos de diferentes PRNGs
        self.prng_signatures = {
            'LCG': {
                'parameters': ['a', 'c', 'm'],
                'formula': 'X_{n+1} = (a * X_n + c) mod m',
                'common_values': {
                    'a': [1103515245, 16807, 48271, 69621, 1],
                    'c': [12345, 0, 1, 2, 3, 4],
                    'm': [2**31-1, 2**32, 2**31]
                }
            },
            'LFSR': {
                'parameters': ['polynomial', 'initial_state'],
                'formula': 'bit-shift with XOR feedback',
                'common_polynomials': [0x80000057, 0x80000062, 0x8000006E]
            },
            'Mersenne_Twister': {
                'parameters': ['w', 'n', 'r', 'm', 'a', 'u', 'd', 's', 'b', 't', 'c', 'l'],
                'formula': 'Complex matrix operations',
                'state_size': 624
            },
            'Xorshift': {
                'parameters': ['a', 'b', 'c'],
                'formula': 'x ^= x << a; x ^= x >> b; x ^= x << c',
                'common_values': {
                    'a': [13, 17, 5],
                    'b': [17, 5, 1],
                    'c': [5, 1, 13]
                }
            },
            'PCG': {
                'parameters': ['multiplier', 'increment', 'state'],
                'formula': 'LCG with permutation output',
                'features': ['rotation', 'xor_shift']
            },
            'ISAAC': {
                'parameters': ['aa', 'bb', 'cc'],
                'formula': 'Cryptographic PRNG',
                'state_size': 256
            }
        }
    
    def reverse_engineer_all_prngs(self):
        """Executa engenharia reversa para todos os tipos de PRNG"""
        print("\n🔧 Iniciando Engenharia Reversa Multi-PRNG...")
        
        results = {}
        
        # 1. Análise LCG
        print("\n   🔍 Analisando LCG...")
        results['LCG'] = self.reverse_engineer_lcg()
        
        # 2. Análise LFSR
        print("\n   🔍 Analisando LFSR...")
        results['LFSR'] = self.reverse_engineer_lfsr()
        
        # 3. Análise Mersenne Twister
        print("\n   🔍 Analisando Mersenne Twister...")
        results['Mersenne_Twister'] = self.reverse_engineer_mersenne_twister()
        
        # 4. Análise Xorshift
        print("\n   🔍 Analisando Xorshift...")
        results['Xorshift'] = self.reverse_engineer_xorshift()
        
        # 5. Análise PCG
        print("\n   🔍 Analisando PCG...")
        results['PCG'] = self.reverse_engineer_pcg()
        
        # 6. Análise híbrida/complexa
        print("\n   🔍 Analisando sistemas híbridos...")
        results['Hybrid'] = self.reverse_engineer_hybrid_system()
        
        # 7. Machine Learning para detecção de padrões
        print("\n   🤖 Análise por Machine Learning...")
        results['ML_Based'] = self.ml_pattern_detection()
        
        self.analysis_results = results
        
        # Classificar resultados por confiança
        self.rank_prng_candidates()
        
        return results
    
    def reverse_engineer_lcg(self):
        """Engenharia reversa específica para LCG"""
        print("     Testando parâmetros LCG...")
        
        # Extrair sequência de somas como proxy
        sequence = [sum(draw['numbers']) for draw in self.historical_data]
        
        lcg_results = []
        
        # Testar combinações conhecidas de parâmetros LCG
        for a in self.prng_signatures['LCG']['common_values']['a']:
            for c in self.prng_signatures['LCG']['common_values']['c']:
                for m in self.prng_signatures['LCG']['common_values']['m']:
                    
                    # Buscar seed inicial
                    best_seed, confidence = self.find_lcg_seed(sequence, a, c, m)
                    
                    if confidence > 0.3:  # Threshold de confiança
                        lcg_results.append({
                            'a': a,
                            'c': c,
                            'm': m,
                            'seed': best_seed,
                            'confidence': confidence,
                            'validation_score': self.validate_lcg_sequence(sequence, best_seed, a, c, m)
                        })
        
        # Ordenar por confiança
        lcg_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Análise estatística adicional
        statistical_analysis = self.analyze_lcg_properties(sequence)
        
        return {
            'candidates': lcg_results[:10],  # Top 10
            'statistical_analysis': statistical_analysis,
            'best_match': lcg_results[0] if lcg_results else None
        }
    
    def find_lcg_seed(self, sequence, a, c, m, max_search=1000):
        """Encontra o melhor seed para parâmetros LCG dados"""
        best_seed = 0
        best_correlation = 0
        
        # Busca inteligente baseada nos primeiros valores
        if len(sequence) >= 3:
            # Usar Hull-Dobell para inferir seed inicial
            x1, x2, x3 = sequence[0], sequence[1], sequence[2]
            
            # Tentar resolver: x2 = (a * x1 + c) mod m
            for seed_candidate in range(0, min(max_search, m), max(1, m//1000)):
                generated_seq = self.generate_lcg_sequence(seed_candidate, a, c, m, len(sequence[:100]))
                
                # Mapear sequência gerada para range esperado
                if generated_seq:
                    scaled_seq = self.scale_sequence_to_lottery_range(generated_seq)
                    correlation = self.calculate_sequence_correlation(sequence[:100], scaled_seq)
                    
                    if correlation > best_correlation:
                        best_correlation = correlation
                        best_seed = seed_candidate
        
        return best_seed, best_correlation
    
    def generate_lcg_sequence(self, seed, a, c, m, length):
        """Gera sequência LCG"""
        sequence = []
        x = seed
        
        for _ in range(length):
            x = (a * x + c) % m
            sequence.append(x)
        
        return sequence
    
    def scale_sequence_to_lottery_range(self, sequence):
        """Escala sequência para range da loteria"""
        if not sequence:
            return []
        
        # Normalizar para range 0-1
        min_val, max_val = min(sequence), max(sequence)
        if max_val == min_val:
            return [200] * len(sequence)  # Valor médio se constante
        
        normalized = [(x - min_val) / (max_val - min_val) for x in sequence]
        
        # Escalar para range da soma da Mega Sena (aproximadamente 80-300)
        scaled = [80 + norm * 220 for norm in normalized]
        
        return scaled
    
    def calculate_sequence_correlation(self, seq1, seq2):
        """Calcula correlação entre duas sequências"""
        if len(seq1) != len(seq2) or len(seq1) < 2:
            return 0
        
        try:
            correlation, _ = stats.pearsonr(seq1, seq2)
            return abs(correlation) if not np.isnan(correlation) else 0
        except:
            return 0
    
    def validate_lcg_sequence(self, original_sequence, seed, a, c, m):
        """Valida sequência LCG contra dados originais"""
        generated = self.generate_lcg_sequence(seed, a, c, m, len(original_sequence))
        scaled = self.scale_sequence_to_lottery_range(generated)
        
        # Múltiplas métricas de validação
        correlation = self.calculate_sequence_correlation(original_sequence, scaled)
        
        # Teste Kolmogorov-Smirnov
        ks_stat, ks_p = stats.ks_2samp(original_sequence, scaled)
        
        # Teste de runs
        runs_score = self.runs_test(original_sequence, scaled)
        
        # Score combinado
        validation_score = (correlation + (1 - ks_stat) + runs_score) / 3
        
        return validation_score
    
    def runs_test(self, seq1, seq2):
        """Teste de runs para aleatoriedade"""
        if len(seq1) != len(seq2):
            return 0
        
        # Diferenças entre sequências
        diffs = [1 if seq1[i] > seq2[i] else 0 for i in range(len(seq1))]
        
        # Contar runs
        runs = 1
        for i in range(1, len(diffs)):
            if diffs[i] != diffs[i-1]:
                runs += 1
        
        # Normalizar pelo comprimento
        expected_runs = len(diffs) / 2
        runs_score = 1 - abs(runs - expected_runs) / expected_runs if expected_runs > 0 else 0
        
        return max(0, runs_score)
    
    def analyze_lcg_properties(self, sequence):
        """Análise estatística das propriedades LCG"""
        # Análise de periodicidade
        autocorr = signal.correlate(sequence, sequence, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Encontrar picos de periodicidade
        peaks, _ = signal.find_peaks(autocorr[1:100], height=0.1)
        
        # Análise espectral
        freqs = np.fft.fftfreq(len(sequence))
        spectrum = np.abs(np.fft.fft(sequence))
        
        # Teste de linearidade
        x = np.arange(len(sequence))
        linear_corr, _ = stats.pearsonr(x, sequence)
        
        return {
            'periodicity_peaks': peaks.tolist(),
            'max_autocorr': np.max(autocorr[1:100]),
            'spectral_peaks': np.argsort(spectrum)[-5:].tolist(),
            'linear_correlation': linear_corr,
            'variance': np.var(sequence),
            'mean': np.mean(sequence)
        }
    
    def reverse_engineer_lfsr(self):
        """Engenharia reversa para LFSR"""
        print("     Analisando polinômios LFSR...")
        
        # Converter para sequência binária
        binary_sequence = self.convert_to_binary_sequence()
        
        lfsr_results = []
        
        # Testar polinômios conhecidos
        for poly in self.prng_signatures['LFSR']['common_polynomials']:
            
            # Tentativa de recuperar estado inicial
            for initial_state in range(1, min(1000, 2**16)):
                confidence = self.test_lfsr_polynomial(binary_sequence, poly, initial_state)
                
                if confidence > 0.2:
                    lfsr_results.append({
                        'polynomial': hex(poly),
                        'initial_state': initial_state,
                        'confidence': confidence,
                        'period_analysis': self.analyze_lfsr_period(binary_sequence, poly, initial_state)
                    })
        
        # Análise de polinômio por força bruta (limitada)
        brute_force_results = self.brute_force_lfsr_analysis(binary_sequence)
        
        lfsr_results.extend(brute_force_results)
        lfsr_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'candidates': lfsr_results[:10],
            'binary_analysis': self.analyze_binary_properties(binary_sequence),
            'best_match': lfsr_results[0] if lfsr_results else None
        }
    
    def convert_to_binary_sequence(self):
        """Converte sorteios para sequência binária"""
        binary_seq = []
        
        for draw_info in self.historical_data:
            for num in draw_info['numbers']:
                # Converter para binário (6 bits)
                binary = format(num, '06b')
                binary_seq.extend([int(b) for b in binary])
        
        return binary_seq
    
    def test_lfsr_polynomial(self, binary_sequence, polynomial, initial_state):
        """Testa polinômio LFSR contra sequência"""
        # Simular LFSR
        generated = self.simulate_lfsr(polynomial, initial_state, len(binary_sequence))
        
        # Comparar com sequência real
        if len(generated) != len(binary_sequence):
            return 0
        
        # Correlação binária
        matches = sum(1 for i in range(len(binary_sequence)) if binary_sequence[i] == generated[i])
        correlation = matches / len(binary_sequence)
        
        return correlation
    
    def simulate_lfsr(self, polynomial, initial_state, length):
        """Simula LFSR com polinômio dado"""
        state = initial_state
        output = []
        
        for _ in range(length):
            # XOR de bits baseado no polinômio
            feedback = 0
            temp_poly = polynomial
            temp_state = state
            
            while temp_poly > 0:
                if temp_poly & 1:
                    feedback ^= temp_state & 1
                temp_poly >>= 1
                temp_state >>= 1
            
            # Shift e adicionar feedback
            state = (state >> 1) | (feedback << 15)  # Assumir 16-bit LFSR
            output.append(state & 1)
        
        return output
    
    def analyze_lfsr_period(self, binary_sequence, polynomial, initial_state):
        """Analisa período do LFSR"""
        # Gerar sequência longa para encontrar período
        long_sequence = self.simulate_lfsr(polynomial, initial_state, min(10000, len(binary_sequence) * 2))
        
        # Procurar por repetições
        for period in range(1, min(1000, len(long_sequence) // 2)):
            if long_sequence[:period] == long_sequence[period:2*period]:
                return period
        
        return -1  # Período não encontrado
    
    def brute_force_lfsr_analysis(self, binary_sequence):
        """Análise por força bruta de LFSR"""
        results = []
        
        # Testar polinômios pequenos (computacionalmente viável)
        for poly_bits in range(8, 17):  # 8 a 16 bits
            polynomial = (1 << poly_bits) | 1  # Polinômio básico
            
            for tap in range(1, poly_bits):
                test_poly = polynomial | (1 << tap)
                
                confidence = self.test_lfsr_polynomial(binary_sequence[:1000], test_poly, 1)
                
                if confidence > 0.3:
                    results.append({
                        'polynomial': hex(test_poly),
                        'initial_state': 1,
                        'confidence': confidence,
                        'period_analysis': -1  # Não calculado para força bruta
                    })
        
        return results[:5]  # Top 5
    
    def analyze_binary_properties(self, binary_sequence):
        """Analisa propriedades da sequência binária"""
        # Frequência de 0s e 1s
        zeros = binary_sequence.count(0)
        ones = binary_sequence.count(1)
        balance = abs(zeros - ones) / len(binary_sequence)
        
        # Runs test
        runs = 1
        for i in range(1, len(binary_sequence)):
            if binary_sequence[i] != binary_sequence[i-1]:
                runs += 1
        
        # Autocorrelação
        autocorr = signal.correlate(binary_sequence, binary_sequence, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        return {
            'balance': balance,
            'runs_count': runs,
            'max_autocorr': np.max(autocorr[1:100]) if len(autocorr) > 100 else 0,
            'entropy': stats.entropy([zeros, ones]) if zeros > 0 and ones > 0 else 0
        }
    
    def reverse_engineer_mersenne_twister(self):
        """Engenharia reversa para Mersenne Twister"""
        print("     Analisando Mersenne Twister...")
        
        # MT é complexo, usar análise estatística
        sequence = [sum(draw['numbers']) for draw in self.historical_data]
        
        # Análise de estado interno (simplificada)
        mt_analysis = {
            'state_recovery': self.attempt_mt_state_recovery(sequence),
            'period_analysis': self.analyze_mt_period(sequence),
            'tempering_detection': self.detect_mt_tempering(sequence),
            'statistical_properties': self.analyze_mt_statistics(sequence)
        }
        
        # Confidence baseada em propriedades estatísticas
        confidence = self.calculate_mt_confidence(mt_analysis)
        
        return {
            'analysis': mt_analysis,
            'confidence': confidence,
            'estimated_parameters': self.estimate_mt_parameters(sequence)
        }
    
    def attempt_mt_state_recovery(self, sequence):
        """Tentativa de recuperar estado interno do MT"""
        # Análise de padrões para detectar estado MT
        
        # MT tem período muito longo, procurar por sub-padrões
        window_size = 624  # Tamanho do estado MT19937
        
        if len(sequence) < window_size:
            return {'status': 'insufficient_data'}
        
        # Analisar correlações em janelas
        correlations = []
        for i in range(len(sequence) - window_size):
            window = sequence[i:i+window_size]
            
            # Correlação com próxima janela
            if i + 2*window_size < len(sequence):
                next_window = sequence[i+window_size:i+2*window_size]
                corr = self.calculate_sequence_correlation(window, next_window)
                correlations.append(corr)
        
        return {
            'status': 'analyzed',
            'window_correlations': correlations,
            'mean_correlation': np.mean(correlations) if correlations else 0
        }
    
    def analyze_mt_period(self, sequence):
        """Analisa período do MT"""
        # MT19937 tem período 2^19937-1, inviável de detectar diretamente
        # Procurar por sub-períodos
        
        max_period_test = min(10000, len(sequence) // 2)
        
        for period in [624, 1248, 2496]:  # Múltiplos do estado MT
            if period < len(sequence) // 2:
                seq1 = sequence[:period]
                seq2 = sequence[period:2*period]
                
                correlation = self.calculate_sequence_correlation(seq1, seq2)
                if correlation > 0.8:
                    return {'detected_period': period, 'confidence': correlation}
        
        return {'detected_period': None, 'confidence': 0}
    
    def detect_mt_tempering(self, sequence):
        """Detecta padrões de tempering do MT"""
        # MT aplica operações de tempering na saída
        
        # Analisar distribuição de bits
        bit_analysis = []
        
        for value in sequence[:1000]:  # Analisar primeiros 1000 valores
            # Converter para inteiro e analisar bits
            int_val = int(value * 1000)  # Escalar para inteiro
            
            bit_pattern = {
                'value': value,
                'bit_count': bin(int_val).count('1'),
                'msb': int_val >> 15,  # Bits mais significativos
                'lsb': int_val & 0xFF  # Bits menos significativos
            }
            bit_analysis.append(bit_pattern)
        
        # Análise estatística dos padrões de bits
        bit_counts = [b['bit_count'] for b in bit_analysis]
        msb_values = [b['msb'] for b in bit_analysis]
        
        return {
            'bit_distribution': np.histogram(bit_counts, bins=16)[0].tolist(),
            'msb_entropy': stats.entropy(np.histogram(msb_values, bins=16)[0] + 1),
            'tempering_indicators': self.calculate_tempering_indicators(bit_analysis)
        }
    
    def calculate_tempering_indicators(self, bit_analysis):
        """Calcula indicadores de tempering"""
        # Indicadores baseados em padrões conhecidos do MT
        
        msb_values = [b['msb'] for b in bit_analysis]
        lsb_values = [b['lsb'] for b in bit_analysis]
        
        # Correlação entre MSB e LSB
        msb_lsb_corr = self.calculate_sequence_correlation(msb_values, lsb_values)
        
        # Distribuição uniforme esperada
        msb_uniformity = self.test_uniformity(msb_values)
        lsb_uniformity = self.test_uniformity(lsb_values)
        
        return {
            'msb_lsb_correlation': msb_lsb_corr,
            'msb_uniformity': msb_uniformity,
            'lsb_uniformity': lsb_uniformity
        }
    
    def test_uniformity(self, values):
        """Testa uniformidade de distribuição"""
        hist, _ = np.histogram(values, bins=min(16, len(set(values))))
        
        # Chi-square test para uniformidade
        expected = len(values) / len(hist)
        chi_square = np.sum((hist - expected)**2 / expected)
        
        # Normalizar para score 0-1
        uniformity_score = 1 / (1 + chi_square / len(hist))
        
        return uniformity_score
    
    def analyze_mt_statistics(self, sequence):
        """Analisa propriedades estatísticas do MT"""
        return {
            'mean': np.mean(sequence),
            'variance': np.var(sequence),
            'skewness': stats.skew(sequence),
            'kurtosis': stats.kurtosis(sequence),
            'normality_test': stats.normaltest(sequence)[1],
            'randomness_score': self.calculate_randomness_score(sequence)
        }
    
    def calculate_randomness_score(self, sequence):
        """Calcula score de aleatoriedade"""
        scores = []
        
        # Teste de runs
        median = np.median(sequence)
        runs = 1
        for i in range(1, len(sequence)):
            if (sequence[i] > median) != (sequence[i-1] > median):
                runs += 1
        
        expected_runs = len(sequence) / 2
        runs_score = 1 - abs(runs - expected_runs) / expected_runs if expected_runs > 0 else 0
        scores.append(runs_score)
        
        # Teste de autocorrelação
        autocorr = signal.correlate(sequence, sequence, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        max_autocorr = np.max(autocorr[1:min(100, len(autocorr))])
        autocorr_score = 1 - max_autocorr
        scores.append(autocorr_score)
        
        return np.mean(scores)
    
    def calculate_mt_confidence(self, mt_analysis):
        """Calcula confiança de detecção MT"""
        scores = []
        
        # Score de recuperação de estado
        if mt_analysis['state_recovery']['status'] == 'analyzed':
            state_score = 1 - mt_analysis['state_recovery']['mean_correlation']
            scores.append(state_score)
        
        # Score de período
        if mt_analysis['period_analysis']['detected_period']:
            period_score = mt_analysis['period_analysis']['confidence']
            scores.append(period_score)
        
        # Score de propriedades estatísticas
        stats_score = mt_analysis['statistical_properties']['randomness_score']
        scores.append(stats_score)
        
        return np.mean(scores) if scores else 0
    
    def estimate_mt_parameters(self, sequence):
        """Estima parâmetros do MT"""
        # Parâmetros padrão do MT19937
        return {
            'w': 32,
            'n': 624,
            'r': 31,
            'm': 397,
            'a': 0x9908B0DF,
            'estimated': True,
            'confidence': 'medium'
        }
    
    def reverse_engineer_xorshift(self):
        """Engenharia reversa para Xorshift"""
        print("     Analisando Xorshift...")
        
        sequence = [sum(draw['numbers']) for draw in self.historical_data]
        
        xorshift_results = []
        
        # Testar combinações conhecidas de parâmetros
        for a in self.prng_signatures['Xorshift']['common_values']['a']:
            for b in self.prng_signatures['Xorshift']['common_values']['b']:
                for c in self.prng_signatures['Xorshift']['common_values']['c']:
                    
                    confidence = self.test_xorshift_parameters(sequence, a, b, c)
                    
                    if confidence > 0.2:
                        xorshift_results.append({
                            'a': a,
                            'b': b,
                            'c': c,
                            'confidence': confidence,
                            'period_estimate': self.estimate_xorshift_period(a, b, c)
                        })
        
        xorshift_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'candidates': xorshift_results[:10],
            'bit_pattern_analysis': self.analyze_xorshift_patterns(sequence),
            'best_match': xorshift_results[0] if xorshift_results else None
        }
    
    def test_xorshift_parameters(self, sequence, a, b, c):
        """Testa parâmetros Xorshift"""
        # Simular Xorshift com seed inicial estimado
        seed = int(np.mean(sequence) * 1000) % (2**32)
        
        generated = self.simulate_xorshift(seed, a, b, c, len(sequence))
        scaled = self.scale_sequence_to_lottery_range(generated)
        
        return self.calculate_sequence_correlation(sequence, scaled)
    
    def simulate_xorshift(self, seed, a, b, c, length):
        """Simula Xorshift"""
        x = seed
        sequence = []
        
        for _ in range(length):
            x ^= x << a
            x ^= x >> b
            x ^= x << c
            x &= 0xFFFFFFFF  # Manter 32 bits
            sequence.append(x)
        
        return sequence
    
    def estimate_xorshift_period(self, a, b, c):
        """Estima período do Xorshift"""
        # Período teórico máximo é 2^32-1
        # Calcular estimativa baseada nos parâmetros
        
        # Simplificado: período depende dos valores a, b, c
        period_estimate = (2**a) * (2**b) * (2**c)
        return min(period_estimate, 2**32 - 1)
    
    def analyze_xorshift_patterns(self, sequence):
        """Analisa padrões específicos do Xorshift"""
        # Analisar operações XOR
        
        # Diferenças consecutivas
        diffs = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
        
        # Padrões de bits nas diferenças
        bit_patterns = []
        for diff in diffs[:100]:
            int_diff = int(abs(diff) * 1000)
            bit_pattern = bin(int_diff).count('1')
            bit_patterns.append(bit_pattern)
        
        return {
            'difference_distribution': np.histogram(diffs, bins=20)[0].tolist(),
            'bit_pattern_distribution': np.histogram(bit_patterns, bins=16)[0].tolist(),
            'xor_indicators': self.calculate_xor_indicators(sequence)
        }
    
    def calculate_xor_indicators(self, sequence):
        """Calcula indicadores de operações XOR"""
        # Analisar padrões típicos de XOR
        
        # Alternância de valores
        alternations = 0
        for i in range(len(sequence) - 1):
            if (sequence[i] > np.mean(sequence)) != (sequence[i+1] > np.mean(sequence)):
                alternations += 1
        
        alternation_rate = alternations / (len(sequence) - 1)
        
        # Distribuição de valores
        value_variance = np.var(sequence)
        
        return {
            'alternation_rate': alternation_rate,
            'value_variance': value_variance,
            'xor_score': alternation_rate * (1 / (1 + value_variance/1000))
        }
    
    def reverse_engineer_pcg(self):
        """Engenharia reversa para PCG"""
        print("     Analisando PCG...")
        
        sequence = [sum(draw['numbers']) for draw in self.historical_data]
        
        # PCG é LCG + permutação
        # Primeiro detectar LCG subjacente
        lcg_analysis = self.reverse_engineer_lcg()
        
        # Depois analisar permutação
        permutation_analysis = self.analyze_pcg_permutation(sequence)
        
        # Combinar análises
        confidence = 0
        if lcg_analysis['best_match']:
            confidence += lcg_analysis['best_match']['confidence'] * 0.7
        
        confidence += permutation_analysis['permutation_score'] * 0.3
        
        return {
            'underlying_lcg': lcg_analysis['best_match'],
            'permutation_analysis': permutation_analysis,
            'confidence': confidence,
            'estimated_parameters': self.estimate_pcg_parameters(lcg_analysis, permutation_analysis)
        }
    
    def analyze_pcg_permutation(self, sequence):
        """Analisa permutação PCG"""
        # PCG aplica rotação e XOR-shift na saída
        
        # Analisar padrões de rotação
        rotation_indicators = []
        
        for i in range(len(sequence) - 1):
            val1 = int(sequence[i] * 1000) % 256
            val2 = int(sequence[i+1] * 1000) % 256
            
            # Testar rotações possíveis
            for rot in range(8):
                rotated = ((val1 << rot) | (val1 >> (8 - rot))) & 0xFF
                if rotated == val2:
                    rotation_indicators.append(rot)
                    break
        
        # Score de permutação
        permutation_score = len(rotation_indicators) / (len(sequence) - 1) if len(sequence) > 1 else 0
        
        return {
            'rotation_patterns': rotation_indicators,
            'permutation_score': permutation_score,
            'output_analysis': self.analyze_pcg_output_function(sequence)
        }
    
    def analyze_pcg_output_function(self, sequence):
        """Analisa função de saída PCG"""
        # PCG usa XOR-shift para saída
        
        # Analisar padrões de bits
        bit_analysis = []
        
        for value in sequence[:100]:
            int_val = int(value * 1000)
            
            # Analisar padrões XOR-shift
            shifted_patterns = []
            for shift in [1, 2, 4, 8]:
                xor_result = int_val ^ (int_val >> shift)
                shifted_patterns.append(xor_result & 0xFF)
            
            bit_analysis.append(shifted_patterns)
        
        # Calcular complexidade dos padrões
        pattern_complexity = np.var([np.std(patterns) for patterns in bit_analysis])
        
        return {
            'bit_patterns': bit_analysis[:10],  # Primeiros 10 para exemplo
            'pattern_complexity': pattern_complexity,
            'xor_shift_score': min(1.0, pattern_complexity / 100)
        }
    
    def estimate_pcg_parameters(self, lcg_analysis, permutation_analysis):
        """Estima parâmetros PCG"""
        parameters = {
            'multiplier': None,
            'increment': None,
            'rotation_constant': None,
            'xor_shift_constant': None
        }
        
        if lcg_analysis:
            parameters['multiplier'] = lcg_analysis.get('a')
            parameters['increment'] = lcg_analysis.get('c')
        
        # Estimar constantes de permutação
        if permutation_analysis['rotation_patterns']:
            most_common_rotation = max(set(permutation_analysis['rotation_patterns']), 
                                     key=permutation_analysis['rotation_patterns'].count)
            parameters['rotation_constant'] = most_common_rotation
        
        return parameters
    
    def reverse_engineer_hybrid_system(self):
        """Analisa sistemas híbridos ou compostos"""
        print("     Analisando sistemas híbridos...")
        
        sequence = [sum(draw['numbers']) for draw in self.historical_data]
        
        # Analisar múltiplas camadas
        hybrid_analysis = {
            'layer_detection': self.detect_multiple_layers(sequence),
            'combination_analysis': self.analyze_prng_combinations(),
            'state_machine_detection': self.detect_state_machine_behavior(sequence),
            'time_dependent_analysis': self.analyze_time_dependent_patterns()
        }
        
        # Calcular confiança híbrida
        confidence = self.calculate_hybrid_confidence(hybrid_analysis)
        
        return {
            'analysis': hybrid_analysis,
            'confidence': confidence,
            'hybrid_type': self.classify_hybrid_type(hybrid_analysis)
        }
    
    def detect_multiple_layers(self, sequence):
        """Detecta múltiplas camadas de geração"""
        # Analisar em diferentes escalas temporais
        
        scales = [10, 50, 100, 500]
        layer_analysis = {}
        
        for scale in scales:
            if len(sequence) >= scale * 2:
                # Dividir em blocos
                blocks = [sequence[i:i+scale] for i in range(0, len(sequence)-scale, scale)]
                
                # Analisar consistência entre blocos
                block_means = [np.mean(block) for block in blocks]
                block_vars = [np.var(block) for block in blocks]
                
                layer_analysis[f'scale_{scale}'] = {
                    'mean_consistency': 1 - np.var(block_means) / np.mean(block_means) if np.mean(block_means) > 0 else 0,
                    'variance_consistency': 1 - np.var(block_vars) / np.mean(block_vars) if np.mean(block_vars) > 0 else 0,
                    'cross_correlations': self.calculate_cross_scale_correlations(blocks)
                }
        
        return layer_analysis
    
    def calculate_cross_scale_correlations(self, blocks):
        """Calcula correlações entre blocos"""
        correlations = []
        
        for i in range(len(blocks) - 1):
            if len(blocks[i]) == len(blocks[i+1]):
                corr = self.calculate_sequence_correlation(blocks[i], blocks[i+1])
                correlations.append(corr)
        
        return {
            'mean_correlation': np.mean(correlations) if correlations else 0,
            'correlation_variance': np.var(correlations) if correlations else 0
        }
    
    def analyze_prng_combinations(self):
        """Analisa combinações de PRNGs"""
        # Verificar se há evidência de múltiplos PRNGs
        
        all_results = [
            self.analysis_results.get('LCG', {}),
            self.analysis_results.get('LFSR', {}),
            self.analysis_results.get('Xorshift', {})
        ]
        
        # Contar candidatos válidos
        valid_candidates = []
        for result in all_results:
            if isinstance(result, dict) and result.get('best_match'):
                if result['best_match'].get('confidence', 0) > 0.3:
                    valid_candidates.append(result)
        
        combination_score = len(valid_candidates) / 3  # Normalizar por número de tipos
        
        return {
            'multiple_prng_evidence': len(valid_candidates) > 1,
            'combination_score': combination_score,
            'candidate_types': [type(c) for c in valid_candidates]
        }
    
    def detect_state_machine_behavior(self, sequence):
        """Detecta comportamento de máquina de estados"""
        # Procurar por mudanças abruptas de comportamento
        
        # Análise de mudança de regime
        window_size = 50
        regime_changes = []
        
        for i in range(window_size, len(sequence) - window_size, 10):
            before = sequence[i-window_size:i]
            after = sequence[i:i+window_size]
            
            # Teste estatístico para mudança
            t_stat, p_value = stats.ttest_ind(before, after)
            
            if p_value < 0.01:  # Mudança significativa
                regime_changes.append({
                    'position': i,
                    'p_value': p_value,
                    'magnitude': abs(np.mean(after) - np.mean(before))
                })
        
        return {
            'regime_changes': regime_changes,
            'state_transitions': len(regime_changes),
            'average_state_duration': np.mean([regime_changes[i+1]['position'] - regime_changes[i]['position'] 
                                             for i in range(len(regime_changes)-1)]) if len(regime_changes) > 1 else 0
        }
    
    def analyze_time_dependent_patterns(self):
        """Analisa padrões dependentes do tempo"""
        # Verificar se há correlação com datas
        
        time_patterns = []
        
        for i, draw_info in enumerate(self.historical_data):
            if draw_info.get('date'):
                try:
                    if isinstance(draw_info['date'], str):
                        date = pd.to_datetime(draw_info['date'], dayfirst=True)
                    else:
                        date = draw_info['date']
                    
                    time_features = {
                        'index': i,
                        'sum': sum(draw_info['numbers']),
                        'day_of_week': date.dayofweek,
                        'month': date.month,
                        'year': date.year,
                        'day_of_year': date.dayofyear
                    }
                    time_patterns.append(time_features)
                except:
                    continue
        
        if len(time_patterns) > 10:
            # Análise de correlação temporal
            sums = [p['sum'] for p in time_patterns]
            days_of_week = [p['day_of_week'] for p in time_patterns]
            months = [p['month'] for p in time_patterns]
            
            correlations = {
                'day_of_week_correlation': self.calculate_categorical_correlation(sums, days_of_week),
                'month_correlation': self.calculate_categorical_correlation(sums, months),
                'temporal_trend': self.calculate_temporal_trend(time_patterns)
            }
            
            return correlations
        
        return {'insufficient_temporal_data': True}
    
    def calculate_categorical_correlation(self, continuous_var, categorical_var):
        """Calcula correlação entre variável contínua e categórica"""
        try:
            # ANOVA one-way
            groups = {}
            for i, cat in enumerate(categorical_var):
                if cat not in groups:
                    groups[cat] = []
                groups[cat].append(continuous_var[i])
            
            if len(groups) > 1:
                f_stat, p_value = stats.f_oneway(*groups.values())
                return {'f_statistic': f_stat, 'p_value': p_value, 'significant': p_value < 0.05}
            else:
                return {'f_statistic': 0, 'p_value': 1, 'significant': False}
        except:
            return {'f_statistic': 0, 'p_value': 1, 'significant': False}
    
    def calculate_temporal_trend(self, time_patterns):
        """Calcula tendência temporal"""
        indices = [p['index'] for p in time_patterns]
        sums = [p['sum'] for p in time_patterns]
        
        try:
            correlation, p_value = stats.pearsonr(indices, sums)
            return {'correlation': correlation, 'p_value': p_value, 'significant': p_value < 0.05}
        except:
            return {'correlation': 0, 'p_value': 1, 'significant': False}
    
    def calculate_hybrid_confidence(self, hybrid_analysis):
        """Calcula confiança de sistema híbrido"""
        scores = []
        
        # Score de múltiplas camadas
        layer_scores = []
        for scale_analysis in hybrid_analysis['layer_detection'].values():
            layer_score = (scale_analysis['mean_consistency'] + scale_analysis['variance_consistency']) / 2
            layer_scores.append(layer_score)
        
        if layer_scores:
            scores.append(np.mean(layer_scores))
        
        # Score de combinação
        combination_score = hybrid_analysis['combination_analysis']['combination_score']
        scores.append(combination_score)
        
        # Score de máquina de estados
        state_score = min(1.0, hybrid_analysis['state_machine_detection']['state_transitions'] / 10)
        scores.append(state_score)
        
        return np.mean(scores) if scores else 0
    
    def classify_hybrid_type(self, hybrid_analysis):
        """Classifica tipo de sistema híbrido"""
        if hybrid_analysis['combination_analysis']['multiple_prng_evidence']:
            return 'Multiple_PRNG_Combination'
        elif hybrid_analysis['state_machine_detection']['state_transitions'] > 5:
            return 'State_Machine_Based'
        elif not hybrid_analysis['time_dependent_analysis'].get('insufficient_temporal_data', False):
            temporal = hybrid_analysis['time_dependent_analysis']
            if any(corr.get('significant', False) for corr in temporal.values() if isinstance(corr, dict)):
                return 'Time_Dependent'
        
        return 'Complex_Unknown'
    
    def ml_pattern_detection(self):
        """Detecção de padrões usando Machine Learning"""
        print("     Aplicando ML para detecção de padrões...")
        
        # Preparar features
        features = self.prepare_ml_features()
        targets = self.prepare_ml_targets()
        
        if len(features) < 10:
            return {'status': 'insufficient_data'}
        
        # Treinar múltiplos modelos
        ml_results = {}
        
        # Random Forest
        ml_results['random_forest'] = self.train_random_forest(features, targets)
        
        # Gradient Boosting
        ml_results['gradient_boosting'] = self.train_gradient_boosting(features, targets)
        
        # Neural Network
        ml_results['neural_network'] = self.train_neural_network(features, targets)
        
        # Ensemble prediction
        ensemble_prediction = self.create_ensemble_prediction(ml_results)
        
        return {
            'individual_models': ml_results,
            'ensemble_prediction': ensemble_prediction,
            'feature_importance': self.analyze_feature_importance(ml_results),
            'pattern_classification': self.classify_detected_patterns(ensemble_prediction)
        }
    
    def prepare_ml_features(self):
        """Prepara features para ML"""
        features = []
        
        for i in range(len(self.historical_data) - 1):
            current_draw = self.historical_data[i]['numbers']
            next_draw = self.historical_data[i + 1]['numbers']
            
            feature_vector = []
            
            # Features do sorteio atual
            feature_vector.extend([
                sum(current_draw),
                np.mean(current_draw),
                np.std(current_draw),
                max(current_draw) - min(current_draw),
                sum(1 for n in current_draw if n % 2 == 0),  # pares
                sum(1 for n in current_draw if n <= 30)      # baixos
            ])
            
            # Features de diferenças consecutivas
            gaps = [current_draw[j+1] - current_draw[j] for j in range(len(current_draw)-1)]
            feature_vector.extend([
                np.mean(gaps),
                np.std(gaps),
                max(gaps),
                min(gaps)
            ])
            
            # Features de posição
            feature_vector.extend([
                i,  # índice temporal
                i % 7,  # dia da semana simulado
                i % 30,  # dia do mês simulado
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def prepare_ml_targets(self):
        """Prepara targets para ML"""
        targets = []
        
        for i in range(1, len(self.historical_data)):
            next_draw = self.historical_data[i]['numbers']
            
            # Target: próxima soma
            target = sum(next_draw)
            targets.append(target)
        
        return np.array(targets)
    
    def train_random_forest(self, features, targets):
        """Treina Random Forest"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Predições
            y_pred = rf.predict(X_test)
            
            # Métricas
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return {
                'model': rf,
                'mse': mse,
                'r2': r2,
                'feature_importance': rf.feature_importances_.tolist(),
                'predictions': y_pred.tolist()[:10]  # Primeiras 10 predições
            }
        except Exception as e:
            return {'error': str(e)}
    
    def train_gradient_boosting(self, features, targets):
        """Treina Gradient Boosting"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
            
            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb.fit(X_train, y_train)
            
            y_pred = gb.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return {
                'model': gb,
                'mse': mse,
                'r2': r2,
                'feature_importance': gb.feature_importances_.tolist(),
                'predictions': y_pred.tolist()[:10]
            }
        except Exception as e:
            return {'error': str(e)}
    
    def train_neural_network(self, features, targets):
        """Treina rede neural"""
        try:
            # Normalizar features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            X_train, X_test, y_train, y_test = train_test_split(features_scaled, targets, test_size=0.2, random_state=42)
            
            nn = MLPRegressor(hidden_layer_sizes=(50, 25), random_state=42, max_iter=500)
            nn.fit(X_train, y_train)
            
            y_pred = nn.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return {
                'model': nn,
                'scaler': scaler,
                'mse': mse,
                'r2': r2,
                'predictions': y_pred.tolist()[:10]
            }
        except Exception as e:
            return {'error': str(e)}
    
    def create_ensemble_prediction(self, ml_results):
        """Cria predição ensemble"""
        # Combinar predições de todos os modelos válidos
        
        valid_models = []
        weights = []
        
        for model_name, result in ml_results.items():
            if 'error' not in result and 'r2' in result:
                if result['r2'] > 0:  # Apenas modelos com performance positiva
                    valid_models.append(result)
                    weights.append(result['r2'])  # Peso baseado em performance
        
        if not valid_models:
            return {'status': 'no_valid_models'}
        
        # Normalizar pesos
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Predição ensemble (média ponderada)
        ensemble_pred = []
        max_predictions = min([len(model['predictions']) for model in valid_models])
        
        for i in range(max_predictions):
            weighted_pred = sum(model['predictions'][i] * weight 
                              for model, weight in zip(valid_models, normalized_weights))
            ensemble_pred.append(weighted_pred)
        
        return {
            'predictions': ensemble_pred,
            'model_weights': dict(zip(ml_results.keys(), normalized_weights)),
            'ensemble_confidence': np.mean([model['r2'] for model in valid_models])
        }
    
    def analyze_feature_importance(self, ml_results):
        """Analisa importância das features"""
        feature_names = [
            'sum', 'mean', 'std', 'range', 'even_count', 'low_count',
            'gap_mean', 'gap_std', 'gap_max', 'gap_min',
            'temporal_index', 'day_of_week', 'day_of_month'
        ]
        
        importance_analysis = {}
        
        for model_name, result in ml_results.items():
            if 'feature_importance' in result:
                importance_dict = dict(zip(feature_names, result['feature_importance']))
                
                # Ordenar por importância
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                importance_analysis[model_name] = {
                    'feature_ranking': sorted_importance,
                    'top_features': [feat for feat, imp in sorted_importance[:5]]
                }
        
        return importance_analysis
    
    def classify_detected_patterns(self, ensemble_prediction):
        """Classifica padrões detectados"""
        if ensemble_prediction.get('status') == 'no_valid_models':
            return {'classification': 'undetectable'}
        
        confidence = ensemble_prediction.get('ensemble_confidence', 0)
        
        if confidence > 0.7:
            pattern_type = 'highly_predictable'
        elif confidence > 0.4:
            pattern_type = 'moderately_predictable'
        elif confidence > 0.1:
            pattern_type = 'weakly_predictable'
        else:
            pattern_type = 'random_or_complex'
        
        return {
            'classification': pattern_type,
            'confidence': confidence,
            'predictability_score': confidence
        }
    
    def rank_prng_candidates(self):
        """Classifica candidatos PRNG por confiança"""
        print("\n📊 Classificando candidatos PRNG...")
        
        all_candidates = []
        
        for prng_type, analysis in self.analysis_results.items():
            if isinstance(analysis, dict):
                confidence = 0
                
                if prng_type in ['LCG', 'LFSR', 'Xorshift']:
                    if analysis.get('best_match'):
                        confidence = analysis['best_match'].get('confidence', 0)
                elif prng_type in ['Mersenne_Twister', 'PCG', 'Hybrid']:
                    confidence = analysis.get('confidence', 0)
                elif prng_type == 'ML_Based':
                    if 'ensemble_prediction' in analysis:
                        confidence = analysis['ensemble_prediction'].get('ensemble_confidence', 0)
                
                if confidence > 0:
                    all_candidates.append({
                        'type': prng_type,
                        'confidence': confidence,
                        'analysis': analysis
                    })
        
        # Ordenar por confiança
        all_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        self.prng_candidates = all_candidates
        
        print(f"   ✓ {len(all_candidates)} candidatos classificados")
        
        if all_candidates:
            print(f"   🏆 Melhor candidato: {all_candidates[0]['type']} (confiança: {all_candidates[0]['confidence']:.3f})")
        
        return all_candidates
    
    def generate_comprehensive_report(self):
        """Gera relatório abrangente da engenharia reversa"""
        print("\n📋 Gerando relatório abrangente...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'Multi-PRNG Reverse Engineering',
            'total_draws_analyzed': len(self.historical_data),
            'prng_types_tested': list(self.prng_signatures.keys()),
            'analysis_results': self.analysis_results,
            'ranked_candidates': self.prng_candidates,
            'summary': {
                'total_candidates': len(self.prng_candidates),
                'best_candidate': self.prng_candidates[0] if self.prng_candidates else None,
                'confidence_threshold_met': len([c for c in self.prng_candidates if c['confidence'] > 0.5]),
                'analysis_quality': self.assess_analysis_quality()
            },
            'recommendations': self.generate_recommendations()
        }
        
        # Salvar relatório
        report_path = f"../output/multi_prng_reverse_engineering_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   ✓ Relatório salvo em: {report_path}")
        
        return report
    
    def assess_analysis_quality(self):
        """Avalia qualidade da análise"""
        quality_scores = []
        
        # Número de candidatos válidos
        valid_candidates = len([c for c in self.prng_candidates if c['confidence'] > 0.3])
        candidate_score = min(1.0, valid_candidates / 3)
        quality_scores.append(candidate_score)
        
        # Diversidade de análises
        analysis_types = len([k for k, v in self.analysis_results.items() if v])
        diversity_score = min(1.0, analysis_types / 7)
        quality_scores.append(diversity_score)
        
        # Confiança máxima
        max_confidence = max([c['confidence'] for c in self.prng_candidates]) if self.prng_candidates else 0
        quality_scores.append(max_confidence)
        
        return np.mean(quality_scores)
    
    def generate_recommendations(self):
        """Gera recomendações baseadas na análise"""
        recommendations = []
        
        if not self.prng_candidates:
            recommendations.append("Nenhum PRNG padrão claramente identificado. Sistema pode usar algoritmo proprietário ou híbrido complexo.")
        
        elif self.prng_candidates[0]['confidence'] > 0.7:
            best_type = self.prng_candidates[0]['type']
            recommendations.append(f"Alta confiança na detecção de {best_type}. Recomenda-se análise detalhada dos parâmetros específicos.")
        
        elif self.prng_candidates[0]['confidence'] > 0.4:
            recommendations.append("Confiança moderada. Recomenda-se análise adicional com mais dados ou métodos alternativos.")
        
        else:
            recommendations.append("Baixa confiança na detecção. Sistema provavelmente usa algoritmo complexo ou múltiplas camadas.")
        
        # Recomendações específicas por tipo
        for candidate in self.prng_candidates[:3]:
            if candidate['type'] == 'LCG' and candidate['confidence'] > 0.5:
                recommendations.append("Para LCG detectado: implementar busca refinada de parâmetros (a, c, m) com validação estendida.")
            
            elif candidate['type'] == 'Hybrid' and candidate['confidence'] > 0.4:
                recommendations.append("Sistema híbrido detectado: analisar combinações temporais e mudanças de regime.")
            
            elif candidate['type'] == 'ML_Based' and candidate['confidence'] > 0.3:
                recommendations.append("Padrões detectáveis por ML: considerar análise neural profunda ou redes adversárias.")
        
        return recommendations

# Script de execução
if __name__ == "__main__":
    from seed_discovery_engine import SeedDiscoveryEngine
    
    print("🔧 ENGENHARIA REVERSA MULTI-PRNG")
    print("="*70)
    
    # Carregar dados
    engine = SeedDiscoveryEngine()
    data_path = "../data/MegaSena3.xlsx"
    engine.load_megasena_data(data_path)
    
    # Inicializar engenheiro reverso
    reverse_engineer = MultiPRNGReverseEngineer(engine.historical_data)
    
    # Executar análise completa
    reverse_engineer.reverse_engineer_all_prngs()
    
    # Gerar relatório
    reverse_engineer.generate_comprehensive_report()
    
    print("\n✅ Engenharia reversa completa!")
    
    if reverse_engineer.prng_candidates:
        print(f"\n🎯 Melhor candidato: {reverse_engineer.prng_candidates[0]['type']}")
        print(f"   Confiança: {reverse_engineer.prng_candidates[0]['confidence']:.3f}")
    else:
        print("\n⚠️ Nenhum PRNG padrão identificado com alta confiança")