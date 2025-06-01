#!/usr/bin/env python3
"""
Mega Sena Quantum Pseudo-Random Analysis
========================================
Advanced analysis tool for detecting patterns in Mega Sena lottery data
using statistical methods and quantum-inspired algorithms.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import itertools
from collections import Counter, defaultdict
import hashlib
import json

warnings.filterwarnings('ignore')

class MegaSenaAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.number_sequences = None
        self.analysis_results = {}
        
    def load_data(self):
        """Load and preprocess the Mega Sena data"""
        print("=== LOADING MEGA SENA DATA ===")
        self.df = pd.read_excel(self.file_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        
        # Display first few rows to understand structure
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        # Extract number columns (assuming they contain the drawn numbers)
        number_cols = []
        for col in self.df.columns:
            if any(keyword in str(col).lower() for keyword in ['num', 'ball', 'bola', 'dezena']):
                number_cols.append(col)
        
        if not number_cols:
            # If no obvious number columns, assume they are the first 6 numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            number_cols = numeric_cols[:6].tolist()
        
        print(f"\nIdentified number columns: {number_cols}")
        
        # Extract sequences
        if len(number_cols) >= 6:
            self.number_sequences = self.df[number_cols[:6]].values
        else:
            print("Warning: Could not identify 6 number columns")
            self.number_sequences = self.df.select_dtypes(include=[np.number]).iloc[:, :6].values
        
        print(f"Extracted {len(self.number_sequences)} drawing sequences")
        print(f"Sample sequence: {self.number_sequences[0] if len(self.number_sequences) > 0 else 'None'}")
        
        return self.df
    
    def basic_statistics(self):
        """Perform basic statistical analysis"""
        print("\n=== BASIC STATISTICAL ANALYSIS ===")
        
        results = {
            'total_drawings': len(self.number_sequences),
            'number_range': (int(np.min(self.number_sequences)), int(np.max(self.number_sequences))),
            'mean_numbers': np.mean(self.number_sequences, axis=0).tolist(),
            'std_numbers': np.std(self.number_sequences, axis=0).tolist()
        }
        
        print(f"Total drawings: {results['total_drawings']}")
        print(f"Number range: {results['number_range']}")
        print(f"Mean values per position: {[f'{x:.2f}' for x in results['mean_numbers']]}")
        print(f"Standard deviation per position: {[f'{x:.2f}' for x in results['std_numbers']]}")
        
        # Frequency analysis
        all_numbers = self.number_sequences.flatten()
        freq_dist = Counter(all_numbers)
        most_common = freq_dist.most_common(10)
        least_common = freq_dist.most_common()[-10:]
        
        print(f"\nMost frequent numbers: {most_common}")
        print(f"Least frequent numbers: {least_common}")
        
        results['frequency_distribution'] = {int(k): int(v) for k, v in freq_dist.items()}
        results['most_common'] = [(int(num), int(freq)) for num, freq in most_common]
        results['least_common'] = [(int(num), int(freq)) for num, freq in least_common]
        
        self.analysis_results['basic_stats'] = results
        return results
    
    def randomness_tests(self):
        """Perform statistical tests for randomness"""
        print("\n=== RANDOMNESS TESTS ===")
        
        results = {}
        
        # Chi-square test for uniform distribution
        all_numbers = self.number_sequences.flatten()
        unique_numbers = np.unique(all_numbers)
        observed_freq = np.bincount(all_numbers.astype(int))[1:]  # Skip 0 if present
        expected_freq = len(all_numbers) / len(unique_numbers)
        
        chi2_stat, chi2_p = stats.chisquare(observed_freq)
        results['chi_square'] = {'statistic': float(chi2_stat), 'p_value': float(chi2_p)}
        
        print(f"Chi-square test: stat={chi2_stat:.4f}, p-value={chi2_p:.6f}")
        
        # Kolmogorov-Smirnov test against uniform distribution
        uniform_data = np.random.uniform(np.min(all_numbers), np.max(all_numbers), len(all_numbers))
        ks_stat, ks_p = stats.ks_2samp(all_numbers, uniform_data)
        results['kolmogorov_smirnov'] = {'statistic': float(ks_stat), 'p_value': float(ks_p)}
        
        print(f"Kolmogorov-Smirnov test: stat={ks_stat:.4f}, p-value={ks_p:.6f}")
        
        # Runs test for randomness
        def runs_test(sequence):
            median = np.median(sequence)
            runs, n1, n2 = 0, 0, 0
            
            # Convert to binary sequence (above/below median)
            binary_seq = sequence > median
            
            # Count runs
            for i in range(len(binary_seq)):
                if binary_seq[i]:
                    n1 += 1
                else:
                    n2 += 1
                    
                if i == 0 or binary_seq[i] != binary_seq[i-1]:
                    runs += 1
            
            # Calculate expected runs and variance
            expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
            variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
            
            if variance > 0:
                z_score = (runs - expected_runs) / np.sqrt(variance)
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                z_score, p_value = 0, 1
                
            return runs, expected_runs, z_score, p_value
        
        runs, exp_runs, z_score, runs_p = runs_test(all_numbers)
        results['runs_test'] = {
            'runs': int(runs), 
            'expected_runs': float(exp_runs), 
            'z_score': float(z_score), 
            'p_value': float(runs_p)
        }
        
        print(f"Runs test: runs={runs}, expected={exp_runs:.2f}, z={z_score:.4f}, p={runs_p:.6f}")
        
        self.analysis_results['randomness_tests'] = results
        return results
    
    def sequential_analysis(self):
        """Analyze sequential patterns and correlations"""
        print("\n=== SEQUENTIAL PATTERN ANALYSIS ===")
        
        results = {}
        
        # Auto-correlation analysis
        correlations = []
        for pos in range(6):
            series = self.number_sequences[:, pos]
            autocorr = [np.corrcoef(series[:-lag], series[lag:])[0,1] for lag in range(1, min(50, len(series)//2))]
            correlations.append(autocorr)
        
        results['autocorrelations'] = correlations
        max_autocorr = np.max([np.max(np.abs(corr)) for corr in correlations])
        print(f"Maximum autocorrelation: {max_autocorr:.4f}")
        
        # Gap analysis between consecutive drawings
        gaps = []
        for i in range(1, len(self.number_sequences)):
            gap = np.sum(np.abs(self.number_sequences[i] - self.number_sequences[i-1]))
            gaps.append(gap)
        
        results['gaps'] = {
            'mean_gap': float(np.mean(gaps)),
            'std_gap': float(np.std(gaps)),
            'min_gap': int(np.min(gaps)),
            'max_gap': int(np.max(gaps))
        }
        
        print(f"Gap analysis - Mean: {np.mean(gaps):.2f}, Std: {np.std(gaps):.2f}")
        
        # Pattern repetition analysis
        sequence_hashes = []
        for seq in self.number_sequences:
            seq_str = ','.join(map(str, sorted(seq)))
            seq_hash = hashlib.md5(seq_str.encode()).hexdigest()
            sequence_hashes.append(seq_hash)
        
        unique_sequences = len(set(sequence_hashes))
        total_sequences = len(sequence_hashes)
        repetition_rate = 1 - (unique_sequences / total_sequences)
        
        results['pattern_repetition'] = {
            'unique_sequences': int(unique_sequences),
            'total_sequences': int(total_sequences),
            'repetition_rate': float(repetition_rate)
        }
        
        print(f"Pattern repetition: {repetition_rate:.6f} ({total_sequences - unique_sequences} repeats)")
        
        self.analysis_results['sequential_analysis'] = results
        return results
    
    def fourier_analysis(self):
        """Perform Fourier analysis to detect periodic patterns"""
        print("\n=== FOURIER ANALYSIS FOR PERIODIC PATTERNS ===")
        
        results = {}
        
        for pos in range(6):
            series = self.number_sequences[:, pos]
            
            # Apply FFT
            fft_values = fft(series)
            frequencies = fftfreq(len(series))
            
            # Find dominant frequencies
            power_spectrum = np.abs(fft_values)**2
            dominant_freq_idx = np.argsort(power_spectrum)[-10:]  # Top 10 frequencies
            dominant_freqs = frequencies[dominant_freq_idx]
            dominant_powers = power_spectrum[dominant_freq_idx]
            
            results[f'position_{pos}'] = {
                'dominant_frequencies': dominant_freqs.tolist(),
                'dominant_powers': dominant_powers.tolist()
            }
        
        print("Fourier analysis completed - checking for periodic patterns")
        
        self.analysis_results['fourier_analysis'] = results
        return results
    
    def quantum_inspired_analysis(self):
        """Quantum-inspired analysis using superposition and entanglement concepts"""
        print("\n=== QUANTUM-INSPIRED ANALYSIS ===")
        
        results = {}
        
        # Quantum superposition simulation: analyze number combinations as quantum states
        def quantum_state_analysis():
            # Create probability amplitude matrix
            max_num = int(np.max(self.number_sequences))
            amplitude_matrix = np.zeros((len(self.number_sequences), max_num + 1))
            
            for i, seq in enumerate(self.number_sequences):
                for num in seq:
                    amplitude_matrix[i, int(num)] = 1/np.sqrt(6)  # Equal superposition
            
            # Calculate interference patterns
            interference = np.zeros(max_num + 1)
            for i in range(len(amplitude_matrix) - 1):
                interference += np.abs(amplitude_matrix[i] + amplitude_matrix[i+1])**2
            
            return interference / len(amplitude_matrix)
        
        interference_pattern = quantum_state_analysis()
        results['quantum_interference'] = interference_pattern.tolist()
        
        # Entanglement analysis: check for correlations between number positions
        entanglement_matrix = np.corrcoef(self.number_sequences.T)
        results['entanglement_matrix'] = entanglement_matrix.tolist()
        
        max_entanglement = np.max(np.abs(entanglement_matrix - np.eye(6)))
        print(f"Maximum quantum entanglement: {max_entanglement:.4f}")
        
        # Quantum phase analysis
        phases = []
        for seq in self.number_sequences:
            phase = np.sum(seq * np.exp(2j * np.pi * np.arange(6) / 6))
            phases.append(np.angle(phase))
        
        phase_coherence = np.std(phases)
        results['phase_coherence'] = float(phase_coherence)
        
        print(f"Quantum phase coherence: {phase_coherence:.4f}")
        
        self.analysis_results['quantum_analysis'] = results
        return results
    
    def prng_seed_detection(self):
        """Attempt to detect PRNG patterns and potential seeds"""
        print("\n=== PRNG SEED DETECTION ANALYSIS ===")
        
        results = {}
        
        # Linear Congruential Generator (LCG) pattern detection
        def detect_lcg_patterns():
            lcg_candidates = []
            
            # Try different modulus values
            for m in [2**31 - 1, 2**32, 2**31]:  # Common LCG modulus values
                for a in range(1, min(1000, m)):  # Multiplier
                    for c in range(0, min(100, m)):  # Increment
                        pattern_match = 0
                        for i in range(1, min(100, len(self.number_sequences))):
                            predicted = (a * np.sum(self.number_sequences[i-1]) + c) % m
                            actual = np.sum(self.number_sequences[i])
                            if abs(predicted - actual) < m * 0.01:  # 1% tolerance
                                pattern_match += 1
                        
                        if pattern_match > 10:  # Threshold for considering a match
                            lcg_candidates.append({
                                'a': a, 'c': c, 'm': m, 
                                'matches': pattern_match,
                                'confidence': pattern_match / min(100, len(self.number_sequences))
                            })
            
            return sorted(lcg_candidates, key=lambda x: x['confidence'], reverse=True)[:5]
        
        lcg_results = detect_lcg_patterns()
        results['lcg_candidates'] = lcg_results
        
        if lcg_results:
            print(f"Found {len(lcg_results)} potential LCG patterns")
            for i, lcg in enumerate(lcg_results[:3]):
                print(f"  LCG {i+1}: a={lcg['a']}, c={lcg['c']}, m={lcg['m']}, confidence={lcg['confidence']:.3f}")
        else:
            print("No strong LCG patterns detected")
        
        # Mersenne Twister pattern detection (simplified)
        def detect_mt_patterns():
            # Look for 32-bit patterns in the sequence
            mt_candidates = []
            
            for seed in range(1, 10000, 100):  # Sample seed space
                np.random.seed(seed)
                synthetic_seq = []
                for _ in range(min(1000, len(self.number_sequences))):
                    draw = sorted(np.random.choice(60, 6, replace=False) + 1)
                    synthetic_seq.append(draw)
                
                # Compare with actual sequence
                matches = 0
                for i in range(min(len(synthetic_seq), len(self.number_sequences))):
                    if np.array_equal(synthetic_seq[i], sorted(self.number_sequences[i])):
                        matches += 1
                
                if matches > 0:
                    mt_candidates.append({
                        'seed': seed,
                        'matches': matches,
                        'confidence': matches / min(len(synthetic_seq), len(self.number_sequences))
                    })
            
            return sorted(mt_candidates, key=lambda x: x['confidence'], reverse=True)[:5]
        
        mt_results = detect_mt_patterns()
        results['mersenne_twister_candidates'] = mt_results
        
        if mt_results:
            print(f"Found {len(mt_results)} potential Mersenne Twister patterns")
            for i, mt in enumerate(mt_results[:3]):
                print(f"  MT {i+1}: seed={mt['seed']}, matches={mt['matches']}, confidence={mt['confidence']:.6f}")
        else:
            print("No Mersenne Twister patterns detected")
        
        self.analysis_results['prng_detection'] = results
        return results
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE MEGA SENA ANALYSIS REPORT")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = {
            'timestamp': timestamp,
            'analysis_summary': {},
            'detailed_results': self.analysis_results,
            'conclusions': [],
            'recommendations': []
        }
        
        # Summarize key findings
        if 'basic_stats' in self.analysis_results:
            report['analysis_summary']['total_drawings'] = self.analysis_results['basic_stats']['total_drawings']
            report['analysis_summary']['number_range'] = self.analysis_results['basic_stats']['number_range']
        
        if 'randomness_tests' in self.analysis_results:
            chi2_p = self.analysis_results['randomness_tests']['chi_square']['p_value']
            ks_p = self.analysis_results['randomness_tests']['kolmogorov_smirnov']['p_value']
            runs_p = self.analysis_results['randomness_tests']['runs_test']['p_value']
            
            report['analysis_summary']['randomness_test_p_values'] = {
                'chi_square': chi2_p,
                'kolmogorov_smirnov': ks_p,
                'runs_test': runs_p
            }
            
            # Conclusions based on p-values
            if chi2_p < 0.05:
                report['conclusions'].append("Chi-square test suggests non-uniform distribution (p < 0.05)")
            else:
                report['conclusions'].append("Chi-square test does not reject uniform distribution hypothesis")
            
            if any(p < 0.01 for p in [chi2_p, ks_p, runs_p]):
                report['conclusions'].append("Strong evidence against true randomness detected")
            elif any(p < 0.05 for p in [chi2_p, ks_p, runs_p]):
                report['conclusions'].append("Moderate evidence against true randomness detected")
            else:
                report['conclusions'].append("No strong evidence against randomness found")
        
        if 'prng_detection' in self.analysis_results:
            lcg_candidates = self.analysis_results['prng_detection'].get('lcg_candidates', [])
            mt_candidates = self.analysis_results['prng_detection'].get('mersenne_twister_candidates', [])
            
            if lcg_candidates and lcg_candidates[0]['confidence'] > 0.1:
                report['conclusions'].append(f"Potential LCG pattern detected with {lcg_candidates[0]['confidence']:.1%} confidence")
                report['recommendations'].append("Investigate LCG parameters further")
            
            if mt_candidates and mt_candidates[0]['confidence'] > 0.001:
                report['conclusions'].append(f"Potential Mersenne Twister pattern detected")
                report['recommendations'].append("Test additional Mersenne Twister seeds")
        
        if 'quantum_analysis' in self.analysis_results:
            max_entanglement = np.max(np.abs(np.array(self.analysis_results['quantum_analysis']['entanglement_matrix']) - np.eye(6)))
            if max_entanglement > 0.3:
                report['conclusions'].append(f"Significant quantum entanglement detected: {max_entanglement:.3f}")
                report['recommendations'].append("Investigate position correlations using quantum algorithms")
        
        # General recommendations
        if not report['recommendations']:
            report['recommendations'].append("Data appears genuinely random - consider alternative analysis methods")
            report['recommendations'].append("Increase sample size for more robust statistical tests")
        
        # Save report
        report_filename = f'/Users/alebrotto/Downloads/quantum_mega_pseudo-aleatorio/analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nREPORT SAVED: {report_filename}")
        print("\nKEY FINDINGS:")
        for conclusion in report['conclusions']:
            print(f"  • {conclusion}")
        
        print("\nRECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        return report

def main():
    # Initialize analyzer
    file_path = '/Users/alebrotto/Downloads/quantum_mega_pseudo-aleatorio/Mega-Sena-3.xlsx'
    analyzer = MegaSenaAnalyzer(file_path)
    
    try:
        # Load and analyze data
        analyzer.load_data()
        analyzer.basic_statistics()
        analyzer.randomness_tests()
        analyzer.sequential_analysis()
        analyzer.fourier_analysis()
        analyzer.quantum_inspired_analysis()
        analyzer.prng_seed_detection()
        
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report()
        
        return analyzer, report
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    analyzer, report = main()