#!/usr/bin/env python3
"""
Analisador Avançado de Padrões
Implementa técnicas sofisticadas para descoberta de seed
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from datetime import datetime
import json

class AdvancedPatternAnalyzer:
    def __init__(self, engine):
        self.engine = engine
        self.analysis_results = {}
        
    def analyze_temporal_patterns(self):
        """Análise de padrões temporais nos dados"""
        print("\n📈 Analisando padrões temporais...")
        
        if not self.engine.data_loaded:
            return None
        
        # Extrair série temporal de médias
        time_series = []
        for draw_info in self.engine.historical_data:
            mean_value = np.mean(draw_info['numbers'])
            time_series.append(mean_value)
        
        time_series = np.array(time_series)
        
        # 1. Análise de Fourier
        fft_values = fft(time_series)
        frequencies = fftfreq(len(time_series))
        
        # Encontrar frequências dominantes
        power_spectrum = np.abs(fft_values) ** 2
        dominant_freq_idx = np.argsort(power_spectrum)[-10:]
        
        # 2. Detecção de pontos de mudança
        change_points = self.detect_change_points(time_series)
        
        # 3. Análise de autocorrelação
        autocorr = signal.correlate(time_series, time_series, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0]
        
        # Encontrar picos de autocorrelação
        peaks, _ = signal.find_peaks(autocorr, height=0.5)
        
        self.analysis_results['temporal'] = {
            'dominant_frequencies': frequencies[dominant_freq_idx].tolist(),
            'change_points': change_points,
            'autocorr_peaks': peaks.tolist(),
            'mean_series': time_series.tolist()
        }
        
        print(f"   ✓ {len(change_points)} pontos de mudança detectados")
        print(f"   ✓ {len(peaks)} picos de autocorrelação encontrados")
        
        return self.analysis_results['temporal']
    
    def detect_change_points(self, series):
        """Detecção avançada de pontos de mudança"""
        from scipy.stats import ttest_ind
        
        window_size = 50
        change_points = []
        
        for i in range(window_size, len(series) - window_size, 10):
            # Teste t entre janelas antes e depois
            before = series[i-window_size:i]
            after = series[i:i+window_size]
            
            _, p_value = ttest_ind(before, after)
            
            if p_value < 0.01:  # Mudança significativa
                change_points.append(i)
        
        # Consolidar pontos próximos
        consolidated = []
        for cp in change_points:
            if not consolidated or cp - consolidated[-1] > 20:
                consolidated.append(cp)
        
        return consolidated
    
    def analyze_number_transitions(self):
        """Analisa transições entre números consecutivos"""
        print("\n🔄 Analisando transições de números...")
        
        if not self.engine.data_loaded:
            return None
        
        # Matriz de transição
        transition_matrix = np.zeros((60, 60))
        
        for i in range(len(self.engine.historical_data) - 1):
            current = self.engine.historical_data[i]['numbers']
            next_draw = self.engine.historical_data[i + 1]['numbers']
            
            # Contar transições
            for num1 in current:
                for num2 in next_draw:
                    transition_matrix[num1-1, num2-1] += 1
        
        # Normalizar
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / (row_sums[:, np.newaxis] + 1e-10)
        
        # Encontrar padrões anômalos
        anomalies = []
        threshold = np.mean(transition_matrix) + 2 * np.std(transition_matrix)
        
        for i in range(60):
            for j in range(60):
                if transition_matrix[i, j] > threshold:
                    anomalies.append({
                        'from': i + 1,
                        'to': j + 1,
                        'probability': float(transition_matrix[i, j])
                    })
        
        self.analysis_results['transitions'] = {
            'matrix': transition_matrix.tolist(),
            'anomalies': anomalies,
            'max_transition': float(np.max(transition_matrix)),
            'mean_transition': float(np.mean(transition_matrix))
        }
        
        print(f"   ✓ {len(anomalies)} transições anômalas detectadas")
        
        return self.analysis_results['transitions']
    
    def find_seed_signature(self, candidates):
        """Encontra assinatura única do seed correto"""
        print("\n🔐 Buscando assinatura do seed...")
        
        signatures = []
        
        for candidate in candidates[:10]:  # Top 10 candidatos
            seed = candidate['seed']
            c = candidate['c']
            
            # Gerar sequência teste
            test_length = 100
            current_state = seed
            sequence = []
            
            for _ in range(test_length):
                draw = self.engine.state_to_lottery_numbers(current_state, c)
                if draw:
                    sequence.append(draw)
                    for _ in range(6):
                        current_state = self.engine.lcg_next(current_state, c)
            
            if len(sequence) < test_length / 2:
                continue
            
            # Calcular características únicas
            signature = {
                'seed': seed,
                'c': c,
                'features': {}
            }
            
            # 1. Distribuição de frequências
            freq_dist = np.zeros(60)
            for draw in sequence:
                for num in draw:
                    freq_dist[num-1] += 1
            
            signature['features']['freq_entropy'] = stats.entropy(freq_dist)
            signature['features']['freq_std'] = float(np.std(freq_dist))
            
            # 2. Padrão de gaps
            gaps = []
            for num in range(1, 61):
                appearances = [i for i, draw in enumerate(sequence) if num in draw]
                if len(appearances) > 1:
                    num_gaps = [appearances[i+1] - appearances[i] 
                               for i in range(len(appearances)-1)]
                    gaps.extend(num_gaps)
            
            if gaps:
                signature['features']['mean_gap'] = float(np.mean(gaps))
                signature['features']['gap_variance'] = float(np.var(gaps))
            
            # 3. Correlação média
            correlations = []
            for i in range(len(sequence) - 1):
                corr = self.engine.calculate_correlation(sequence[i], sequence[i+1])
                correlations.append(corr)
            
            signature['features']['mean_correlation'] = float(np.mean(correlations))
            signature['features']['corr_stability'] = float(np.std(correlations))
            
            # Calcular score de proximidade com padrão esperado
            target_corr_diff = abs(signature['features']['mean_correlation'] - 0.754)
            signature['match_score'] = 1 / (1 + target_corr_diff)
            
            signatures.append(signature)
        
        # Ordenar por match_score
        signatures.sort(key=lambda x: x['match_score'], reverse=True)
        
        self.analysis_results['signatures'] = signatures
        
        if signatures:
            best = signatures[0]
            print(f"   ✓ Melhor assinatura encontrada:")
            print(f"     Seed: {best['seed']}, c: {best['c']}")
            print(f"     Correlação média: {best['features']['mean_correlation']:.3f}")
            print(f"     Score de match: {best['match_score']:.3f}")
        
        return signatures
    
    def generate_validation_report(self, best_candidate):
        """Gera relatório detalhado de validação"""
        print("\n📋 Gerando relatório de validação...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'best_candidate': best_candidate,
            'analysis_summary': self.analysis_results,
            'validation_metrics': {}
        }
        
        # Validação extensa
        seed = best_candidate['seed']
        c = best_candidate['c']
        
        # 1. Teste de correlação em toda a sequência
        print("   Testando correlação completa...")
        full_validation = self.engine.validate_recent_draws(seed, c, 
                                                          len(self.engine.historical_data))
        report['validation_metrics']['full_sequence'] = full_validation
        
        # 2. Teste por períodos
        periods = [100, 500, 1000]
        for period in periods:
            if period < len(self.engine.historical_data):
                validation = self.engine.validate_recent_draws(seed, c, period)
                report['validation_metrics'][f'last_{period}'] = validation
        
        # 3. Previsões futuras
        predictions = self.engine.generate_predictions(seed, c, 20)
        report['future_predictions'] = predictions
        
        # Salvar relatório
        report_path = f"../output/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   ✓ Relatório salvo em: {report_path}")
        
        return report
    
    def visualize_analysis(self):
        """Cria visualizações dos padrões encontrados"""
        print("\n📊 Gerando visualizações...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Série temporal com pontos de mudança
        ax1 = axes[0, 0]
        if 'temporal' in self.analysis_results:
            temporal = self.analysis_results['temporal']
            ax1.plot(temporal['mean_series'], alpha=0.7)
            
            for cp in temporal['change_points']:
                ax1.axvline(x=cp, color='red', linestyle='--', alpha=0.5)
            
            ax1.set_title('Série Temporal - Média dos Sorteios')
            ax1.set_xlabel('Concurso')
            ax1.set_ylabel('Média')
        
        # 2. Autocorrelação
        ax2 = axes[0, 1]
        if 'temporal' in self.analysis_results:
            # Recalcular autocorrelação para visualização
            time_series = np.array(temporal['mean_series'])
            autocorr = signal.correlate(time_series, time_series, mode='full')
            autocorr = autocorr[len(autocorr)//2:][:200]  # Primeiros 200 lags
            autocorr /= autocorr[0]
            
            ax2.plot(autocorr)
            ax2.set_title('Função de Autocorrelação')
            ax2.set_xlabel('Lag')
            ax2.set_ylabel('Correlação')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 3. Matriz de transição (heatmap parcial)
        ax3 = axes[1, 0]
        if 'transitions' in self.analysis_results:
            matrix = np.array(self.analysis_results['transitions']['matrix'])
            # Mostrar apenas primeiros 20x20
            im = ax3.imshow(matrix[:20, :20], cmap='hot', aspect='auto')
            ax3.set_title('Matriz de Transição (20x20)')
            ax3.set_xlabel('Número Seguinte')
            ax3.set_ylabel('Número Atual')
            plt.colorbar(im, ax=ax3)
        
        # 4. Distribuição de correlações
        ax4 = axes[1, 1]
        correlations = []
        for i in range(len(self.engine.historical_data) - 1):
            draw1 = self.engine.historical_data[i]['numbers']
            draw2 = self.engine.historical_data[i + 1]['numbers']
            corr = self.engine.calculate_correlation(draw1, draw2)
            correlations.append(corr)
        
        ax4.hist(correlations, bins=50, alpha=0.7, density=True)
        ax4.axvline(x=0.754, color='red', linestyle='--', 
                   label='Alvo: 0.754', linewidth=2)
        ax4.set_title('Distribuição de Correlações')
        ax4.set_xlabel('Correlação')
        ax4.set_ylabel('Densidade')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('../output/pattern_analysis.png', dpi=300)
        plt.close()
        
        print("   ✓ Visualizações salvas em: output/pattern_analysis.png")

# Script de execução
if __name__ == "__main__":
    from seed_discovery_engine import SeedDiscoveryEngine
    
    print("🔬 ANÁLISE AVANÇADA DE PADRÕES")
    print("="*60)
    
    # Inicializar engine
    engine = SeedDiscoveryEngine()
    
    # Carregar dados
    data_path = "../data/MegaSena3.xlsx"
    engine.load_megasena_data(data_path)
    
    # Inicializar analisador
    analyzer = AdvancedPatternAnalyzer(engine)
    
    # Executar análises
    analyzer.analyze_temporal_patterns()
    analyzer.analyze_number_transitions()
    
    # Buscar seeds primeiro
    print("\n🔍 Buscando seeds candidatos...")
    candidates = engine.search_seed_at_change_point(engine.CHANGE_POINTS[0])
    
    if candidates:
        # Encontrar assinatura
        signatures = analyzer.find_seed_signature(candidates)
        
        if signatures:
            # Gerar relatório para o melhor
            best = signatures[0]
            best_candidate = {
                'seed': best['seed'],
                'c': best['c'],
                'signature': best
            }
            analyzer.generate_validation_report(best_candidate)
    
    # Gerar visualizações
    analyzer.visualize_analysis()
    
    print("\n✅ Análise completa!")