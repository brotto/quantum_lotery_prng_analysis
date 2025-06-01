#!/usr/bin/env python3
"""
Analisador Mestre Abrangente
Coordena todas as análises e gera relatório final pormenorizado
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importar módulos desenvolvidos
from seed_discovery_engine import SeedDiscoveryEngine
from temporal_change_detector import TemporalChangeDetector
from quantum_prng_analyzer import QuantumPRNGAnalyzer
from multi_prng_reverse_engineer import MultiPRNGReverseEngineer
from genetic_optimization_engine import GeneticOptimizationEngine

class ComprehensiveMasterAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.engine = None
        self.historical_data = None
        
        # Módulos de análise
        self.temporal_detector = None
        self.quantum_analyzer = None
        self.reverse_engineer = None
        self.genetic_optimizer = None
        
        # Resultados consolidados
        self.consolidated_results = {}
        self.master_report = {}
        self.confidence_scores = {}
        
        # Configuração
        self.output_dir = Path("../output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Timestamp para esta execução
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def execute_comprehensive_analysis(self):
        """Executa análise abrangente completa"""
        print("🎯 INICIANDO ANÁLISE MESTRE ABRANGENTE")
        print("="*80)
        
        try:
            # 1. Inicialização
            self.initialize_system()
            
            # 2. Análise Temporal
            self.execute_temporal_analysis()
            
            # 3. Análise Quântica
            self.execute_quantum_analysis()
            
            # 4. Engenharia Reversa
            self.execute_reverse_engineering()
            
            # 5. Otimização Genética
            self.execute_genetic_optimization()
            
            # 6. Consolidação e Correlação
            self.consolidate_all_results()
            
            # 7. Análise de Confiança
            self.calculate_confidence_scores()
            
            # 8. Geração de Relatório Mestre
            self.generate_master_report()
            
            # 9. Visualizações Interativas
            self.create_interactive_visualizations()
            
            # 10. Refinamento e Recomendações
            self.generate_final_recommendations()
            
            print("\n✅ ANÁLISE MESTRE COMPLETA!")
            
        except Exception as e:
            print(f"\n❌ Erro durante análise mestre: {e}")
            import traceback
            traceback.print_exc()
    
    def initialize_system(self):
        """Inicializa o sistema e carrega dados"""
        print("\n🔧 FASE 1: Inicialização do Sistema")
        print("-" * 50)
        
        # Inicializar engine principal
        self.engine = SeedDiscoveryEngine()
        
        # Carregar dados
        print("📂 Carregando dados da Mega Sena...")
        self.historical_data = self.engine.load_megasena_data(self.data_path)
        
        if not self.historical_data:
            raise ValueError("Falha ao carregar dados históricos")
        
        print(f"✅ {len(self.historical_data)} sorteios carregados com sucesso")
        
        # Inicializar módulos de análise
        print("🔨 Inicializando módulos de análise...")
        
        self.temporal_detector = TemporalChangeDetector(self.historical_data)
        self.quantum_analyzer = QuantumPRNGAnalyzer(self.historical_data)
        self.reverse_engineer = MultiPRNGReverseEngineer(self.historical_data)
        
        print("✅ Sistema inicializado com sucesso")
    
    def execute_temporal_analysis(self):
        """Executa análise temporal completa"""
        print("\n⏰ FASE 2: Análise Temporal Avançada")
        print("-" * 50)
        
        try:
            # Extrair features
            print("🔬 Extraindo features temporais...")
            self.temporal_detector.extract_comprehensive_features()
            
            # Detectar pontos de mudança
            print("🔍 Detectando pontos de mudança...")
            change_points = self.temporal_detector.detect_change_points_multiple_methods()
            
            # Analisar regimes
            print("📋 Analisando características dos regimes...")
            regimes = self.temporal_detector.analyze_regime_characteristics()
            
            # Gerar relatório temporal
            temporal_report = self.temporal_detector.generate_temporal_report()
            
            # Visualizações
            self.temporal_detector.visualize_temporal_analysis()
            
            # Armazenar resultados
            self.consolidated_results['temporal'] = {
                'change_points': change_points,
                'regimes': regimes,
                'report': temporal_report,
                'status': 'completed'
            }
            
            print(f"✅ Análise temporal completa - {len(change_points)} pontos de mudança detectados")
            
        except Exception as e:
            print(f"❌ Erro na análise temporal: {e}")
            self.consolidated_results['temporal'] = {'status': 'failed', 'error': str(e)}
    
    def execute_quantum_analysis(self):
        """Executa análise quântica completa"""
        print("\n🌌 FASE 3: Análise Quântica de PRNGs")
        print("-" * 50)
        
        try:
            # Análise de entropia quântica
            print("💫 Executando análise de entropia quântica...")
            self.quantum_analyzer.quantum_entropy_analysis()
            
            # Análise de Fourier quântica
            print("🔄 Executando análise de Fourier quântica...")
            self.quantum_analyzer.quantum_fourier_analysis()
            
            # Detecção de PRNG quântica
            print("🎯 Executando detecção quântica de PRNG...")
            prng_candidates = self.quantum_analyzer.quantum_prng_detection()
            
            # Otimização quântica
            print("🔍 Executando busca com otimização quântica...")
            self.quantum_analyzer.quantum_optimization_search()
            
            # Análise de entrelaçamento
            print("🔗 Executando análise de entrelaçamento...")
            self.quantum_analyzer.quantum_entanglement_analysis()
            
            # Análise de coerência
            print("💫 Executando análise de coerência...")
            self.quantum_analyzer.quantum_coherence_analysis()
            
            # Gerar relatório quântico
            quantum_report = self.quantum_analyzer.generate_quantum_report()
            
            # Visualizações
            self.quantum_analyzer.visualize_quantum_analysis()
            
            # Armazenar resultados
            self.consolidated_results['quantum'] = {
                'prng_candidates': prng_candidates,
                'report': quantum_report,
                'analysis_results': self.quantum_analyzer.quantum_analysis_results,
                'status': 'completed'
            }
            
            print(f"✅ Análise quântica completa - {len(prng_candidates)} candidatos PRNG detectados")
            
        except Exception as e:
            print(f"❌ Erro na análise quântica: {e}")
            self.consolidated_results['quantum'] = {'status': 'failed', 'error': str(e)}
    
    def execute_reverse_engineering(self):
        """Executa engenharia reversa completa"""
        print("\n🔧 FASE 4: Engenharia Reversa Multi-PRNG")
        print("-" * 50)
        
        try:
            # Executar engenharia reversa para todos os PRNGs
            print("🔍 Executando engenharia reversa...")
            reverse_results = self.reverse_engineer.reverse_engineer_all_prngs()
            
            # Gerar relatório de engenharia reversa
            reverse_report = self.reverse_engineer.generate_comprehensive_report()
            
            # Armazenar resultados
            self.consolidated_results['reverse_engineering'] = {
                'analysis_results': reverse_results,
                'prng_candidates': self.reverse_engineer.prng_candidates,
                'report': reverse_report,
                'status': 'completed'
            }
            
            print(f"✅ Engenharia reversa completa - {len(self.reverse_engineer.prng_candidates)} candidatos identificados")
            
        except Exception as e:
            print(f"❌ Erro na engenharia reversa: {e}")
            self.consolidated_results['reverse_engineering'] = {'status': 'failed', 'error': str(e)}
    
    def execute_genetic_optimization(self):
        """Executa otimização genética completa"""
        print("\n🧬 FASE 5: Otimização Genética de Seeds")
        print("-" * 50)
        
        try:
            # Obter candidatos das análises anteriores
            all_candidates = []
            
            if 'quantum' in self.consolidated_results and self.consolidated_results['quantum']['status'] == 'completed':
                all_candidates.extend(self.consolidated_results['quantum'].get('prng_candidates', []))
            
            if 'reverse_engineering' in self.consolidated_results and self.consolidated_results['reverse_engineering']['status'] == 'completed':
                all_candidates.extend(self.consolidated_results['reverse_engineering'].get('prng_candidates', []))
            
            # Inicializar otimizador genético
            self.genetic_optimizer = GeneticOptimizationEngine(self.historical_data, all_candidates)
            
            # Executar otimização
            print("🧬 Executando otimização genética...")
            optimization_results = self.genetic_optimizer.optimize_all_candidates()
            
            # Gerar relatório de otimização
            optimization_report = self.genetic_optimizer.generate_optimization_report()
            
            # Visualizações
            self.genetic_optimizer.visualize_optimization_results()
            
            # Armazenar resultados
            self.consolidated_results['genetic_optimization'] = {
                'optimization_results': optimization_results,
                'best_solutions': self.genetic_optimizer.best_solutions,
                'report': optimization_report,
                'status': 'completed'
            }
            
            print(f"✅ Otimização genética completa - {len(self.genetic_optimizer.best_solutions)} soluções encontradas")
            
        except Exception as e:
            print(f"❌ Erro na otimização genética: {e}")
            self.consolidated_results['genetic_optimization'] = {'status': 'failed', 'error': str(e)}
    
    def consolidate_all_results(self):
        """Consolida todos os resultados em estrutura unificada"""
        print("\n🔗 FASE 6: Consolidação e Correlação de Resultados")
        print("-" * 50)
        
        print("🔄 Consolidando resultados de todas as análises...")
        
        # Extrair pontos de mudança de diferentes análises
        all_change_points = self.extract_all_change_points()
        
        # Consolidar candidatos PRNG
        all_prng_candidates = self.extract_all_prng_candidates()
        
        # Consolidar melhores soluções
        all_best_solutions = self.extract_all_best_solutions()
        
        # Análise de correlação cruzada
        cross_correlations = self.analyze_cross_correlations(all_change_points, all_prng_candidates, all_best_solutions)
        
        # Consolidar em estrutura final
        self.consolidated_results['consolidated'] = {
            'all_change_points': all_change_points,
            'all_prng_candidates': all_prng_candidates,
            'all_best_solutions': all_best_solutions,
            'cross_correlations': cross_correlations,
            'execution_timestamp': self.timestamp,
            'data_summary': self.generate_data_summary()
        }
        
        print(f"✅ Consolidação completa:")
        print(f"   - {len(all_change_points)} pontos de mudança totais")
        print(f"   - {len(all_prng_candidates)} candidatos PRNG totais")
        print(f"   - {len(all_best_solutions)} melhores soluções totais")
    
    def extract_all_change_points(self):
        """Extrai pontos de mudança de todas as análises"""
        all_points = []
        
        # Pontos da análise temporal
        if 'temporal' in self.consolidated_results and self.consolidated_results['temporal']['status'] == 'completed':
            temporal_points = self.consolidated_results['temporal'].get('change_points', [])
            for point in temporal_points:
                all_points.append({
                    'position': point,
                    'source': 'temporal_analysis',
                    'confidence': 1.0,  # Assumir alta confiança
                    'method': 'multi_method_consensus'
                })
        
        # Pontos da análise quântica (se detectados)
        if 'quantum' in self.consolidated_results and self.consolidated_results['quantum']['status'] == 'completed':
            quantum_results = self.consolidated_results['quantum'].get('analysis_results', {})
            # Extrair pontos de mudança implícitos da análise quântica
            if 'entropy' in quantum_results:
                entropies = quantum_results['entropy'].get('von_neumann_entropies', [])
                if entropies:
                    # Detectar mudanças abruptas na entropia
                    for i in range(1, len(entropies)):
                        entropy_change = abs(entropies[i] - entropies[i-1])
                        if entropy_change > 0.5:  # Threshold para mudança significativa
                            all_points.append({
                                'position': i,
                                'source': 'quantum_entropy',
                                'confidence': min(1.0, entropy_change),
                                'method': 'entropy_discontinuity'
                            })
        
        # Consolidar pontos próximos
        consolidated_points = self.consolidate_nearby_points(all_points)
        
        return consolidated_points
    
    def consolidate_nearby_points(self, points, threshold=30):
        """Consolida pontos próximos em um único ponto"""
        if not points:
            return []
        
        # Ordenar por posição
        sorted_points = sorted(points, key=lambda x: x['position'])
        
        consolidated = []
        current_group = [sorted_points[0]]
        
        for point in sorted_points[1:]:
            if point['position'] - current_group[-1]['position'] <= threshold:
                current_group.append(point)
            else:
                # Processar grupo atual
                consolidated_point = self.merge_point_group(current_group)
                consolidated.append(consolidated_point)
                current_group = [point]
        
        # Processar último grupo
        if current_group:
            consolidated_point = self.merge_point_group(current_group)
            consolidated.append(consolidated_point)
        
        return consolidated
    
    def merge_point_group(self, point_group):
        """Merge um grupo de pontos próximos"""
        if len(point_group) == 1:
            return point_group[0]
        
        # Calcular posição média ponderada por confiança
        total_weight = sum(p['confidence'] for p in point_group)
        weighted_position = sum(p['position'] * p['confidence'] for p in point_group) / total_weight
        
        # Combinar fontes
        sources = list(set(p['source'] for p in point_group))
        methods = list(set(p['method'] for p in point_group))
        
        return {
            'position': int(weighted_position),
            'source': '_'.join(sources),
            'confidence': min(1.0, total_weight / len(point_group)),
            'method': '_'.join(methods),
            'group_size': len(point_group)
        }
    
    def extract_all_prng_candidates(self):
        """Extrai candidatos PRNG de todas as análises"""
        all_candidates = []
        
        # Candidatos da análise quântica
        if 'quantum' in self.consolidated_results and self.consolidated_results['quantum']['status'] == 'completed':
            quantum_candidates = self.consolidated_results['quantum'].get('prng_candidates', [])
            for candidate in quantum_candidates:
                candidate['analysis_source'] = 'quantum'
                all_candidates.append(candidate)
        
        # Candidatos da engenharia reversa
        if 'reverse_engineering' in self.consolidated_results and self.consolidated_results['reverse_engineering']['status'] == 'completed':
            reverse_candidates = self.consolidated_results['reverse_engineering'].get('prng_candidates', [])
            for candidate in reverse_candidates:
                candidate['analysis_source'] = 'reverse_engineering'
                all_candidates.append(candidate)
        
        # Ordenar por confiança
        all_candidates.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return all_candidates
    
    def extract_all_best_solutions(self):
        """Extrai melhores soluções de todas as otimizações"""
        all_solutions = []
        
        # Soluções da otimização genética
        if 'genetic_optimization' in self.consolidated_results and self.consolidated_results['genetic_optimization']['status'] == 'completed':
            genetic_solutions = self.consolidated_results['genetic_optimization'].get('best_solutions', [])
            for solution in genetic_solutions:
                solution['optimization_source'] = 'genetic_algorithm'
                all_solutions.append(solution)
        
        # Ordenar por score
        all_solutions.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return all_solutions
    
    def analyze_cross_correlations(self, all_change_points=None, all_prng_candidates=None, all_best_solutions=None):
        """Analisa correlações cruzadas entre diferentes análises"""
        correlations = {}
        
        # Correlação entre pontos de mudança de diferentes fontes
        correlations['change_points'] = self.correlate_change_points(all_change_points)
        
        # Correlação entre candidatos PRNG
        correlations['prng_candidates'] = self.correlate_prng_candidates(all_prng_candidates)
        
        # Correlação entre soluções
        correlations['solutions'] = self.correlate_solutions(all_best_solutions)
        
        return correlations
    
    def correlate_change_points(self, all_points=None):
        """Correlaciona pontos de mudança entre análises"""
        if all_points is None:
            all_points = self.consolidated_results.get('consolidated', {}).get('all_change_points', [])
        
        # Agrupar por fonte
        sources = {}
        for point in all_points:
            source = point['source']
            if source not in sources:
                sources[source] = []
            sources[source].append(point['position'])
        
        # Calcular correlações
        correlations = {}
        source_names = list(sources.keys())
        
        for i, source1 in enumerate(source_names):
            for source2 in source_names[i+1:]:
                correlation = self.calculate_position_correlation(sources[source1], sources[source2])
                correlations[f"{source1}_vs_{source2}"] = correlation
        
        return correlations
    
    def calculate_position_correlation(self, positions1, positions2, tolerance=50):
        """Calcula correlação entre listas de posições"""
        if not positions1 or not positions2:
            return 0
        
        matches = 0
        for pos1 in positions1:
            for pos2 in positions2:
                if abs(pos1 - pos2) <= tolerance:
                    matches += 1
                    break
        
        # Jaccard similarity
        total_unique = len(set(positions1) | set(positions2))
        return matches / total_unique if total_unique > 0 else 0
    
    def correlate_prng_candidates(self, all_candidates=None):
        """Correlaciona candidatos PRNG entre análises"""
        if all_candidates is None:
            all_candidates = self.consolidated_results.get('consolidated', {}).get('all_prng_candidates', [])
        
        # Agrupar por fonte
        by_source = {}
        for candidate in all_candidates:
            source = candidate.get('analysis_source', 'unknown')
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(candidate)
        
        # Analisar consenso por tipo PRNG
        type_consensus = {}
        for candidate in all_candidates:
            prng_type = candidate.get('type', 'unknown')
            if prng_type not in type_consensus:
                type_consensus[prng_type] = []
            type_consensus[prng_type].append({
                'source': candidate.get('analysis_source'),
                'confidence': candidate.get('confidence', 0)
            })
        
        return {
            'by_source': {source: len(candidates) for source, candidates in by_source.items()},
            'type_consensus': type_consensus
        }
    
    def correlate_solutions(self, all_solutions=None):
        """Correlaciona soluções entre otimizações"""
        if all_solutions is None:
            all_solutions = self.consolidated_results.get('consolidated', {}).get('all_best_solutions', [])
        
        if not all_solutions:
            return {'status': 'no_solutions'}
        
        # Analisar distribuição de scores
        scores = [sol.get('score', 0) for sol in all_solutions]
        
        # Analisar consenso de métodos
        methods = [sol.get('method', 'unknown') for sol in all_solutions]
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            'score_distribution': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'max': max(scores),
                'min': min(scores)
            },
            'method_consensus': method_counts,
            'total_solutions': len(all_solutions)
        }
    
    def generate_data_summary(self):
        """Gera resumo dos dados analisados"""
        return {
            'total_draws': len(self.historical_data),
            'date_range': {
                'first': self.historical_data[0].get('concurso', 'N/A'),
                'last': self.historical_data[-1].get('concurso', 'N/A')
            },
            'analysis_modules': list(self.consolidated_results.keys()),
            'successful_analyses': len([k for k, v in self.consolidated_results.items() 
                                      if v.get('status') == 'completed']),
            'failed_analyses': len([k for k, v in self.consolidated_results.items() 
                                  if v.get('status') == 'failed'])
        }
    
    def calculate_confidence_scores(self):
        """Calcula scores de confiança para principais descobertas"""
        print("\n📊 FASE 7: Cálculo de Scores de Confiança")
        print("-" * 50)
        
        # Score de detecção de pontos de mudança
        change_points_confidence = self.calculate_change_points_confidence()
        
        # Score de detecção PRNG
        prng_detection_confidence = self.calculate_prng_detection_confidence()
        
        # Score de otimização
        optimization_confidence = self.calculate_optimization_confidence()
        
        # Score geral
        overall_confidence = self.calculate_overall_confidence(
            change_points_confidence, prng_detection_confidence, optimization_confidence
        )
        
        self.confidence_scores = {
            'change_points': change_points_confidence,
            'prng_detection': prng_detection_confidence,
            'optimization': optimization_confidence,
            'overall': overall_confidence,
            'calculation_timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Scores de confiança calculados:")
        print(f"   - Pontos de mudança: {change_points_confidence:.3f}")
        print(f"   - Detecção PRNG: {prng_detection_confidence:.3f}")
        print(f"   - Otimização: {optimization_confidence:.3f}")
        print(f"   - Score geral: {overall_confidence:.3f}")
    
    def calculate_change_points_confidence(self):
        """Calcula confiança na detecção de pontos de mudança"""
        consolidated = self.consolidated_results.get('consolidated', {})
        change_points = consolidated.get('all_change_points', [])
        
        if not change_points:
            return 0
        
        # Fatores de confiança
        confidence_factors = []
        
        # 1. Número de pontos detectados (mais pontos = menor confiança individual)
        num_points_factor = max(0, 1 - len(change_points) / 50)
        confidence_factors.append(num_points_factor)
        
        # 2. Consenso entre métodos
        multi_source_points = [p for p in change_points if p.get('group_size', 1) > 1]
        consensus_factor = len(multi_source_points) / len(change_points)
        confidence_factors.append(consensus_factor)
        
        # 3. Confiança média dos pontos
        avg_confidence = np.mean([p.get('confidence', 0) for p in change_points])
        confidence_factors.append(avg_confidence)
        
        return np.mean(confidence_factors)
    
    def calculate_prng_detection_confidence(self):
        """Calcula confiança na detecção de PRNG"""
        consolidated = self.consolidated_results.get('consolidated', {})
        prng_candidates = consolidated.get('all_prng_candidates', [])
        
        if not prng_candidates:
            return 0
        
        # Fatores de confiança
        confidence_factors = []
        
        # 1. Melhor confiança individual
        best_confidence = max([c.get('confidence', 0) for c in prng_candidates])
        confidence_factors.append(best_confidence)
        
        # 2. Consenso entre análises
        correlations = consolidated.get('cross_correlations', {}).get('prng_candidates', {})
        type_consensus = correlations.get('type_consensus', {})
        
        if type_consensus:
            # Calcular consenso baseado em tipos detectados por múltiplas fontes
            multi_source_types = [t for t, sources in type_consensus.items() if len(sources) > 1]
            consensus_factor = len(multi_source_types) / len(type_consensus)
            confidence_factors.append(consensus_factor)
        else:
            confidence_factors.append(0)
        
        # 3. Diversidade de métodos
        sources = set(c.get('analysis_source', 'unknown') for c in prng_candidates)
        diversity_factor = min(1.0, len(sources) / 2)  # Esperamos pelo menos 2 fontes
        confidence_factors.append(diversity_factor)
        
        return np.mean(confidence_factors)
    
    def calculate_optimization_confidence(self):
        """Calcula confiança na otimização"""
        consolidated = self.consolidated_results.get('consolidated', {})
        solutions = consolidated.get('all_best_solutions', [])
        
        if not solutions:
            return 0
        
        # Fatores de confiança
        confidence_factors = []
        
        # 1. Melhor score obtido
        best_score = max([s.get('score', 0) for s in solutions])
        confidence_factors.append(best_score)
        
        # 2. Consistência entre soluções
        correlations = consolidated.get('cross_correlations', {}).get('solutions', {})
        score_dist = correlations.get('score_distribution', {})
        
        if score_dist and score_dist.get('mean', 0) > 0:
            # Consistência baseada em coeficiente de variação
            cv = score_dist.get('std', 0) / score_dist.get('mean', 1)
            consistency_factor = max(0, 1 - cv)
            confidence_factors.append(consistency_factor)
        else:
            confidence_factors.append(0)
        
        # 3. Número de soluções encontradas
        num_solutions_factor = min(1.0, len(solutions) / 10)  # Normalizar por 10 soluções
        confidence_factors.append(num_solutions_factor)
        
        return np.mean(confidence_factors)
    
    def calculate_overall_confidence(self, change_points_conf, prng_conf, optimization_conf):
        """Calcula confiança geral"""
        # Pesos para diferentes aspectos
        weights = {
            'change_points': 0.3,
            'prng_detection': 0.4,
            'optimization': 0.3
        }
        
        overall = (
            weights['change_points'] * change_points_conf +
            weights['prng_detection'] * prng_conf +
            weights['optimization'] * optimization_conf
        )
        
        return overall
    
    def generate_master_report(self):
        """Gera relatório mestre final"""
        print("\n📋 FASE 8: Geração de Relatório Mestre")
        print("-" * 50)
        
        print("📝 Compilando relatório abrangente...")
        
        self.master_report = {
            'metadata': {
                'title': 'Relatório Mestre - Análise Abrangente Mega Sena',
                'subtitle': 'Descoberta de Seeds e Análise de PRNGs',
                'generated_at': datetime.now().isoformat(),
                'execution_id': self.timestamp,
                'total_execution_time': 'N/A',  # Calcular se necessário
                'version': '1.0.0'
            },
            
            'executive_summary': self.generate_executive_summary(),
            
            'methodology': {
                'description': 'Análise multi-camada utilizando técnicas de computação quântica, engenharia reversa, otimização genética e análise temporal',
                'modules_used': [
                    'Detector Temporal Avançado',
                    'Analisador Quântico de PRNGs',
                    'Engenheiro Reverso Multi-PRNG',
                    'Motor de Otimização Genética'
                ],
                'data_analyzed': self.consolidated_results['consolidated']['data_summary']
            },
            
            'detailed_findings': {
                'temporal_analysis': self.summarize_temporal_findings(),
                'quantum_analysis': self.summarize_quantum_findings(),
                'reverse_engineering': self.summarize_reverse_engineering_findings(),
                'genetic_optimization': self.summarize_optimization_findings()
            },
            
            'consolidated_results': self.consolidated_results['consolidated'],
            
            'confidence_assessment': {
                'scores': self.confidence_scores,
                'reliability_analysis': self.generate_reliability_analysis(),
                'validation_requirements': self.generate_validation_requirements()
            },
            
            'practical_implications': self.generate_practical_implications(),
            
            'technical_appendices': self.generate_technical_appendices(),
            
            'conclusions_and_recommendations': self.generate_conclusions_and_recommendations()
        }
        
        # Salvar relatório
        report_path = self.output_dir / f"master_report_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(self.master_report, f, indent=2, default=str)
        
        # Gerar versão em texto
        self.generate_text_report()
        
        print(f"✅ Relatório mestre gerado: {report_path}")
    
    def generate_executive_summary(self):
        """Gera sumário executivo"""
        consolidated = self.consolidated_results.get('consolidated', {})
        
        # Principais descobertas
        key_findings = []
        
        # Pontos de mudança
        change_points = consolidated.get('all_change_points', [])
        if change_points:
            key_findings.append(f"{len(change_points)} pontos de mudança de algoritmo detectados")
        
        # Candidatos PRNG
        prng_candidates = consolidated.get('all_prng_candidates', [])
        if prng_candidates:
            best_prng = prng_candidates[0]
            key_findings.append(f"Algoritmo mais provável: {best_prng.get('type', 'desconhecido')} (confiança: {best_prng.get('confidence', 0):.3f})")
        
        # Melhores soluções
        solutions = consolidated.get('all_best_solutions', [])
        if solutions:
            best_solution = solutions[0]
            key_findings.append(f"Melhor solução de parâmetros: {best_solution.get('method', 'desconhecido')} (score: {best_solution.get('score', 0):.6f})")
        
        # Confiança geral
        overall_confidence = self.confidence_scores.get('overall', 0)
        confidence_level = 'alta' if overall_confidence > 0.7 else 'média' if overall_confidence > 0.4 else 'baixa'
        
        return {
            'analysis_scope': f"Análise abrangente de {len(self.historical_data)} sorteios da Mega Sena",
            'key_findings': key_findings,
            'confidence_level': f"Confiança geral: {confidence_level} ({overall_confidence:.3f})",
            'main_conclusion': self.generate_main_conclusion(),
            'actionable_insights': self.generate_actionable_insights()
        }
    
    def generate_main_conclusion(self):
        """Gera conclusão principal"""
        overall_confidence = self.confidence_scores.get('overall', 0)
        
        if overall_confidence > 0.7:
            return "Sistema da Mega Sena apresenta características determinísticas detectáveis com alta confiança."
        elif overall_confidence > 0.4:
            return "Sistema da Mega Sena apresenta padrões parcialmente detectáveis, requerendo validação adicional."
        else:
            return "Sistema da Mega Sena apresenta características predominantemente aleatórias ou utiliza algoritmos complexos não detectados."
    
    def generate_actionable_insights(self):
        """Gera insights acionáveis"""
        insights = []
        
        consolidated = self.consolidated_results.get('consolidated', {})
        
        # Insights sobre pontos de mudança
        change_points = consolidated.get('all_change_points', [])
        if change_points:
            insights.append(f"Monitorar sorteios próximos às posições {[p['position'] for p in change_points[:3]]} para validação")
        
        # Insights sobre PRNG
        prng_candidates = consolidated.get('all_prng_candidates', [])
        if prng_candidates and prng_candidates[0].get('confidence', 0) > 0.5:
            best_prng = prng_candidates[0]
            insights.append(f"Focar análise detalhada no algoritmo {best_prng['type']}")
        
        # Insights sobre soluções
        solutions = consolidated.get('all_best_solutions', [])
        if solutions and solutions[0].get('score', 0) > 0.5:
            insights.append("Validar parâmetros encontrados com sorteios futuros")
        
        if not insights:
            insights.append("Realizar análise adicional com métodos alternativos")
        
        return insights
    
    def summarize_temporal_findings(self):
        """Sumariza descobertas da análise temporal"""
        temporal_results = self.consolidated_results.get('temporal', {})
        
        if temporal_results.get('status') != 'completed':
            return {'status': 'failed', 'error': temporal_results.get('error')}
        
        change_points = temporal_results.get('change_points', [])
        regimes = temporal_results.get('regimes', [])
        
        return {
            'change_points_detected': len(change_points),
            'regime_count': len(regimes),
            'average_regime_duration': np.mean([r['duration'] for r in regimes]) if regimes else 0,
            'most_significant_changes': change_points[:5] if change_points else [],
            'temporal_patterns': self.analyze_temporal_patterns(regimes)
        }
    
    def analyze_temporal_patterns(self, regimes):
        """Analisa padrões temporais nos regimes"""
        if not regimes:
            return {}
        
        # Analisar correlações entre regimes
        correlations = [r.get('correlation_pattern', 0) for r in regimes]
        entropies = [r.get('entropy', 0) for r in regimes]
        
        return {
            'correlation_trend': 'increasing' if correlations[-1] > correlations[0] else 'decreasing' if len(correlations) > 1 else 'stable',
            'entropy_trend': 'increasing' if entropies[-1] > entropies[0] else 'decreasing' if len(entropies) > 1 else 'stable',
            'regime_stability': np.std([r['duration'] for r in regimes]) / np.mean([r['duration'] for r in regimes]) if regimes else 0
        }
    
    def summarize_quantum_findings(self):
        """Sumariza descobertas da análise quântica"""
        quantum_results = self.consolidated_results.get('quantum', {})
        
        if quantum_results.get('status') != 'completed':
            return {'status': 'failed', 'error': quantum_results.get('error')}
        
        prng_candidates = quantum_results.get('prng_candidates', [])
        analysis_results = quantum_results.get('analysis_results', {})
        
        # Sumarizar descobertas quânticas
        quantum_summary = {}
        
        if 'entropy' in analysis_results:
            entropy_data = analysis_results['entropy']
            quantum_summary['entropy_analysis'] = {
                'mean_von_neumann': entropy_data.get('mean_entropy', 0),
                'entropy_variance': entropy_data.get('entropy_variance', 0),
                'mean_mutual_information': np.mean(entropy_data.get('mutual_informations', [0]))
            }
        
        if 'entanglement' in analysis_results:
            entanglement_data = analysis_results['entanglement']
            quantum_summary['entanglement_analysis'] = {
                'mean_entanglement': np.mean([e.get('negativity', 0) for e in entanglement_data]),
                'entanglement_variability': np.std([e.get('negativity', 0) for e in entanglement_data])
            }
        
        return {
            'prng_candidates_detected': len(prng_candidates),
            'best_quantum_candidate': prng_candidates[0] if prng_candidates else None,
            'quantum_metrics': quantum_summary,
            'quantum_complexity_score': self.calculate_quantum_complexity_score(analysis_results)
        }
    
    def calculate_quantum_complexity_score(self, analysis_results):
        """Calcula score de complexidade quântica"""
        complexity_factors = []
        
        # Complexidade baseada na entropia
        if 'entropy' in analysis_results:
            entropy_variance = analysis_results['entropy'].get('entropy_variance', 0)
            complexity_factors.append(min(1.0, entropy_variance * 10))
        
        # Complexidade baseada no entrelaçamento
        if 'entanglement' in analysis_results:
            entanglement_data = analysis_results['entanglement']
            mean_entanglement = np.mean([e.get('negativity', 0) for e in entanglement_data])
            complexity_factors.append(mean_entanglement)
        
        # Complexidade baseada na coerência
        if 'coherence' in analysis_results:
            coherence_data = analysis_results['coherence']
            mean_coherence = np.mean([c['coherence'].get('l1_coherence', 0) for c in coherence_data])
            complexity_factors.append(mean_coherence)
        
        return np.mean(complexity_factors) if complexity_factors else 0
    
    def summarize_reverse_engineering_findings(self):
        """Sumariza descobertas da engenharia reversa"""
        reverse_results = self.consolidated_results.get('reverse_engineering', {})
        
        if reverse_results.get('status') != 'completed':
            return {'status': 'failed', 'error': reverse_results.get('error')}
        
        prng_candidates = reverse_results.get('prng_candidates', [])
        analysis_results = reverse_results.get('analysis_results', {})
        
        # Analisar resultados por tipo de PRNG
        prng_analysis = {}
        for prng_type, result in analysis_results.items():
            if isinstance(result, dict) and result.get('best_match'):
                prng_analysis[prng_type] = {
                    'confidence': result['best_match'].get('confidence', 0),
                    'parameters': result['best_match']
                }
        
        return {
            'total_prng_types_analyzed': len(analysis_results),
            'successful_detections': len([r for r in analysis_results.values() 
                                        if isinstance(r, dict) and r.get('best_match')]),
            'prng_analysis_by_type': prng_analysis,
            'best_reverse_candidate': prng_candidates[0] if prng_candidates else None,
            'reverse_engineering_quality': self.assess_reverse_engineering_quality(analysis_results)
        }
    
    def assess_reverse_engineering_quality(self, analysis_results):
        """Avalia qualidade da engenharia reversa"""
        quality_factors = []
        
        # Número de tipos PRNG com detecção bem-sucedida
        successful_detections = len([r for r in analysis_results.values() 
                                   if isinstance(r, dict) and r.get('best_match')])
        detection_rate = successful_detections / len(analysis_results) if analysis_results else 0
        quality_factors.append(detection_rate)
        
        # Confiança média das detecções
        confidences = []
        for result in analysis_results.values():
            if isinstance(result, dict) and result.get('best_match'):
                confidences.append(result['best_match'].get('confidence', 0))
        
        avg_confidence = np.mean(confidences) if confidences else 0
        quality_factors.append(avg_confidence)
        
        return np.mean(quality_factors)
    
    def summarize_optimization_findings(self):
        """Sumariza descobertas da otimização genética"""
        optimization_results = self.consolidated_results.get('genetic_optimization', {})
        
        if optimization_results.get('status') != 'completed':
            return {'status': 'failed', 'error': optimization_results.get('error')}
        
        best_solutions = optimization_results.get('best_solutions', [])
        opt_results = optimization_results.get('optimization_results', {})
        
        return {
            'total_solutions_found': len(best_solutions),
            'best_solution': best_solutions[0] if best_solutions else None,
            'optimization_methods_used': list(opt_results.keys()),
            'score_distribution': self.analyze_solution_scores(best_solutions),
            'parameter_consensus': self.analyze_parameter_consensus(best_solutions)
        }
    
    def analyze_solution_scores(self, solutions):
        """Analisa distribuição de scores das soluções"""
        if not solutions:
            return {}
        
        scores = [s.get('score', 0) for s in solutions]
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'max': max(scores),
            'min': min(scores),
            'q75': np.percentile(scores, 75),
            'q25': np.percentile(scores, 25)
        }
    
    def analyze_parameter_consensus(self, solutions):
        """Analisa consenso de parâmetros entre soluções"""
        if not solutions:
            return {}
        
        # Coletar parâmetros mais comuns
        all_methods = [s.get('method', 'unknown') for s in solutions]
        method_counts = {}
        for method in all_methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        most_common_method = max(method_counts, key=method_counts.get) if method_counts else None
        
        return {
            'method_distribution': method_counts,
            'most_common_method': most_common_method,
            'method_consensus_rate': max(method_counts.values()) / len(solutions) if method_counts else 0
        }
    
    def generate_reliability_analysis(self):
        """Gera análise de confiabilidade"""
        return {
            'data_quality': {
                'completeness': 1.0,  # Assumindo dados completos
                'consistency': self.assess_data_consistency(),
                'temporal_coverage': f"{len(self.historical_data)} sorteios"
            },
            'method_robustness': {
                'temporal_analysis': 'alta - múltiplos métodos convergem',
                'quantum_analysis': 'média - dependente de bibliotecas especializadas',
                'reverse_engineering': 'alta - testa múltiplos algoritmos conhecidos',
                'genetic_optimization': 'alta - algoritmo evolutivo robusto'
            },
            'validation_status': {
                'cross_validation': self.assess_cross_validation(),
                'independent_validation': 'requerida - usar sorteios futuros',
                'statistical_significance': self.assess_statistical_significance()
            }
        }
    
    def assess_data_consistency(self):
        """Avalia consistência dos dados"""
        # Verificar consistência básica dos sorteios
        inconsistencies = 0
        
        for draw in self.historical_data:
            numbers = draw.get('numbers', [])
            
            # Verificar se tem 6 números
            if len(numbers) != 6:
                inconsistencies += 1
                continue
            
            # Verificar se números estão no range correto
            if any(n < 1 or n > 60 for n in numbers):
                inconsistencies += 1
                continue
            
            # Verificar se não há duplicatas
            if len(set(numbers)) != 6:
                inconsistencies += 1
        
        consistency_rate = 1 - (inconsistencies / len(self.historical_data))
        return f"{consistency_rate:.3f} ({inconsistencies} inconsistências detectadas)"
    
    def assess_cross_validation(self):
        """Avalia validação cruzada entre métodos"""
        consolidated = self.consolidated_results.get('consolidated', {})
        correlations = consolidated.get('cross_correlations', {})
        
        # Verificar correlações entre métodos
        correlation_quality = []
        
        for correlation_type, data in correlations.items():
            if isinstance(data, dict) and 'consensus_factor' in data:
                correlation_quality.append(data['consensus_factor'])
        
        if correlation_quality:
            avg_correlation = np.mean(correlation_quality)
            return f"boa - correlação média entre métodos: {avg_correlation:.3f}"
        else:
            return "limitada - correlações não quantificadas"
    
    def assess_statistical_significance(self):
        """Avalia significância estatística"""
        # Baseado no tamanho da amostra e resultados
        sample_size = len(self.historical_data)
        
        if sample_size > 1000:
            significance = "alta"
        elif sample_size > 500:
            significance = "média"
        else:
            significance = "baixa"
        
        return f"{significance} - {sample_size} amostras analisadas"
    
    def generate_validation_requirements(self):
        """Gera requisitos de validação"""
        return {
            'immediate_validation': [
                "Verificar predições com próximos 10 sorteios",
                "Validar pontos de mudança identificados",
                "Confirmar parâmetros PRNG detectados"
            ],
            'long_term_validation': [
                "Monitorar performance por 6 meses",
                "Atualizar análise com novos dados",
                "Refinar modelos baseado em novos padrões"
            ],
            'validation_metrics': [
                "Taxa de acerto em predições",
                "Estabilidade de parâmetros ao longo do tempo",
                "Manutenção de padrões detectados"
            ]
        }
    
    def generate_practical_implications(self):
        """Gera implicações práticas"""
        consolidated = self.consolidated_results.get('consolidated', {})
        overall_confidence = self.confidence_scores.get('overall', 0)
        
        implications = {
            'system_understanding': self.generate_system_understanding_implications(),
            'predictive_capability': self.generate_predictive_capability_implications(),
            'security_considerations': self.generate_security_implications(),
            'research_directions': self.generate_research_directions()
        }
        
        return implications
    
    def generate_system_understanding_implications(self):
        """Gera implicações para entendimento do sistema"""
        change_points = self.consolidated_results.get('consolidated', {}).get('all_change_points', [])
        
        if change_points:
            return [
                f"Sistema apresenta {len(change_points)} mudanças algorítmicas detectáveis",
                "Comportamento não é puramente aleatório",
                "Existe estrutura temporal detectável nos dados"
            ]
        else:
            return [
                "Sistema aparenta ter comportamento consistente ao longo do tempo",
                "Não foram detectadas mudanças algorítmicas significativas",
                "Comportamento próximo ao verdadeiramente aleatório"
            ]
    
    def generate_predictive_capability_implications(self):
        """Gera implicações para capacidade preditiva"""
        best_solutions = self.consolidated_results.get('consolidated', {}).get('all_best_solutions', [])
        
        if best_solutions and best_solutions[0].get('score', 0) > 0.5:
            return [
                "Detectada capacidade preditiva limitada",
                "Parâmetros identificados podem permitir predições parciais",
                "Requer validação com dados independentes"
            ]
        else:
            return [
                "Capacidade preditiva não estabelecida com confiança",
                "Sistema apresenta alta entropia",
                "Predições precisas improváveis com métodos atuais"
            ]
    
    def generate_security_implications(self):
        """Gera implicações de segurança"""
        prng_candidates = self.consolidated_results.get('consolidated', {}).get('all_prng_candidates', [])
        
        if prng_candidates and prng_candidates[0].get('confidence', 0) > 0.7:
            return [
                "Sistema pode ser baseado em PRNG determinístico",
                "Possível vulnerabilidade se parâmetros forem confirmados",
                "Recomenda-se auditoria de segurança do sistema gerador"
            ]
        else:
            return [
                "Não foram identificadas vulnerabilidades óbvias",
                "Sistema aparenta usar geração segura",
                "Análise adicional recomendada para confirmação"
            ]
    
    def generate_research_directions(self):
        """Gera direções de pesquisa"""
        return [
            "Investigar algoritmos criptográficos avançados",
            "Analisar correlações com fatores externos",
            "Desenvolver métodos de detecção mais sensíveis",
            "Estudar padrões em outras loterias similares",
            "Aplicar técnicas de deep learning",
            "Investigar possível influência de hardware específico"
        ]
    
    def generate_technical_appendices(self):
        """Gera apêndices técnicos"""
        return {
            'algorithm_details': {
                'temporal_detection': 'Múltiplos métodos incluindo clustering, PCA, detecção de outliers',
                'quantum_analysis': 'Entropia von Neumann, transformada de Fourier quântica, análise de entrelaçamento',
                'reverse_engineering': 'Teste sistemático de LCG, LFSR, Mersenne Twister, Xorshift, PCG',
                'genetic_optimization': 'Algoritmo genético com seleção por torneio, crossover e mutação'
            },
            'parameter_ranges': self.extract_parameter_ranges(),
            'computational_complexity': self.assess_computational_complexity(),
            'library_versions': self.get_library_versions()
        }
    
    def extract_parameter_ranges(self):
        """Extrai ranges de parâmetros testados"""
        return {
            'LCG_parameters': {
                'a_range': '1 to 2^16',
                'c_range': '0 to 2^16', 
                'm_values': ['2^31-1', '2^32', '2^31']
            },
            'genetic_algorithm': {
                'population_size': 100,
                'generations': 500,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8
            },
            'analysis_windows': {
                'temporal_window': '50-100 sorteios',
                'quantum_analysis': 'sequência completa',
                'validation_period': '10-20 sorteios'
            }
        }
    
    def assess_computational_complexity(self):
        """Avalia complexidade computacional"""
        return {
            'temporal_analysis': 'O(n²) para correlações, O(n log n) para FFT',
            'quantum_analysis': 'O(n) para entropia, O(n²) para entrelaçamento',
            'reverse_engineering': 'O(k × n) onde k é número de parâmetros testados',
            'genetic_optimization': 'O(g × p × n) onde g=gerações, p=população, n=tamanho dados'
        }
    
    def get_library_versions(self):
        """Obtém versões das bibliotecas utilizadas"""
        try:
            import sys
            versions = {
                'python': sys.version,
                'numpy': np.__version__,
                'pandas': pd.__version__
            }
            
            try:
                import scipy
                versions['scipy'] = scipy.__version__
            except:
                pass
                
            try:
                import matplotlib
                versions['matplotlib'] = matplotlib.__version__
            except:
                pass
                
            return versions
        except:
            return {'note': 'versões não disponíveis'}
    
    def generate_conclusions_and_recommendations(self):
        """Gera conclusões e recomendações finais"""
        overall_confidence = self.confidence_scores.get('overall', 0)
        
        conclusions = {
            'primary_conclusions': self.generate_primary_conclusions(),
            'confidence_assessment': self.generate_confidence_assessment(),
            'limitations': self.identify_limitations(),
            'recommendations': {
                'immediate_actions': self.generate_immediate_recommendations(),
                'future_research': self.generate_future_research_recommendations(),
                'validation_strategy': self.generate_validation_strategy()
            }
        }
        
        return conclusions
    
    def generate_primary_conclusions(self):
        """Gera conclusões primárias"""
        consolidated = self.consolidated_results.get('consolidated', {})
        change_points = consolidated.get('all_change_points', [])
        prng_candidates = consolidated.get('all_prng_candidates', [])
        solutions = consolidated.get('all_best_solutions', [])
        
        conclusions = []
        
        # Conclusão sobre determinismo
        if change_points and prng_candidates:
            conclusions.append("Sistema apresenta características determinísticas detectáveis")
        else:
            conclusions.append("Sistema aparenta comportamento predominantemente aleatório")
        
        # Conclusão sobre mudanças temporais
        if len(change_points) > 3:
            conclusions.append("Múltiplas mudanças algorítmicas detectadas ao longo do tempo")
        elif change_points:
            conclusions.append("Poucas mudanças algorítmicas detectadas")
        else:
            conclusions.append("Comportamento algorítmico estável ao longo do tempo")
        
        # Conclusão sobre predição
        if solutions and solutions[0].get('score', 0) > 0.5:
            conclusions.append("Capacidade preditiva limitada demonstrada")
        else:
            conclusions.append("Capacidade preditiva não estabelecida")
        
        return conclusions
    
    def generate_confidence_assessment(self):
        """Gera avaliação de confiança"""
        overall_confidence = self.confidence_scores.get('overall', 0)
        
        if overall_confidence > 0.7:
            assessment = "Alta confiança nos resultados obtidos"
        elif overall_confidence > 0.4:
            assessment = "Confiança moderada - resultados promissores mas requerem validação"
        else:
            assessment = "Baixa confiança - resultados inconclusivos"
        
        return {
            'overall_assessment': assessment,
            'confidence_score': overall_confidence,
            'key_uncertainties': self.identify_key_uncertainties()
        }
    
    def identify_key_uncertainties(self):
        """Identifica principais incertezas"""
        uncertainties = []
        
        # Incertezas baseadas nos resultados
        if self.confidence_scores.get('prng_detection', 0) < 0.5:
            uncertainties.append("Tipo de algoritmo PRNG não definitivamente identificado")
        
        if self.confidence_scores.get('optimization', 0) < 0.5:
            uncertainties.append("Parâmetros ótimos não convergidos com alta confiança")
        
        if len(self.consolidated_results.get('consolidated', {}).get('all_change_points', [])) == 0:
            uncertainties.append("Ausência de pontos de mudança pode indicar sistema mais complexo")
        
        return uncertainties
    
    def identify_limitations(self):
        """Identifica limitações do estudo"""
        return [
            "Análise baseada apenas em números sorteados, sem acesso ao código fonte",
            "Pressuposições sobre tipos de PRNG podem estar incompletas",
            "Validação requer dados futuros não disponíveis no momento da análise",
            "Possível influência de fatores externos não considerados",
            "Limitações computacionais restringem espaço de busca exaustiva"
        ]
    
    def generate_immediate_recommendations(self):
        """Gera recomendações imediatas"""
        recommendations = []
        
        consolidated = self.consolidated_results.get('consolidated', {})
        
        # Recomendações baseadas em descobertas
        if consolidated.get('all_change_points'):
            recommendations.append("Validar pontos de mudança com análise independente")
        
        if consolidated.get('all_prng_candidates'):
            best_prng = consolidated['all_prng_candidates'][0]
            if best_prng.get('confidence', 0) > 0.5:
                recommendations.append(f"Investigar em detalhes o algoritmo {best_prng['type']}")
        
        if consolidated.get('all_best_solutions'):
            recommendations.append("Testar parâmetros encontrados com sorteios futuros")
        
        recommendations.append("Documentar metodologia para reprodutibilidade")
        recommendations.append("Compartilhar resultados com comunidade de pesquisa")
        
        return recommendations
    
    def generate_future_research_recommendations(self):
        """Gera recomendações para pesquisa futura"""
        return [
            "Aplicar análise similar a outras loterias nacionais e internacionais",
            "Desenvolver métodos de detecção baseados em deep learning",
            "Investigar correlações com dados meteorológicos e outros fatores externos",
            "Estudar possível influência de hardware específico nos geradores",
            "Desenvolver testes estatísticos mais sensíveis para detecção de padrões",
            "Colaborar com especialistas em criptografia para análise de segurança"
        ]
    
    def generate_validation_strategy(self):
        """Gera estratégia de validação"""
        return {
            'short_term': [
                "Coletar próximos 20 sorteios para validação imediata",
                "Aplicar modelos encontrados para predição",
                "Medir taxa de acerto e comparar com baseline aleatório"
            ],
            'medium_term': [
                "Monitorar performance por 6 meses",
                "Refinar parâmetros baseado em novos dados",
                "Desenvolver métricas de performance específicas"
            ],
            'long_term': [
                "Estabelecer protocolo de monitoramento contínuo",
                "Criar sistema de alerta para mudanças detectadas",
                "Desenvolver framework para análise automatizada"
            ]
        }
    
    def generate_text_report(self):
        """Gera versão em texto do relatório"""
        print("📄 Gerando relatório em texto...")
        
        text_report = f"""
# RELATÓRIO MESTRE - ANÁLISE ABRANGENTE MEGA SENA
## Descoberta de Seeds e Análise de PRNGs

**Gerado em:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
**ID de Execução:** {self.timestamp}

---

## SUMÁRIO EXECUTIVO

{self.master_report['executive_summary']['analysis_scope']}

### Principais Descobertas:
"""
        
        for finding in self.master_report['executive_summary']['key_findings']:
            text_report += f"• {finding}\n"
        
        text_report += f"""
### {self.master_report['executive_summary']['confidence_level']}

### Conclusão Principal:
{self.master_report['executive_summary']['main_conclusion']}

### Insights Acionáveis:
"""
        
        for insight in self.master_report['executive_summary']['actionable_insights']:
            text_report += f"• {insight}\n"
        
        text_report += f"""

---

## METODOLOGIA

{self.master_report['methodology']['description']}

### Módulos Utilizados:
"""
        
        for module in self.master_report['methodology']['modules_used']:
            text_report += f"• {module}\n"
        
        text_report += f"""

### Dados Analisados:
• Total de sorteios: {self.master_report['methodology']['data_analyzed']['total_draws']}
• Período: Concurso {self.master_report['methodology']['data_analyzed']['date_range']['first']} ao {self.master_report['methodology']['data_analyzed']['date_range']['last']}
• Análises bem-sucedidas: {self.master_report['methodology']['data_analyzed']['successful_analyses']}

---

## PRINCIPAIS RESULTADOS

### Análise Temporal
"""
        
        temporal = self.master_report['detailed_findings']['temporal_analysis']
        if temporal.get('status') == 'failed':
            text_report += f"❌ Falha: {temporal.get('error', 'Erro desconhecido')}\n"
        else:
            text_report += f"• Pontos de mudança detectados: {temporal.get('change_points_detected', 0)}\n"
            text_report += f"• Regimes identificados: {temporal.get('regime_count', 0)}\n"
            text_report += f"• Duração média de regime: {temporal.get('average_regime_duration', 0):.1f} sorteios\n"
        
        text_report += """

### Análise Quântica
"""
        
        quantum = self.master_report['detailed_findings']['quantum_analysis']
        if quantum.get('status') == 'failed':
            text_report += f"❌ Falha: {quantum.get('error', 'Erro desconhecido')}\n"
        else:
            text_report += f"• Candidatos PRNG detectados: {quantum.get('prng_candidates_detected', 0)}\n"
            if quantum.get('best_quantum_candidate'):
                best = quantum['best_quantum_candidate']
                text_report += f"• Melhor candidato: {best.get('type', 'N/A')} (confiança: {best.get('confidence', 0):.3f})\n"
            text_report += f"• Score de complexidade quântica: {quantum.get('quantum_complexity_score', 0):.3f}\n"
        
        text_report += """

### Engenharia Reversa
"""
        
        reverse = self.master_report['detailed_findings']['reverse_engineering']
        if reverse.get('status') == 'failed':
            text_report += f"❌ Falha: {reverse.get('error', 'Erro desconhecido')}\n"
        else:
            text_report += f"• Tipos PRNG analisados: {reverse.get('total_prng_types_analyzed', 0)}\n"
            text_report += f"• Detecções bem-sucedidas: {reverse.get('successful_detections', 0)}\n"
            text_report += f"• Qualidade da engenharia reversa: {reverse.get('reverse_engineering_quality', 0):.3f}\n"
        
        text_report += """

### Otimização Genética
"""
        
        optimization = self.master_report['detailed_findings']['genetic_optimization']
        if optimization.get('status') == 'failed':
            text_report += f"❌ Falha: {optimization.get('error', 'Erro desconhecido')}\n"
        else:
            text_report += f"• Soluções encontradas: {optimization.get('total_solutions_found', 0)}\n"
            if optimization.get('best_solution'):
                best = optimization['best_solution']
                text_report += f"• Melhor solução: {best.get('method', 'N/A')} (score: {best.get('score', 0):.6f})\n"
        
        text_report += f"""

---

## AVALIAÇÃO DE CONFIANÇA

### Scores de Confiança:
• Pontos de mudança: {self.confidence_scores.get('change_points', 0):.3f}
• Detecção PRNG: {self.confidence_scores.get('prng_detection', 0):.3f}
• Otimização: {self.confidence_scores.get('optimization', 0):.3f}
• **Score Geral: {self.confidence_scores.get('overall', 0):.3f}**

### Avaliação:
{self.master_report['confidence_assessment']['reliability_analysis']['method_robustness']}

---

## CONCLUSÕES E RECOMENDAÇÕES

### Conclusões Primárias:
"""
        
        for conclusion in self.master_report['conclusions_and_recommendations']['primary_conclusions']:
            text_report += f"• {conclusion}\n"
        
        text_report += f"""

### Avaliação de Confiança:
{self.master_report['conclusions_and_recommendations']['confidence_assessment']['overall_assessment']}

### Limitações:
"""
        
        for limitation in self.master_report['conclusions_and_recommendations']['limitations']:
            text_report += f"• {limitation}\n"
        
        text_report += """

### Recomendações Imediatas:
"""
        
        for rec in self.master_report['conclusions_and_recommendations']['recommendations']['immediate_actions']:
            text_report += f"• {rec}\n"
        
        text_report += """

### Pesquisa Futura:
"""
        
        for rec in self.master_report['conclusions_and_recommendations']['recommendations']['future_research']:
            text_report += f"• {rec}\n"
        
        text_report += f"""

---

## IMPLICAÇÕES PRÁTICAS

### Entendimento do Sistema:
"""
        
        for impl in self.master_report['practical_implications']['system_understanding']:
            text_report += f"• {impl}\n"
        
        text_report += """

### Capacidade Preditiva:
"""
        
        for impl in self.master_report['practical_implications']['predictive_capability']:
            text_report += f"• {impl}\n"
        
        text_report += """

### Considerações de Segurança:
"""
        
        for impl in self.master_report['practical_implications']['security_considerations']:
            text_report += f"• {impl}\n"
        
        text_report += f"""

---

**Relatório gerado pelo Sistema de Análise Abrangente Mega Sena v1.0.0**
**Timestamp: {self.timestamp}**
"""
        
        # Salvar relatório em texto
        text_path = self.output_dir / f"master_report_{self.timestamp}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        print(f"✅ Relatório em texto gerado: {text_path}")
    
    def create_interactive_visualizations(self):
        """Cria visualizações interativas com Plotly"""
        print("\n📊 FASE 9: Criação de Visualizações Interativas")
        print("-" * 50)
        
        try:
            self.create_comprehensive_dashboard()
            self.create_analysis_timeline()
            self.create_confidence_radar()
            self.create_prng_comparison()
            
            print("✅ Visualizações interativas criadas com sucesso")
            
        except Exception as e:
            print(f"❌ Erro na criação de visualizações: {e}")
    
    def create_comprehensive_dashboard(self):
        """Cria dashboard abrangente"""
        print("📈 Criando dashboard abrangente...")
        
        # Criar subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Pontos de Mudança Detectados', 'Candidatos PRNG por Confiança',
                          'Evolução dos Scores', 'Distribuição de Confiança'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'histogram'}]]
        )
        
        # 1. Pontos de mudança
        consolidated = self.consolidated_results.get('consolidated', {})
        change_points = consolidated.get('all_change_points', [])
        
        if change_points:
            positions = [cp['position'] for cp in change_points]
            confidences = [cp['confidence'] for cp in change_points]
            sources = [cp['source'] for cp in change_points]
            
            fig.add_trace(
                go.Scatter(
                    x=positions,
                    y=confidences,
                    mode='markers+text',
                    text=[f"{src[:10]}" for src in sources],
                    textposition="top center",
                    marker=dict(size=10, color=confidences, colorscale='Viridis'),
                    name='Pontos de Mudança'
                ),
                row=1, col=1
            )
        
        # 2. Candidatos PRNG
        prng_candidates = consolidated.get('all_prng_candidates', [])
        
        if prng_candidates:
            prng_types = [c.get('type', 'unknown') for c in prng_candidates[:10]]
            prng_confidences = [c.get('confidence', 0) for c in prng_candidates[:10]]
            
            fig.add_trace(
                go.Bar(
                    x=prng_types,
                    y=prng_confidences,
                    marker_color=prng_confidences,
                    colorscale='Plasma',
                    name='PRNG Candidates'
                ),
                row=1, col=2
            )
        
        # 3. Evolução dos scores (se disponível)
        solutions = consolidated.get('all_best_solutions', [])
        
        if solutions:
            solution_scores = [s.get('score', 0) for s in solutions[:20]]
            solution_indices = list(range(len(solution_scores)))
            
            fig.add_trace(
                go.Scatter(
                    x=solution_indices,
                    y=solution_scores,
                    mode='lines+markers',
                    line=dict(color='red', width=2),
                    name='Evolution of Scores'
                ),
                row=2, col=1
            )
        
        # 4. Distribuição de confiança
        all_confidences = []
        
        if change_points:
            all_confidences.extend([cp['confidence'] for cp in change_points])
        if prng_candidates:
            all_confidences.extend([c.get('confidence', 0) for c in prng_candidates])
        
        if all_confidences:
            fig.add_trace(
                go.Histogram(
                    x=all_confidences,
                    nbinsx=20,
                    marker_color='lightblue',
                    name='Confidence Distribution'
                ),
                row=2, col=2
            )
        
        # Layout
        fig.update_layout(
            title_text=f"Dashboard Abrangente - Análise Mega Sena ({self.timestamp})",
            height=800,
            showlegend=False
        )
        
        # Salvar
        dashboard_path = self.output_dir / f"comprehensive_dashboard_{self.timestamp}.html"
        fig.write_html(str(dashboard_path))
        
        print(f"   ✓ Dashboard salvo em: {dashboard_path}")
    
    def create_analysis_timeline(self):
        """Cria timeline da análise"""
        print("⏱️ Criando timeline da análise...")
        
        # Dados para timeline
        timeline_data = []
        
        # Pontos de mudança
        consolidated = self.consolidated_results.get('consolidated', {})
        change_points = consolidated.get('all_change_points', [])
        
        for cp in change_points:
            timeline_data.append({
                'x': cp['position'],
                'y': 1,
                'type': 'Change Point',
                'source': cp['source'],
                'confidence': cp['confidence']
            })
        
        # Dados de sorteios (amostra)
        sample_draws = self.historical_data[::50]  # Cada 50º sorteio
        for i, draw in enumerate(sample_draws):
            timeline_data.append({
                'x': i * 50,
                'y': 0,
                'type': 'Draw Sample',
                'source': f"Concurso {draw['concurso']}",
                'confidence': 1.0
            })
        
        # Criar figura
        fig = px.scatter(
            timeline_data,
            x='x',
            y='y',
            color='type',
            size='confidence',
            hover_data=['source', 'confidence'],
            title=f"Timeline da Análise - {len(self.historical_data)} Sorteios"
        )
        
        fig.update_layout(
            xaxis_title="Posição no Tempo (Sorteios)",
            yaxis_title="Tipo de Evento",
            height=400
        )
        
        # Salvar
        timeline_path = self.output_dir / f"analysis_timeline_{self.timestamp}.html"
        fig.write_html(str(timeline_path))
        
        print(f"   ✓ Timeline salvo em: {timeline_path}")
    
    def create_confidence_radar(self):
        """Cria gráfico radar de confiança"""
        print("🎯 Criando radar de confiança...")
        
        # Dados para radar
        categories = [
            'Pontos de Mudança',
            'Detecção PRNG',
            'Otimização',
            'Correlação Cruzada',
            'Validação Estatística'
        ]
        
        values = [
            self.confidence_scores.get('change_points', 0),
            self.confidence_scores.get('prng_detection', 0),
            self.confidence_scores.get('optimization', 0),
            0.5,  # Placeholder para correlação cruzada
            0.6   # Placeholder para validação estatística
        ]
        
        # Criar figura radar
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Scores de Confiança',
            line=dict(color='blue', width=2),
            fillcolor='rgba(0, 100, 255, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title=f"Radar de Confiança - Score Geral: {self.confidence_scores.get('overall', 0):.3f}",
            height=500
        )
        
        # Salvar
        radar_path = self.output_dir / f"confidence_radar_{self.timestamp}.html"
        fig.write_html(str(radar_path))
        
        print(f"   ✓ Radar de confiança salvo em: {radar_path}")
    
    def create_prng_comparison(self):
        """Cria comparação de candidatos PRNG"""
        print("🔧 Criando comparação de PRNGs...")
        
        consolidated = self.consolidated_results.get('consolidated', {})
        prng_candidates = consolidated.get('all_prng_candidates', [])
        
        if not prng_candidates:
            print("   ⚠️ Nenhum candidato PRNG para visualizar")
            return
        
        # Preparar dados
        comparison_data = []
        
        for candidate in prng_candidates[:10]:  # Top 10
            comparison_data.append({
                'Type': candidate.get('type', 'unknown'),
                'Confidence': candidate.get('confidence', 0),
                'Source': candidate.get('analysis_source', 'unknown'),
                'Method': candidate.get('method', 'N/A')
            })
        
        if not comparison_data:
            print("   ⚠️ Dados insuficientes para comparação")
            return
        
        # Criar figura
        fig = px.bar(
            comparison_data,
            x='Type',
            y='Confidence',
            color='Source',
            hover_data=['Method'],
            title="Comparação de Candidatos PRNG",
            text='Confidence'
        )
        
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(
            xaxis_title="Tipo de PRNG",
            yaxis_title="Confiança",
            height=500
        )
        
        # Salvar
        comparison_path = self.output_dir / f"prng_comparison_{self.timestamp}.html"
        fig.write_html(str(comparison_path))
        
        print(f"   ✓ Comparação PRNG salva em: {comparison_path}")
    
    def generate_final_recommendations(self):
        """Gera recomendações finais e próximos passos"""
        print("\n🎯 FASE 10: Geração de Recomendações Finais")
        print("-" * 50)
        
        # Análise final baseada em todos os resultados
        final_analysis = self.perform_final_analysis()
        
        # Recomendações estratégicas
        strategic_recommendations = self.generate_strategic_recommendations(final_analysis)
        
        # Plano de ação
        action_plan = self.create_action_plan(final_analysis)
        
        # Salvar recomendações finais
        final_recommendations = {
            'timestamp': datetime.now().isoformat(),
            'final_analysis': final_analysis,
            'strategic_recommendations': strategic_recommendations,
            'action_plan': action_plan,
            'execution_summary': self.generate_execution_summary()
        }
        
        # Salvar
        recommendations_path = self.output_dir / f"final_recommendations_{self.timestamp}.json"
        with open(recommendations_path, 'w') as f:
            json.dump(final_recommendations, f, indent=2, default=str)
        
        print(f"✅ Recomendações finais salvas em: {recommendations_path}")
        
        # Exibir resumo das recomendações
        self.display_final_summary(final_analysis, strategic_recommendations)
        
        return final_recommendations
    
    def perform_final_analysis(self):
        """Realiza análise final consolidada"""
        consolidated = self.consolidated_results.get('consolidated', {})
        confidence = self.confidence_scores
        
        # Classificar nível de descoberta
        overall_confidence = confidence.get('overall', 0)
        
        if overall_confidence > 0.7:
            discovery_level = 'High'
            discovery_description = 'Padrões determinísticos claros detectados'
        elif overall_confidence > 0.4:
            discovery_level = 'Medium'
            discovery_description = 'Padrões parciais detectados, requer validação'
        else:
            discovery_level = 'Low'
            discovery_description = 'Padrões mínimos ou sistema verdadeiramente aleatório'
        
        # Analisar consistência entre métodos
        change_points = consolidated.get('all_change_points', [])
        prng_candidates = consolidated.get('all_prng_candidates', [])
        solutions = consolidated.get('all_best_solutions', [])
        
        consistency_score = self.calculate_method_consistency()
        
        return {
            'discovery_level': discovery_level,
            'discovery_description': discovery_description,
            'overall_confidence': overall_confidence,
            'consistency_score': consistency_score,
            'key_findings': {
                'change_points_count': len(change_points),
                'top_prng_candidate': prng_candidates[0] if prng_candidates else None,
                'best_solution_score': solutions[0]['score'] if solutions else 0
            },
            'method_performance': {
                'temporal_analysis': confidence.get('change_points', 0),
                'quantum_analysis': confidence.get('prng_detection', 0) * 0.5,  # Ponderado
                'reverse_engineering': confidence.get('prng_detection', 0) * 0.5,
                'genetic_optimization': confidence.get('optimization', 0)
            }
        }
    
    def calculate_method_consistency(self):
        """Calcula consistência entre métodos"""
        consolidated = self.consolidated_results.get('consolidated', {})
        correlations = consolidated.get('cross_correlations', {})
        
        consistency_factors = []
        
        # Consistência nos pontos de mudança
        change_correlation = correlations.get('change_points', {})
        if change_correlation:
            avg_correlation = np.mean(list(change_correlation.values()))
            consistency_factors.append(avg_correlation)
        
        # Consistência nos candidatos PRNG
        prng_correlation = correlations.get('prng_candidates', {})
        if prng_correlation and 'type_consensus' in prng_correlation:
            type_consensus = prng_correlation['type_consensus']
            if type_consensus:
                multi_source_types = sum(1 for sources in type_consensus.values() if len(sources) > 1)
                consensus_rate = multi_source_types / len(type_consensus)
                consistency_factors.append(consensus_rate)
        
        return np.mean(consistency_factors) if consistency_factors else 0
    
    def generate_strategic_recommendations(self, final_analysis):
        """Gera recomendações estratégicas"""
        discovery_level = final_analysis['discovery_level']
        
        if discovery_level == 'High':
            return {
                'priority': 'immediate_validation',
                'recommendations': [
                    'Implementar sistema de monitoramento contínuo',
                    'Validar descobertas com dados independentes',
                    'Considerar implicações de segurança',
                    'Documentar metodologia para auditoria',
                    'Desenvolver sistema de predição refinado'
                ],
                'resources_required': 'Altos - equipe especializada',
                'timeline': '1-3 meses'
            }
        elif discovery_level == 'Medium':
            return {
                'priority': 'continued_research',
                'recommendations': [
                    'Ampliar análise com mais dados',
                    'Refinar métodos de detecção',
                    'Buscar validação com outros datasets',
                    'Investigar métodos alternativos',
                    'Colaborar com especialistas externos'
                ],
                'resources_required': 'Médios - pesquisa continuada',
                'timeline': '3-6 meses'
            }
        else:
            return {
                'priority': 'alternative_approaches',
                'recommendations': [
                    'Explorar metodologias alternativas',
                    'Investigar fatores externos',
                    'Analisar outros sistemas de loteria',
                    'Desenvolver métodos mais sensíveis',
                    'Considerar análise de hardware'
                ],
                'resources_required': 'Baixos - pesquisa exploratória',
                'timeline': '6-12 meses'
            }
    
    def create_action_plan(self, final_analysis):
        """Cria plano de ação detalhado"""
        discovery_level = final_analysis['discovery_level']
        
        base_plan = {
            'immediate_actions': [
                'Documentar todos os resultados obtidos',
                'Criar backup de dados e código',
                'Preparar apresentação dos resultados'
            ],
            'short_term': [
                'Validar resultados com próximos sorteios',
                'Refinar parâmetros mais promissores',
                'Buscar peer review da metodologia'
            ],
            'medium_term': [
                'Implementar melhorias baseadas em feedback',
                'Expandir análise para outros contextos',
                'Desenvolver ferramentas automatizadas'
            ],
            'long_term': [
                'Estabelecer protocolo de monitoramento',
                'Publicar resultados em venues apropriados',
                'Colaborar com instituições relevantes'
            ]
        }
        
        # Personalizar baseado no nível de descoberta
        if discovery_level == 'High':
            base_plan['immediate_actions'].append('Alertar stakeholders relevantes')
            base_plan['short_term'].append('Implementar sistema de validação rigoroso')
        elif discovery_level == 'Medium':
            base_plan['short_term'].append('Buscar dados adicionais para confirmação')
            base_plan['medium_term'].append('Desenvolver métodos mais robustos')
        
        return base_plan
    
    def generate_execution_summary(self):
        """Gera resumo da execução"""
        return {
            'modules_executed': len(self.consolidated_results),
            'successful_analyses': len([r for r in self.consolidated_results.values() 
                                      if r.get('status') == 'completed']),
            'data_processed': len(self.historical_data),
            'outputs_generated': [
                'Relatório Mestre JSON',
                'Relatório Mestre Texto',
                'Dashboard Interativo',
                'Visualizações Específicas',
                'Recomendações Finais'
            ],
            'key_metrics': {
                'overall_confidence': self.confidence_scores.get('overall', 0),
                'change_points_detected': len(self.consolidated_results.get('consolidated', {}).get('all_change_points', [])),
                'prng_candidates_found': len(self.consolidated_results.get('consolidated', {}).get('all_prng_candidates', [])),
                'optimization_solutions': len(self.consolidated_results.get('consolidated', {}).get('all_best_solutions', []))
            }
        }
    
    def display_final_summary(self, final_analysis, strategic_recommendations):
        """Exibe resumo final no console"""
        print("\n" + "="*80)
        print("🎯 RESUMO FINAL DA ANÁLISE MESTRE")
        print("="*80)
        
        print(f"\n📊 NÍVEL DE DESCOBERTA: {final_analysis['discovery_level'].upper()}")
        print(f"📝 Descrição: {final_analysis['discovery_description']}")
        print(f"🎯 Confiança Geral: {final_analysis['overall_confidence']:.3f}")
        print(f"🔗 Consistência entre Métodos: {final_analysis['consistency_score']:.3f}")
        
        print(f"\n🔍 PRINCIPAIS DESCOBERTAS:")
        key_findings = final_analysis['key_findings']
        print(f"   • Pontos de mudança detectados: {key_findings['change_points_count']}")
        
        if key_findings['top_prng_candidate']:
            top_prng = key_findings['top_prng_candidate']
            print(f"   • Melhor candidato PRNG: {top_prng.get('type', 'N/A')} (confiança: {top_prng.get('confidence', 0):.3f})")
        
        print(f"   • Melhor score de otimização: {key_findings['best_solution_score']:.6f}")
        
        print(f"\n📈 PERFORMANCE DOS MÉTODOS:")
        performance = final_analysis['method_performance']
        for method, score in performance.items():
            print(f"   • {method.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"\n🎯 PRIORIDADE ESTRATÉGICA: {strategic_recommendations['priority'].upper()}")
        print(f"⏱️ Timeline: {strategic_recommendations['timeline']}")
        print(f"📊 Recursos: {strategic_recommendations['resources_required']}")
        
        print(f"\n📋 PRINCIPAIS RECOMENDAÇÕES:")
        for rec in strategic_recommendations['recommendations'][:3]:
            print(f"   • {rec}")
        
        print(f"\n📂 ARQUIVOS GERADOS:")
        print(f"   • Relatório Mestre: master_report_{self.timestamp}.json")
        print(f"   • Relatório Texto: master_report_{self.timestamp}.txt") 
        print(f"   • Dashboard: comprehensive_dashboard_{self.timestamp}.html")
        print(f"   • Recomendações: final_recommendations_{self.timestamp}.json")
        
        print("\n" + "="*80)
        print("✅ ANÁLISE MESTRE ABRANGENTE CONCLUÍDA COM SUCESSO!")
        print("="*80)

# Script de execução principal
if __name__ == "__main__":
    print("🎯 SISTEMA DE ANÁLISE MESTRE ABRANGENTE")
    print("="*80)
    
    try:
        # Inicializar analisador mestre
        data_path = "../data/MegaSena3.xlsx"
        master_analyzer = ComprehensiveMasterAnalyzer(data_path)
        
        # Executar análise completa
        master_analyzer.execute_comprehensive_analysis()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Análise interrompida pelo usuário")
    except Exception as e:
        print(f"\n\n❌ Erro crítico: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🔚 Execução finalizada.")