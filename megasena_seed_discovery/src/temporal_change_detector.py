#!/usr/bin/env python3
"""
Detector Avançado de Mudanças Temporais
Implementa múltiplas técnicas para identificar pontos de mudança de seed na timeline
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import json
import warnings
warnings.filterwarnings('ignore')

class TemporalChangeDetector:
    def __init__(self, historical_data):
        self.historical_data = historical_data
        self.change_points = []
        self.analysis_results = {}
        self.features_matrix = None
        
    def extract_comprehensive_features(self):
        """Extrai features abrangentes para cada sorteio"""
        print("\n🔬 Extraindo features abrangentes...")
        
        features = []
        
        for i, draw_info in enumerate(self.historical_data):
            numbers = draw_info['numbers']
            
            # Features básicas
            feature_vector = {
                'index': i,
                'concurso': draw_info['concurso'],
                'sum': sum(numbers),
                'mean': np.mean(numbers),
                'std': np.std(numbers),
                'min': min(numbers),
                'max': max(numbers),
                'range': max(numbers) - min(numbers),
                'median': np.median(numbers)
            }
            
            # Features estatísticas avançadas
            feature_vector.update({
                'skewness': stats.skew(numbers),
                'kurtosis': stats.kurtosis(numbers),
                'variance': np.var(numbers),
                'coefficient_of_variation': np.std(numbers) / np.mean(numbers) if np.mean(numbers) > 0 else 0
            })
            
            # Features de distribuição
            feature_vector.update({
                'first_quartile': np.percentile(numbers, 25),
                'third_quartile': np.percentile(numbers, 75),
                'iqr': np.percentile(numbers, 75) - np.percentile(numbers, 25),
                'mad': np.median(np.abs(numbers - np.median(numbers)))  # Median Absolute Deviation
            })
            
            # Features de padrões
            gaps = [numbers[j+1] - numbers[j] for j in range(len(numbers)-1)]
            feature_vector.update({
                'gap_mean': np.mean(gaps),
                'gap_std': np.std(gaps),
                'gap_max': max(gaps),
                'gap_min': min(gaps),
                'consecutive_pairs': sum(1 for gap in gaps if gap == 1)
            })
            
            # Features de paridade
            evens = sum(1 for n in numbers if n % 2 == 0)
            odds = 6 - evens
            feature_vector.update({
                'even_count': evens,
                'odd_count': odds,
                'even_odd_ratio': evens / odds if odds > 0 else 6
            })
            
            # Features de dezenas
            low_numbers = sum(1 for n in numbers if n <= 30)
            high_numbers = 6 - low_numbers
            feature_vector.update({
                'low_count': low_numbers,
                'high_count': high_numbers,
                'low_high_ratio': low_numbers / high_numbers if high_numbers > 0 else 6
            })
            
            # Features temporais (se dados de data disponíveis)
            if draw_info.get('date'):
                try:
                    if isinstance(draw_info['date'], str):
                        date = pd.to_datetime(draw_info['date'], dayfirst=True)
                    else:
                        date = draw_info['date']
                    
                    feature_vector.update({
                        'day_of_week': date.dayofweek,
                        'month': date.month,
                        'quarter': date.quarter,
                        'day_of_year': date.dayofyear,
                        'is_weekend': 1 if date.dayofweek >= 5 else 0
                    })
                except:
                    feature_vector.update({
                        'day_of_week': 0, 'month': 1, 'quarter': 1,
                        'day_of_year': 1, 'is_weekend': 0
                    })
            else:
                feature_vector.update({
                    'day_of_week': 0, 'month': 1, 'quarter': 1,
                    'day_of_year': 1, 'is_weekend': 0
                })
            
            # Features de correlação com sorteio anterior
            if i > 0:
                prev_numbers = self.historical_data[i-1]['numbers']
                intersection = len(set(numbers) & set(prev_numbers))
                jaccard = intersection / len(set(numbers) | set(prev_numbers))
                
                feature_vector.update({
                    'prev_intersection': intersection,
                    'prev_jaccard': jaccard,
                    'prev_sum_diff': abs(sum(numbers) - sum(prev_numbers)),
                    'prev_mean_diff': abs(np.mean(numbers) - np.mean(prev_numbers))
                })
            else:
                feature_vector.update({
                    'prev_intersection': 0,
                    'prev_jaccard': 0,
                    'prev_sum_diff': 0,
                    'prev_mean_diff': 0
                })
            
            # Features de periodicidade
            feature_vector.update({
                'sum_mod_7': sum(numbers) % 7,
                'sum_mod_13': sum(numbers) % 13,
                'digit_sum': sum(int(d) for n in numbers for d in str(n)),
                'product_last_digits': np.prod([n % 10 for n in numbers])
            })
            
            features.append(feature_vector)
        
        self.features_df = pd.DataFrame(features)
        self.features_matrix = self.features_df.select_dtypes(include=[np.number]).values
        
        print(f"   ✓ {len(features)} vetores de features extraídos")
        print(f"   ✓ {self.features_matrix.shape[1]} características por sorteio")
        
        return self.features_df
    
    def detect_change_points_multiple_methods(self):
        """Detecta pontos de mudança usando múltiplos métodos"""
        print("\n🔍 Detectando pontos de mudança com múltiplos métodos...")
        
        if self.features_matrix is None:
            self.extract_comprehensive_features()
        
        change_points = {}
        
        # 1. Clustering baseado em janelas deslizantes
        change_points['sliding_window'] = self._sliding_window_clustering()
        
        # 2. Detecção por mudança de distribuição
        change_points['distribution_shift'] = self._distribution_shift_detection()
        
        # 3. Análise de componentes principais
        change_points['pca_based'] = self._pca_change_detection()
        
        # 4. Detecção de outliers
        change_points['outlier_based'] = self._outlier_based_detection()
        
        # 5. Análise de autocorrelação
        change_points['autocorrelation'] = self._autocorrelation_change_detection()
        
        # 6. Teste de estacionariedade
        change_points['stationarity'] = self._stationarity_test()
        
        # 7. Detecção por machine learning
        change_points['ml_based'] = self._ml_change_detection()
        
        # Consolidar resultados
        self.change_points = self._consolidate_change_points(change_points)
        
        self.analysis_results['change_detection'] = {
            'methods': change_points,
            'consolidated': self.change_points
        }
        
        print(f"   ✓ {len(self.change_points)} pontos de mudança consolidados")
        
        return self.change_points
    
    def _sliding_window_clustering(self, window_size=50, step=10):
        """Clustering em janelas deslizantes"""
        print("     🔄 Análise por janelas deslizantes...")
        
        change_points = []
        n_samples = len(self.features_matrix)
        
        # Normalizar dados
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(self.features_matrix)
        
        prev_cluster_centers = None
        
        for start in range(0, n_samples - window_size, step):
            end = start + window_size
            window_data = normalized_data[start:end]
            
            # Clustering K-means
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(window_data)
            
            # Comparar com janela anterior
            if prev_cluster_centers is not None:
                # Calcular distância entre centros de clusters
                center_distance = np.linalg.norm(
                    kmeans.cluster_centers_ - prev_cluster_centers
                )
                
                # Threshold para mudança significativa
                if center_distance > 2.0:
                    change_points.append(start + window_size // 2)
            
            prev_cluster_centers = kmeans.cluster_centers_.copy()
        
        return change_points
    
    def _distribution_shift_detection(self):
        """Detecção baseada em mudança de distribuição estatística"""
        print("     📊 Análise de mudança de distribuição...")
        
        change_points = []
        window_size = 100
        
        # Usar soma dos números como proxy da distribuição
        series = self.features_df['sum'].values
        
        for i in range(window_size, len(series) - window_size, 20):
            before = series[i-window_size:i]
            after = series[i:i+window_size]
            
            # Teste Kolmogorov-Smirnov
            ks_stat, p_value = stats.ks_2samp(before, after)
            
            # Teste t
            t_stat, t_p_value = stats.ttest_ind(before, after)
            
            # Mann-Whitney U test
            u_stat, u_p_value = stats.mannwhitneyu(before, after, alternative='two-sided')
            
            # Se múltiplos testes indicam mudança
            significant_tests = sum([
                p_value < 0.01,
                t_p_value < 0.01,
                u_p_value < 0.01
            ])
            
            if significant_tests >= 2:
                change_points.append(i)
        
        return change_points
    
    def _pca_change_detection(self):
        """Detecção baseada em análise de componentes principais"""
        print("     🎯 Análise PCA...")
        
        change_points = []
        window_size = 80
        
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(self.features_matrix)
        
        prev_components = None
        
        for i in range(window_size, len(normalized_data) - window_size, 15):
            window_data = normalized_data[i-window_size:i+window_size]
            
            # PCA
            pca = PCA(n_components=5)
            pca.fit(window_data)
            
            if prev_components is not None:
                # Calcular mudança nos componentes principais
                component_change = np.mean([
                    1 - abs(np.dot(prev_components[j], pca.components_[j]))
                    for j in range(min(len(prev_components), len(pca.components_)))
                ])
                
                if component_change > 0.3:
                    change_points.append(i)
            
            prev_components = pca.components_.copy()
        
        return change_points
    
    def _outlier_based_detection(self):
        """Detecção baseada em análise de outliers"""
        print("     🎪 Análise de outliers...")
        
        # Isolation Forest para detectar outliers
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(self.features_matrix)
        outlier_scores = iso_forest.decision_function(self.features_matrix)
        outliers = iso_forest.predict(self.features_matrix) == -1
        
        # Pontos de mudança onde há clusters de outliers
        change_points = []
        outlier_indices = np.where(outliers)[0]
        
        # Agrupar outliers próximos
        if len(outlier_indices) > 0:
            groups = []
            current_group = [outlier_indices[0]]
            
            for i in range(1, len(outlier_indices)):
                if outlier_indices[i] - outlier_indices[i-1] <= 10:
                    current_group.append(outlier_indices[i])
                else:
                    groups.append(current_group)
                    current_group = [outlier_indices[i]]
            groups.append(current_group)
            
            # Centro de cada grupo como ponto de mudança
            for group in groups:
                if len(group) >= 3:  # Pelo menos 3 outliers próximos
                    change_points.append(int(np.mean(group)))
        
        return change_points
    
    def _autocorrelation_change_detection(self):
        """Detecção baseada em mudanças na autocorrelação"""
        print("     🔄 Análise de autocorrelação...")
        
        change_points = []
        series = self.features_df['sum'].values
        window_size = 60
        
        def autocorr(x, max_lag=20):
            """Calcula autocorrelação até max_lag"""
            autocorrs = []
            for lag in range(1, min(max_lag, len(x)//2)):
                if lag < len(x):
                    corr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
                    autocorrs.append(corr if not np.isnan(corr) else 0)
                else:
                    autocorrs.append(0)
            return np.array(autocorrs)
        
        prev_autocorr = None
        
        for i in range(window_size, len(series) - window_size, 20):
            window_data = series[i-window_size:i+window_size]
            current_autocorr = autocorr(window_data)
            
            if prev_autocorr is not None and len(current_autocorr) > 0 and len(prev_autocorr) > 0:
                # Calcular diferença na estrutura de autocorrelação
                min_len = min(len(current_autocorr), len(prev_autocorr))
                if min_len > 0:
                    autocorr_diff = np.mean(np.abs(current_autocorr[:min_len] - prev_autocorr[:min_len]))
                    
                    if autocorr_diff > 0.2:
                        change_points.append(i)
            
            prev_autocorr = current_autocorr.copy()
        
        return change_points
    
    def _stationarity_test(self):
        """Teste de estacionariedade usando ADF"""
        print("     📈 Teste de estacionariedade...")
        
        from scipy.stats import chi2
        
        change_points = []
        series = self.features_df['sum'].values
        window_size = 100
        
        def simple_adf_test(x):
            """Versão simplificada do teste ADF"""
            if len(x) < 10:
                return 0.5
            
            # Diferença primeira
            diff = np.diff(x)
            
            # Teste de normalidade das diferenças
            _, p_value = stats.normaltest(diff)
            
            # Teste de tendência
            t_trend = np.arange(len(x))
            corr_trend, p_trend = stats.pearsonr(x, t_trend)
            
            # Combinar evidências
            return min(p_value, p_trend)
        
        for i in range(window_size, len(series) - window_size, 25):
            window_data = series[i-window_size:i+window_size]
            
            # Teste de estacionariedade
            p_value = simple_adf_test(window_data)
            
            # Se não estacionário (p_value > 0.05), pode indicar mudança
            if p_value > 0.05:
                change_points.append(i)
        
        return change_points
    
    def _ml_change_detection(self):
        """Detecção usando técnicas de machine learning"""
        print("     🤖 Análise por ML...")
        
        change_points = []
        
        # Usar DBSCAN para detectar clusters temporais
        scaler = StandardScaler()
        
        # Adicionar componente temporal
        temporal_features = np.column_stack([
            self.features_matrix,
            np.arange(len(self.features_matrix)).reshape(-1, 1)
        ])
        
        normalized_data = scaler.fit_transform(temporal_features)
        
        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        clusters = dbscan.fit_predict(normalized_data)
        
        # Encontrar mudanças de cluster
        prev_cluster = clusters[0]
        for i, cluster in enumerate(clusters[1:], 1):
            if cluster != prev_cluster and cluster != -1:  # -1 é noise
                change_points.append(i)
            prev_cluster = cluster
        
        return change_points
    
    def _consolidate_change_points(self, methods_results):
        """Consolida pontos de mudança de diferentes métodos"""
        print("     🔗 Consolidando resultados...")
        
        all_points = []
        for method, points in methods_results.items():
            all_points.extend(points)
        
        if not all_points:
            return []
        
        # Ordenar pontos
        all_points = sorted(set(all_points))
        
        # Agrupar pontos próximos (dentro de 30 posições)
        consolidated = []
        current_group = [all_points[0]]
        
        for point in all_points[1:]:
            if point - current_group[-1] <= 30:
                current_group.append(point)
            else:
                # Usar mediana do grupo como ponto representativo
                consolidated.append(int(np.median(current_group)))
                current_group = [point]
        
        # Adicionar último grupo
        if current_group:
            consolidated.append(int(np.median(current_group)))
        
        # Filtrar pontos com suporte de múltiplos métodos
        final_points = []
        for point in consolidated:
            support_count = 0
            for method, points in methods_results.items():
                if any(abs(p - point) <= 30 for p in points):
                    support_count += 1
            
            # Manter apenas pontos com suporte de pelo menos 2 métodos
            if support_count >= 2:
                final_points.append(point)
        
        return final_points
    
    def analyze_regime_characteristics(self):
        """Analisa características de cada regime entre pontos de mudança"""
        print("\n📋 Analisando características dos regimes...")
        
        if not self.change_points:
            self.detect_change_points_multiple_methods()
        
        # Definir regimes
        regime_boundaries = [0] + self.change_points + [len(self.historical_data)]
        regimes = []
        
        for i in range(len(regime_boundaries) - 1):
            start = regime_boundaries[i]
            end = regime_boundaries[i + 1]
            
            regime_data = self.historical_data[start:end]
            regime_features = self.features_df.iloc[start:end]
            
            # Calcular estatísticas do regime
            regime_stats = {
                'start_index': start,
                'end_index': end,
                'duration': end - start,
                'start_concurso': regime_data[0]['concurso'],
                'end_concurso': regime_data[-1]['concurso'],
                'mean_sum': regime_features['sum'].mean(),
                'std_sum': regime_features['sum'].std(),
                'mean_range': regime_features['range'].mean(),
                'correlation_pattern': self._calculate_regime_correlation(regime_data),
                'entropy': self._calculate_regime_entropy(regime_data),
                'dominant_patterns': self._find_dominant_patterns(regime_data)
            }
            
            regimes.append(regime_stats)
        
        self.analysis_results['regimes'] = regimes
        
        print(f"   ✓ {len(regimes)} regimes identificados")
        for i, regime in enumerate(regimes):
            print(f"     Regime {i+1}: Concursos {regime['start_concurso']}-{regime['end_concurso']} ({regime['duration']} sorteios)")
        
        return regimes
    
    def _calculate_regime_correlation(self, regime_data):
        """Calcula padrão de correlação dentro de um regime"""
        if len(regime_data) < 2:
            return 0
        
        correlations = []
        for i in range(len(regime_data) - 1):
            draw1 = set(regime_data[i]['numbers'])
            draw2 = set(regime_data[i + 1]['numbers'])
            
            intersection = len(draw1 & draw2)
            union = len(draw1 | draw2)
            
            jaccard = intersection / union if union > 0 else 0
            correlations.append(jaccard)
        
        return np.mean(correlations) if correlations else 0
    
    def _calculate_regime_entropy(self, regime_data):
        """Calcula entropia do regime"""
        # Frequência de cada número
        freq_dist = np.zeros(60)
        for draw_info in regime_data:
            for num in draw_info['numbers']:
                freq_dist[num - 1] += 1
        
        # Normalizar para probabilidades
        total = np.sum(freq_dist)
        if total > 0:
            prob_dist = freq_dist / total
            # Calcular entropia de Shannon
            entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
            return entropy / np.log2(60)  # Normalizar
        
        return 0
    
    def _find_dominant_patterns(self, regime_data):
        """Encontra padrões dominantes no regime"""
        if len(regime_data) < 10:
            return {}
        
        # Padrões de soma
        sums = [sum(draw['numbers']) for draw in regime_data]
        sum_mean = np.mean(sums)
        sum_std = np.std(sums)
        
        # Padrões de paridade
        even_counts = [sum(1 for n in draw['numbers'] if n % 2 == 0) for draw in regime_data]
        even_mean = np.mean(even_counts)
        
        # Padrões de range
        ranges = [max(draw['numbers']) - min(draw['numbers']) for draw in regime_data]
        range_mean = np.mean(ranges)
        
        return {
            'sum_pattern': {'mean': sum_mean, 'std': sum_std},
            'even_pattern': {'mean': even_mean},
            'range_pattern': {'mean': range_mean}
        }
    
    def generate_temporal_report(self):
        """Gera relatório detalhado da análise temporal"""
        print("\n📊 Gerando relatório temporal...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_draws': len(self.historical_data),
            'analysis_methods': list(self.analysis_results.get('change_detection', {}).get('methods', {}).keys()),
            'change_points': self.change_points,
            'regimes': self.analysis_results.get('regimes', []),
            'summary': {
                'total_change_points': len(self.change_points),
                'total_regimes': len(self.analysis_results.get('regimes', [])),
                'average_regime_duration': np.mean([r['duration'] for r in self.analysis_results.get('regimes', [])]) if self.analysis_results.get('regimes') else 0
            }
        }
        
        # Salvar relatório
        report_path = f"../output/temporal_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   ✓ Relatório salvo em: {report_path}")
        
        return report
    
    def visualize_temporal_analysis(self):
        """Cria visualizações da análise temporal"""
        print("\n📈 Gerando visualizações temporais...")
        
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        # 1. Série temporal com pontos de mudança
        ax1 = axes[0, 0]
        sums = [sum(draw['numbers']) for draw in self.historical_data]
        ax1.plot(sums, alpha=0.7, linewidth=1)
        
        for cp in self.change_points:
            ax1.axvline(x=cp, color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        ax1.set_title('Série Temporal - Soma dos Números\ncom Pontos de Mudança')
        ax1.set_xlabel('Concurso')
        ax1.set_ylabel('Soma')
        ax1.grid(True, alpha=0.3)
        
        # 2. Heatmap de features ao longo do tempo
        ax2 = axes[0, 1]
        key_features = ['sum', 'std', 'range', 'even_count', 'gap_mean']
        
        if hasattr(self, 'features_df'):
            heatmap_data = self.features_df[key_features].T
            im = ax2.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')
            ax2.set_title('Heatmap de Features ao Longo do Tempo')
            ax2.set_xlabel('Concurso')
            ax2.set_ylabel('Features')
            ax2.set_yticks(range(len(key_features)))
            ax2.set_yticklabels(key_features)
            plt.colorbar(im, ax=ax2)
        
        # 3. Distribuição por regime
        ax3 = axes[1, 0]
        if 'regimes' in self.analysis_results:
            regime_sums = []
            regime_labels = []
            
            for i, regime in enumerate(self.analysis_results['regimes']):
                start = regime['start_index']
                end = regime['end_index']
                regime_sum_data = [sum(self.historical_data[j]['numbers']) for j in range(start, end)]
                regime_sums.extend(regime_sum_data)
                regime_labels.extend([f'Regime {i+1}'] * len(regime_sum_data))
            
            # Box plot por regime
            unique_regimes = list(set(regime_labels))
            regime_data = [regime_sums[i] for i, label in enumerate(regime_labels)]
            
            # Preparar dados para boxplot
            boxplot_data = []
            boxplot_labels = []
            for regime in unique_regimes:
                regime_values = [regime_sums[i] for i, label in enumerate(regime_labels) if label == regime]
                boxplot_data.append(regime_values)
                boxplot_labels.append(regime)
            
            ax3.boxplot(boxplot_data, labels=boxplot_labels)
            ax3.set_title('Distribuição de Somas por Regime')
            ax3.set_ylabel('Soma dos Números')
            ax3.tick_params(axis='x', rotation=45)
        
        # 4. Correlação entre sorteios consecutivos
        ax4 = axes[1, 1]
        correlations = []
        for i in range(len(self.historical_data) - 1):
            draw1 = set(self.historical_data[i]['numbers'])
            draw2 = set(self.historical_data[i + 1]['numbers'])
            jaccard = len(draw1 & draw2) / len(draw1 | draw2)
            correlations.append(jaccard)
        
        ax4.plot(correlations, alpha=0.6, linewidth=1)
        for cp in self.change_points:
            if cp < len(correlations):
                ax4.axvline(x=cp, color='red', linestyle='--', alpha=0.8)
        
        ax4.set_title('Correlação entre Sorteios Consecutivos')
        ax4.set_xlabel('Concurso')
        ax4.set_ylabel('Correlação Jaccard')
        ax4.grid(True, alpha=0.3)
        
        # 5. PCA dos regimes
        ax5 = axes[2, 0]
        if hasattr(self, 'features_matrix') and 'regimes' in self.analysis_results:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Preparar dados
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(self.features_matrix)
            
            # PCA
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(normalized_data)
            
            # Colorir por regime
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.analysis_results['regimes'])))
            
            for i, regime in enumerate(self.analysis_results['regimes']):
                start = regime['start_index']
                end = regime['end_index']
                ax5.scatter(pca_data[start:end, 0], pca_data[start:end, 1], 
                           c=[colors[i]], label=f"Regime {i+1}", alpha=0.6)
            
            ax5.set_title('Análise PCA - Separação por Regimes')
            ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variância)')
            ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variância)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Entropia por janela temporal
        ax6 = axes[2, 1]
        window_size = 50
        entropies = []
        
        for i in range(window_size, len(self.historical_data) - window_size, 10):
            window_data = self.historical_data[i-window_size:i+window_size]
            entropy = self._calculate_regime_entropy(window_data)
            entropies.append(entropy)
        
        entropy_indices = range(window_size, len(self.historical_data) - window_size, 10)
        ax6.plot(entropy_indices, entropies, alpha=0.7, linewidth=2)
        
        for cp in self.change_points:
            if window_size <= cp <= len(self.historical_data) - window_size:
                ax6.axvline(x=cp, color='red', linestyle='--', alpha=0.8)
        
        ax6.set_title('Entropia por Janela Temporal')
        ax6.set_xlabel('Concurso')
        ax6.set_ylabel('Entropia Normalizada')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../output/temporal_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✓ Visualizações salvas em: temporal_analysis_comprehensive.png")

# Script de execução
if __name__ == "__main__":
    from seed_discovery_engine import SeedDiscoveryEngine
    
    print("🕒 ANÁLISE TEMPORAL AVANÇADA")
    print("="*70)
    
    # Carregar dados
    engine = SeedDiscoveryEngine()
    data_path = "../data/MegaSena3.xlsx"
    engine.load_megasena_data(data_path)
    
    # Inicializar detector
    detector = TemporalChangeDetector(engine.historical_data)
    
    # Executar análise completa
    detector.extract_comprehensive_features()
    detector.detect_change_points_multiple_methods()
    detector.analyze_regime_characteristics()
    
    # Gerar relatórios
    detector.generate_temporal_report()
    detector.visualize_temporal_analysis()
    
    print("\n✅ Análise temporal completa!")
    print(f"   Pontos de mudança detectados: {detector.change_points}")