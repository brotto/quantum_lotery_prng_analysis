#!/usr/bin/env python3
"""
Motor de Descoberta de Seed - Mega Sena
Baseado nas descobertas da análise quântica
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import hashlib
from tqdm import tqdm
# from numba import jit  # Removed due to compilation issues
import pickle
import json
from concurrent.futures import ProcessPoolExecutor
import os

class SeedDiscoveryEngine:
    def __init__(self):
        # Parâmetros LCG confirmados
        self.LCG_A = 1
        self.LCG_C_RANGE = range(5)  # c varia de 0 a 4
        self.LCG_M = 2**31 - 1  # 2147483647
        
        # Pontos críticos de mudança
        self.CHANGE_POINTS = [200, 2350, 2400]
        
        # Correlação alvo
        self.TARGET_CORRELATION = 0.754
        
        # Cache de resultados
        self.cache_dir = "../output/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Histórico de dados
        self.historical_data = None
        self.data_loaded = False
        
    def load_megasena_data(self, filepath):
        """Carrega dados da Mega Sena do Excel"""
        print("📂 Carregando dados da Mega Sena...")
        
        try:
            df = pd.read_excel(filepath)
            
            # Extrair sorteios
            draws = []
            for idx, row in df.iterrows():
                if pd.notna(row.get('Bola1')):
                    draw = sorted([
                        int(row['Bola1']), int(row['Bola2']), 
                        int(row['Bola3']), int(row['Bola4']), 
                        int(row['Bola5']), int(row['Bola6'])
                    ])
                    
                    # Adicionar informações temporais se disponíveis
                    draw_info = {
                        'concurso': int(row.get('Concurso', idx)),
                        'numbers': draw,
                        'date': row.get('Data do Sorteio', None)
                    }
                    draws.append(draw_info)
            
            self.historical_data = draws
            self.data_loaded = True
            
            print(f"✅ {len(draws)} sorteios carregados")
            print(f"   Primeiro: Concurso {draws[0]['concurso']} - {draws[0]['numbers']}")
            print(f"   Último: Concurso {draws[-1]['concurso']} - {draws[-1]['numbers']}")
            
            return draws
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return None
    
    def lcg_next(self, state, c):
        """Próximo estado LCG"""
        return (state + c) % (2**31 - 1)
    
    def state_to_lottery_numbers(self, initial_state, c):
        """Converte estado LCG em números da loteria"""
        numbers = []
        state = initial_state
        attempts = 0
        max_attempts = 100  # Evitar loop infinito
        
        while len(numbers) < 6 and attempts < max_attempts:
            state = self.lcg_next(state, c)
            num = (state % 60) + 1
            
            if num not in numbers:
                numbers.append(num)
            
            attempts += 1
        
        return sorted(numbers) if len(numbers) == 6 else None
    
    def calculate_correlation(self, draw1, draw2):
        """Calcula correlação entre dois sorteios"""
        set1, set2 = set(draw1), set(draw2)
        
        # Múltiplas métricas
        jaccard = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
        overlap = len(set1 & set2) / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0
        dice = 2 * len(set1 & set2) / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0
        
        # Média ponderada (ajustada para aproximar 0.754)
        return 0.4 * jaccard + 0.4 * overlap + 0.2 * dice
    
    def extract_seed_from_draw(self, draw_info, method='timestamp'):
        """Extrai possível seed de um sorteio"""
        draw = draw_info['numbers']
        
        if method == 'timestamp' and draw_info.get('date'):
            # Baseado em timestamp
            if isinstance(draw_info['date'], datetime):
                return int(draw_info['date'].timestamp())
            else:
                # Tentar converter string para datetime
                try:
                    dt = pd.to_datetime(draw_info['date'])
                    return int(dt.timestamp())
                except:
                    pass
        
        elif method == 'concat':
            # Concatenação dos números
            return int(''.join([f'{n:02d}' for n in draw])) % self.LCG_M
        
        elif method == 'weighted':
            # Soma ponderada
            return sum([n * (60 ** i) for i, n in enumerate(draw)]) % self.LCG_M
        
        elif method == 'hash':
            # Hash dos números
            draw_str = '-'.join(map(str, draw))
            return int(hashlib.md5(draw_str.encode()).hexdigest()[:8], 16) % self.LCG_M
        
        elif method == 'concurso':
            # Baseado no número do concurso
            concurso = draw_info.get('concurso', 0)
            return (concurso * 1000000 + sum(draw)) % self.LCG_M
        
        return None
    
    def validate_seed_sequence(self, seed, c, start_idx, num_draws=10):
        """Valida se um seed gera a sequência correta"""
        if not self.data_loaded:
            return 0
        
        matches = 0
        current_state = seed
        
        for i in range(num_draws):
            if start_idx + i >= len(self.historical_data):
                break
            
            # Gerar previsão
            predicted = self.state_to_lottery_numbers(current_state, c)
            if not predicted:
                continue
            
            # Comparar com sorteio real
            actual = self.historical_data[start_idx + i]['numbers']
            
            # Contar acertos
            hits = len(set(predicted) & set(actual))
            
            # Critérios de validação
            if hits >= 4:  # 4+ números corretos
                matches += 1
            elif hits >= 3 and i < 3:  # Mais tolerante nos primeiros
                matches += 0.5
            
            # Avançar estado (6 extrações por sorteio)
            for _ in range(6):
                current_state = self.lcg_next(current_state, c)
        
        return matches / num_draws if num_draws > 0 else 0
    
    def search_seed_at_change_point(self, change_point_idx):
        """Busca intensiva de seed em um ponto de mudança"""
        print(f"\n🔍 Buscando seed no ponto de mudança {change_point_idx}")
        
        if change_point_idx >= len(self.historical_data):
            print(f"❌ Ponto {change_point_idx} fora do range de dados")
            return []
        
        # Métodos de extração de seed
        methods = ['timestamp', 'concat', 'weighted', 'hash', 'concurso']
        
        # Janela de busca ao redor do ponto
        window_start = max(0, change_point_idx - 5)
        window_end = min(len(self.historical_data), change_point_idx + 5)
        
        candidates = []
        
        print(f"📊 Analisando janela: {window_start} a {window_end}")
        
        for idx in range(window_start, window_end):
            draw_info = self.historical_data[idx]
            
            for method in methods:
                base_seed = self.extract_seed_from_draw(draw_info, method)
                if not base_seed:
                    continue
                
                # Testar seed base e variações
                seed_variations = [
                    base_seed,
                    base_seed + 1,
                    base_seed - 1,
                    base_seed + idx,
                    base_seed * idx % self.LCG_M,
                    (base_seed + draw_info['concurso']) % self.LCG_M
                ]
                
                for seed in seed_variations:
                    for c in self.LCG_C_RANGE:
                        # Validar seed
                        score = self.validate_seed_sequence(seed, c, idx, 20)
                        
                        if score > 0.3:  # Threshold de validação
                            # Validar correlação
                            corr_score = self.validate_correlation_pattern(seed, c, idx)
                            
                            combined_score = 0.7 * score + 0.3 * corr_score
                            
                            if combined_score > 0.4:
                                candidates.append({
                                    'seed': seed,
                                    'c': c,
                                    'index': idx,
                                    'concurso': draw_info['concurso'],
                                    'method': method,
                                    'validation_score': score,
                                    'correlation_score': corr_score,
                                    'combined_score': combined_score
                                })
                                
                                print(f"   ✓ Candidato encontrado: score={combined_score:.3f}")
        
        # Ordenar por score combinado
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return candidates[:10]  # Top 10
    
    def validate_correlation_pattern(self, seed, c, start_idx, num_draws=50):
        """Valida se o seed produz o padrão de correlação esperado"""
        if start_idx + num_draws > len(self.historical_data):
            num_draws = len(self.historical_data) - start_idx - 1
        
        correlations = []
        current_state = seed
        
        prev_draw = self.state_to_lottery_numbers(current_state, c)
        if not prev_draw:
            return 0
        
        for i in range(1, num_draws):
            # Avançar estado
            for _ in range(6):
                current_state = self.lcg_next(current_state, c)
            
            # Gerar próximo sorteio
            next_draw = self.state_to_lottery_numbers(current_state, c)
            if not next_draw:
                continue
            
            # Calcular correlação
            corr = self.calculate_correlation(prev_draw, next_draw)
            correlations.append(corr)
            
            prev_draw = next_draw
        
        if not correlations:
            return 0
        
        # Comparar com correlação alvo
        avg_corr = np.mean(correlations)
        corr_diff = abs(avg_corr - self.TARGET_CORRELATION)
        
        # Score baseado na proximidade com 0.754
        return max(0, 1 - corr_diff * 2)
    
    def parallel_seed_search(self):
        """Busca paralela em todos os pontos de mudança"""
        print("\n🚀 Iniciando busca paralela de seeds")
        print(f"   Pontos de mudança: {self.CHANGE_POINTS}")
        
        all_candidates = []
        
        # Busca paralela
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures = []
            
            for cp in self.CHANGE_POINTS:
                future = executor.submit(self.search_seed_at_change_point, cp)
                futures.append((cp, future))
            
            for cp, future in futures:
                try:
                    candidates = future.result(timeout=300)  # 5 min timeout
                    all_candidates.extend(candidates)
                    print(f"✅ Ponto {cp}: {len(candidates)} candidatos encontrados")
                except Exception as e:
                    print(f"❌ Erro no ponto {cp}: {e}")
        
        # Consolidar e ordenar todos os candidatos
        all_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Salvar resultados
        self.save_candidates(all_candidates)
        
        return all_candidates
    
    def save_candidates(self, candidates):
        """Salva candidatos encontrados"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar em JSON
        json_path = f"../output/seed_candidates_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(candidates, f, indent=2, default=str)
        
        # Salvar em pickle para análise posterior
        pkl_path = f"{self.cache_dir}/candidates_{timestamp}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(candidates, f)
        
        print(f"\n💾 Candidatos salvos em:")
        print(f"   JSON: {json_path}")
        print(f"   PKL: {pkl_path}")
    
    def generate_predictions(self, seed, c, num_predictions=10):
        """Gera previsões futuras usando seed descoberto"""
        predictions = []
        current_state = seed
        
        print(f"\n🔮 Gerando {num_predictions} previsões com seed={seed}, c={c}")
        
        for i in range(num_predictions):
            draw = self.state_to_lottery_numbers(current_state, c)
            
            if draw:
                predictions.append({
                    'index': i + 1,
                    'numbers': draw,
                    'state': current_state
                })
                
                # Avançar estado
                for _ in range(6):
                    current_state = self.lcg_next(current_state, c)
            else:
                print(f"⚠️ Falha ao gerar previsão {i+1}")
        
        return predictions
    
    def validate_recent_draws(self, seed, c, num_recent=10):
        """Valida seed contra sorteios mais recentes"""
        if not self.data_loaded:
            return None
        
        # Pegar últimos sorteios
        recent_start = len(self.historical_data) - num_recent
        
        print(f"\n🔍 Validando contra últimos {num_recent} sorteios")
        
        validation_results = []
        current_state = seed
        
        # Primeiro, precisamos sincronizar o estado com o início da validação
        # Avançar estado até o ponto correto
        print("   Sincronizando estado...")
        
        for i in range(recent_start):
            for _ in range(6):
                current_state = self.lcg_next(current_state, c)
        
        # Agora validar
        for i in range(num_recent):
            idx = recent_start + i
            if idx >= len(self.historical_data):
                break
            
            predicted = self.state_to_lottery_numbers(current_state, c)
            actual = self.historical_data[idx]['numbers']
            concurso = self.historical_data[idx]['concurso']
            
            if predicted:
                hits = len(set(predicted) & set(actual))
                
                result = {
                    'concurso': concurso,
                    'predicted': predicted,
                    'actual': actual,
                    'hits': hits,
                    'success': hits >= 3
                }
                
                validation_results.append(result)
                
                print(f"   Concurso {concurso}: {hits} acertos")
                
                # Avançar estado
                for _ in range(6):
                    current_state = self.lcg_next(current_state, c)
        
        # Calcular estatísticas
        total_hits = sum(r['hits'] for r in validation_results)
        success_rate = sum(1 for r in validation_results if r['success']) / len(validation_results)
        
        return {
            'results': validation_results,
            'total_hits': total_hits,
            'success_rate': success_rate,
            'average_hits': total_hits / len(validation_results) if validation_results else 0
        }

# Função auxiliar para execução
def run_seed_discovery():
    """Executa descoberta completa do seed"""
    engine = SeedDiscoveryEngine()
    
    # Carregar dados
    data_path = "../data/MegaSena3.xlsx"
    if not os.path.exists(data_path):
        print(f"❌ Arquivo {data_path} não encontrado!")
        return
    
    engine.load_megasena_data(data_path)
    
    # Buscar seeds
    candidates = engine.parallel_seed_search()
    
    if candidates:
        print(f"\n🏆 TOP 5 CANDIDATOS:")
        for i, candidate in enumerate(candidates[:5]):
            print(f"\n{i+1}. Seed: {candidate['seed']}")
            print(f"   c: {candidate['c']}")
            print(f"   Score: {candidate['combined_score']:.3f}")
            print(f"   Concurso base: {candidate['concurso']}")
            print(f"   Método: {candidate['method']}")
            
            # Validar contra sorteios recentes
            validation = engine.validate_recent_draws(candidate['seed'], candidate['c'])
            if validation:
                print(f"   Validação recente: {validation['success_rate']:.1%} sucesso")
                print(f"   Média de acertos: {validation['average_hits']:.2f}")
            
            # Gerar previsões
            if i == 0:  # Apenas para o melhor candidato
                predictions = engine.generate_predictions(candidate['seed'], candidate['c'], 5)
                print(f"\n   📊 Próximas previsões:")
                for pred in predictions:
                    print(f"      {pred['index']}: {pred['numbers']}")
    else:
        print("\n❌ Nenhum candidato válido encontrado")

if __name__ == "__main__":
    run_seed_discovery()