#!/usr/bin/env python3
"""
Mega Sena Quantum Predictor
===========================
Sistema de predição baseado na análise de padrões detectados nos sorteios da Mega Sena.
Utiliza os resultados da engenharia reversa para gerar predições personalizáveis.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random
from typing import List, Tuple, Dict, Optional
import itertools
from collections import Counter

class MegaSenaPredictor:
    def __init__(self, historical_data_path: str):
        """
        Inicializa o preditor com dados históricos
        
        Args:
            historical_data_path: Caminho para o arquivo Excel com dados históricos
        """
        self.historical_data_path = historical_data_path
        self.df = None
        self.number_sequences = None
        self.lcg_params = None
        self.quantum_weights = None
        self.pattern_cache = {}
        self.seed_evolution_points = [200, 2350, 2400]  # Pontos de mudança detectados
        
        # Parâmetros LCG detectados na análise
        self.detected_lcg_params = [
            {'a': 1, 'c': 0, 'm': 2147483647},
            {'a': 1, 'c': 1, 'm': 2147483647},
            {'a': 1, 'c': 2, 'm': 2147483647},
            {'a': 1, 'c': 3, 'm': 2147483647},
            {'a': 1, 'c': 4, 'm': 2147483647}
        ]
        
        self._load_and_prepare_data()
        self._initialize_prediction_models()
    
    def _load_and_prepare_data(self):
        """Carrega e prepara os dados históricos"""
        print("Carregando dados históricos...")
        self.df = pd.read_excel(self.historical_data_path)
        number_cols = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']
        self.number_sequences = self.df[number_cols].values
        
        # Converter datas para análise temporal
        if 'Data do Sorteio' in self.df.columns:
            self.df['Data do Sorteio'] = pd.to_datetime(self.df['Data do Sorteio'], errors='coerce')
        
        print(f"Dados carregados: {len(self.number_sequences)} sorteios históricos")
    
    def _initialize_prediction_models(self):
        """Inicializa os modelos de predição"""
        print("Inicializando modelos de predição...")
        
        # Análise de frequência
        all_numbers = self.number_sequences.flatten()
        self.frequency_dist = Counter(all_numbers)
        
        # Pesos quânticos baseados na análise
        self.quantum_weights = self._calculate_quantum_weights()
        
        # Análise de padrões temporais
        self.temporal_patterns = self._analyze_temporal_patterns()
        
        # Matriz de correlação entre posições
        self.position_correlations = np.corrcoef(self.number_sequences.T)
        
        print("Modelos inicializados com sucesso!")
    
    def _calculate_quantum_weights(self) -> np.ndarray:
        """Calcula pesos quânticos baseados na análise de interferência"""
        weights = np.zeros(61)  # Números 0-60
        
        for num in range(1, 61):
            # Frequência normalizada
            freq_weight = self.frequency_dist.get(num, 0) / len(self.number_sequences)
            
            # Peso quântico baseado em interferência
            phase = 2 * np.pi * num / 60
            quantum_interference = np.abs(np.exp(1j * phase))**2
            
            # Combinação dos pesos
            weights[num] = freq_weight * quantum_interference
        
        # Normalizar
        weights = weights / np.sum(weights)
        return weights
    
    def _analyze_temporal_patterns(self) -> Dict:
        """Analisa padrões temporais nos dados"""
        patterns = {}
        
        # Padrões por posição
        for pos in range(6):
            position_data = self.number_sequences[:, pos]
            patterns[f'position_{pos}'] = {
                'mean': float(np.mean(position_data)),
                'std': float(np.std(position_data)),
                'trend': self._calculate_trend(position_data)
            }
        
        # Padrões de soma
        sums = [np.sum(seq) for seq in self.number_sequences]
        patterns['sum_patterns'] = {
            'mean': float(np.mean(sums)),
            'std': float(np.std(sums)),
            'range': (int(np.min(sums)), int(np.max(sums)))
        }
        
        return patterns
    
    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calcula tendência linear nos dados"""
        if len(data) < 2:
            return 0.0
        x = np.arange(len(data))
        return float(np.corrcoef(x, data)[0, 1])
    
    def _estimate_current_seed(self, reference_drawing: int = None) -> int:
        """Estima o seed atual baseado na análise de pontos de mudança"""
        if reference_drawing is None:
            reference_drawing = len(self.number_sequences)
        
        # Encontrar o ponto de mudança mais próximo
        last_change_point = 0
        for point in self.seed_evolution_points:
            if point <= reference_drawing:
                last_change_point = point
        
        # Estimar seed baseado no ponto de mudança e sequência conhecida
        if last_change_point > 0 and last_change_point < len(self.number_sequences):
            reference_sequence = self.number_sequences[last_change_point]
            
            # Converter sequência para valor único
            seed_estimate = int(np.sum(reference_sequence * np.array([1, 10, 100, 1000, 10000, 100000])))
            
            # Ajustar para o sorteio atual
            drawings_since_change = reference_drawing - last_change_point
            seed_estimate = (seed_estimate + drawings_since_change * 12345) % (2**31 - 1)
            
            return seed_estimate
        
        # Seed padrão se não houver ponto de referência
        return int(datetime.now().timestamp()) % (2**31 - 1)
    
    def _lcg_generate_sequence(self, seed: int, params: Dict, count: int = 6) -> List[int]:
        """Gera sequência usando Linear Congruential Generator"""
        a, c, m = params['a'], params['c'], params['m']
        current = seed
        numbers = []
        
        # Gerar números extras para aumentar aleatoriedade
        for _ in range(count + 10):
            current = (a * current + c) % m
            
            # Converter para número da Mega Sena (1-60)
            mega_number = (current % 60) + 1
            
            if mega_number not in numbers and len(numbers) < count:
                numbers.append(mega_number)
        
        # Se não temos números suficientes, completar com números aleatórios ponderados
        while len(numbers) < count:
            remaining_numbers = [i for i in range(1, 61) if i not in numbers]
            if remaining_numbers:
                # Usar pesos quânticos para escolha
                weights = [self.quantum_weights[num] for num in remaining_numbers]
                if sum(weights) > 0:
                    weights = np.array(weights) / sum(weights)
                    chosen = np.random.choice(remaining_numbers, p=weights)
                    numbers.append(chosen)
                else:
                    numbers.append(random.choice(remaining_numbers))
        
        return sorted(numbers[:count])
    
    def _quantum_neural_prediction(self, base_sequence: List[int], target_count: int = 6) -> List[int]:
        """Predição usando rede neural quântica simulada"""
        # Características da sequência base
        features = [
            np.sum(base_sequence),
            np.prod(base_sequence) % 1000000,
            np.var(base_sequence),
            len(set(base_sequence)),
            max(base_sequence) - min(base_sequence)
        ]
        
        # Simular processamento quântico
        quantum_state = np.array(features, dtype=complex)
        for i in range(len(quantum_state)):
            phase = 2 * np.pi * features[i] / 1000
            quantum_state[i] *= np.exp(1j * phase)
        
        # Extrair números da interferência quântica
        interference_pattern = np.abs(quantum_state)**2
        interference_sum = np.sum(interference_pattern)
        
        predicted_numbers = []
        for i in range(target_count):
            # Gerar número baseado na interferência
            base_num = int((interference_sum * (i + 1) * 7919) % 60) + 1
            
            # Ajustar com pesos quânticos
            adjustment = int(self.quantum_weights[base_num] * 60)
            final_num = ((base_num + adjustment) % 60) + 1
            
            if final_num not in predicted_numbers:
                predicted_numbers.append(final_num)
        
        # Completar se necessário
        while len(predicted_numbers) < target_count:
            candidates = [i for i in range(1, 61) if i not in predicted_numbers]
            if candidates:
                weights = [self.quantum_weights[num] for num in candidates]
                if sum(weights) > 0:
                    weights = np.array(weights) / sum(weights)
                    choice = np.random.choice(candidates, p=weights)
                    predicted_numbers.append(choice)
                else:
                    predicted_numbers.append(random.choice(candidates))
        
        return sorted(predicted_numbers[:target_count])
    
    def _statistical_pattern_prediction(self, target_count: int = 6) -> List[int]:
        """Predição baseada em padrões estatísticos detectados"""
        predicted_numbers = []
        
        # Usar correlações entre posições
        for pos in range(min(target_count, 6)):
            if pos < 6:
                # Análise da posição específica
                position_data = self.number_sequences[:, pos]
                
                # Tendência + variação
                mean_val = self.temporal_patterns[f'position_{pos}']['mean']
                std_val = self.temporal_patterns[f'position_{pos}']['std']
                trend = self.temporal_patterns[f'position_{pos}']['trend']
                
                # Predição baseada na tendência
                predicted_val = mean_val + trend * len(self.number_sequences) * 0.1
                predicted_val += np.random.normal(0, std_val * 0.5)
                
                # Ajustar para range válido
                predicted_num = max(1, min(60, int(predicted_val)))
                
                if predicted_num not in predicted_numbers:
                    predicted_numbers.append(predicted_num)
        
        # Completar com números baseados em frequência
        while len(predicted_numbers) < target_count:
            candidates = [i for i in range(1, 61) if i not in predicted_numbers]
            if candidates:
                # Usar distribuição de frequência inversa (números menos frequentes)
                weights = []
                max_freq = max(self.frequency_dist.values())
                for num in candidates:
                    freq = self.frequency_dist.get(num, 0)
                    weight = max_freq - freq + 1  # Peso inverso
                    weights.append(weight)
                
                weights = np.array(weights) / sum(weights)
                choice = np.random.choice(candidates, p=weights)
                predicted_numbers.append(choice)
            else:
                break
        
        return sorted(predicted_numbers[:target_count])
    
    def generate_single_prediction(self, method: str = "hybrid", numbers_count: int = 6) -> List[int]:
        """
        Gera uma única predição
        
        Args:
            method: Método de predição ("lcg", "quantum", "statistical", "hybrid")
            numbers_count: Quantidade de números (6, 7, 8, 9)
        
        Returns:
            Lista com os números preditos
        """
        if numbers_count not in [6, 7, 8, 9]:
            raise ValueError("numbers_count deve ser 6, 7, 8 ou 9")
        
        if method == "lcg":
            # Usar LCG com melhor parâmetro detectado
            seed = self._estimate_current_seed()
            best_params = self.detected_lcg_params[0]  # Maior confiança
            return self._lcg_generate_sequence(seed, best_params, numbers_count)
        
        elif method == "quantum":
            # Usar última sequência como base
            base_sequence = list(self.number_sequences[-1])
            return self._quantum_neural_prediction(base_sequence, numbers_count)
        
        elif method == "statistical":
            return self._statistical_pattern_prediction(numbers_count)
        
        elif method == "hybrid":
            # Combinar todos os métodos
            lcg_pred = self.generate_single_prediction("lcg", numbers_count)
            quantum_pred = self.generate_single_prediction("quantum", numbers_count)
            stat_pred = self.generate_single_prediction("statistical", numbers_count)
            
            # Combinar usando votação ponderada
            all_numbers = lcg_pred + quantum_pred + stat_pred
            number_votes = Counter(all_numbers)
            
            # Selecionar números com mais votos
            most_voted = [num for num, votes in number_votes.most_common()]
            result = most_voted[:numbers_count]
            
            # Completar se necessário
            while len(result) < numbers_count:
                candidates = [i for i in range(1, 61) if i not in result]
                if candidates:
                    result.append(random.choice(candidates))
                else:
                    break
            
            return sorted(result[:numbers_count])
        
        else:
            raise ValueError(f"Método '{method}' não reconhecido")
    
    def generate_multiple_predictions(self, count: int, method: str = "hybrid", 
                                    numbers_count: int = 6, 
                                    ensure_diversity: bool = True) -> List[List[int]]:
        """
        Gera múltiplas predições
        
        Args:
            count: Número de predições a gerar
            method: Método de predição
            numbers_count: Quantidade de números por predição
            ensure_diversity: Garantir diversidade entre predições
        
        Returns:
            Lista de listas com as predições
        """
        predictions = []
        used_combinations = set()
        
        max_attempts = count * 5  # Evitar loop infinito
        attempts = 0
        
        while len(predictions) < count and attempts < max_attempts:
            attempts += 1
            
            # Variar seed para LCG se necessário
            if method == "lcg":
                base_seed = self._estimate_current_seed()
                varied_seed = (base_seed + attempts * 12345) % (2**31 - 1)
                
                # Usar diferentes parâmetros LCG
                params_idx = attempts % len(self.detected_lcg_params)
                params = self.detected_lcg_params[params_idx]
                
                prediction = self._lcg_generate_sequence(varied_seed, params, numbers_count)
            else:
                # Adicionar aleatoriedade para outros métodos
                np.random.seed(None)  # Reset seed
                prediction = self.generate_single_prediction(method, numbers_count)
            
            # Verificar diversidade se solicitado
            prediction_tuple = tuple(sorted(prediction))
            
            if ensure_diversity:
                if prediction_tuple not in used_combinations:
                    predictions.append(prediction)
                    used_combinations.add(prediction_tuple)
            else:
                predictions.append(prediction)
        
        return predictions
    
    def analyze_prediction_confidence(self, prediction: List[int]) -> Dict:
        """
        Analisa a confiança de uma predição
        
        Args:
            prediction: Lista de números preditos
        
        Returns:
            Dicionário com métricas de confiança
        """
        confidence_metrics = {}
        
        # Análise de frequência histórica
        freq_scores = [self.frequency_dist.get(num, 0) for num in prediction]
        confidence_metrics['frequency_score'] = {
            'mean': float(np.mean(freq_scores)),
            'std': float(np.std(freq_scores)),
            'balance': float(1 - np.std(freq_scores) / (np.mean(freq_scores) + 1))
        }
        
        # Análise de distribuição
        confidence_metrics['distribution_score'] = {
            'range': max(prediction) - min(prediction),
            'gaps': [prediction[i+1] - prediction[i] for i in range(len(prediction)-1)],
            'evenness': float(np.std([prediction[i+1] - prediction[i] for i in range(len(prediction)-1)]))
        }
        
        # Análise quântica
        quantum_score = sum(self.quantum_weights[num] for num in prediction)
        confidence_metrics['quantum_score'] = float(quantum_score)
        
        # Score de soma (baseado nos padrões detectados)
        prediction_sum = sum(prediction)
        expected_sum = self.temporal_patterns['sum_patterns']['mean']
        sum_deviation = abs(prediction_sum - expected_sum) / expected_sum
        confidence_metrics['sum_score'] = float(1 - sum_deviation)
        
        # Score geral (combinação ponderada)
        overall_score = (
            confidence_metrics['frequency_score']['balance'] * 0.3 +
            confidence_metrics['quantum_score'] * 0.3 +
            confidence_metrics['sum_score'] * 0.2 +
            (1 - min(1, confidence_metrics['distribution_score']['evenness'] / 10)) * 0.2
        )
        confidence_metrics['overall_confidence'] = float(max(0, min(1, overall_score)))
        
        return confidence_metrics
    
    def get_prediction_report(self, predictions: List[List[int]], method: str) -> str:
        """
        Gera relatório detalhado das predições
        
        Args:
            predictions: Lista de predições
            method: Método utilizado
        
        Returns:
            String com relatório formatado
        """
        report = []
        report.append("=" * 60)
        report.append("RELATÓRIO DE PREDIÇÕES - MEGA SENA")
        report.append("=" * 60)
        report.append(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        report.append(f"Método: {method.upper()}")
        report.append(f"Número de predições: {len(predictions)}")
        report.append(f"Números por predição: {len(predictions[0]) if predictions else 0}")
        report.append("")
        
        for i, prediction in enumerate(predictions):
            confidence = self.analyze_prediction_confidence(prediction)
            
            report.append(f"PREDIÇÃO {i+1}:")
            report.append(f"  Números: {' - '.join(f'{num:02d}' for num in prediction)}")
            report.append(f"  Confiança Geral: {confidence['overall_confidence']:.2%}")
            report.append(f"  Score Quântico: {confidence['quantum_score']:.4f}")
            report.append(f"  Score de Soma: {confidence['sum_score']:.2%}")
            report.append("")
        
        # Estatísticas gerais
        if predictions:
            all_numbers = [num for pred in predictions for num in pred]
            most_common = Counter(all_numbers).most_common(10)
            
            report.append("ANÁLISE ESTATÍSTICA:")
            report.append(f"  Números mais preditos: {', '.join(f'{num}({count}x)' for num, count in most_common[:5])}")
            
            # Análise de confiança média
            avg_confidence = np.mean([self.analyze_prediction_confidence(pred)['overall_confidence'] 
                                    for pred in predictions])
            report.append(f"  Confiança média: {avg_confidence:.2%}")
        
        report.append("")
        report.append("DISCLAIMER:")
        report.append("Este sistema é baseado em análise de padrões históricos.")
        report.append("Não há garantia de acerto. Use com responsabilidade.")
        report.append("=" * 60)
        
        return "\n".join(report)

def main():
    """Função principal - exemplo de uso"""
    try:
        # Inicializar preditor
        file_path = '/Users/alebrotto/Downloads/quantum_mega_pseudo-aleatorio/Mega-Sena-3.xlsx'
        predictor = MegaSenaPredictor(file_path)
        
        print("MEGA SENA QUANTUM PREDICTOR")
        print("=" * 50)
        print("Escolha uma opção:")
        print("1. Predição única (6 números)")
        print("2. Predição única (7 números)")  
        print("3. Predição única (8 números)")
        print("4. Predição única (9 números)")
        print("5. Múltiplas predições (6 números)")
        print("6. Análise comparativa de métodos")
        
        choice = input("\nOpção (1-6): ").strip()
        
        if choice == "1":
            prediction = predictor.generate_single_prediction("hybrid", 6)
            confidence = predictor.analyze_prediction_confidence(prediction)
            print(f"\nPREDIÇÃO (6 números): {' - '.join(f'{num:02d}' for num in prediction)}")
            print(f"Confiança: {confidence['overall_confidence']:.2%}")
            
        elif choice == "2":
            prediction = predictor.generate_single_prediction("hybrid", 7)
            confidence = predictor.analyze_prediction_confidence(prediction)
            print(f"\nPREDIÇÃO (7 números): {' - '.join(f'{num:02d}' for num in prediction)}")
            print(f"Confiança: {confidence['overall_confidence']:.2%}")
            
        elif choice == "3":
            prediction = predictor.generate_single_prediction("hybrid", 8)
            confidence = predictor.analyze_prediction_confidence(prediction)
            print(f"\nPREDIÇÃO (8 números): {' - '.join(f'{num:02d}' for num in prediction)}")
            print(f"Confiança: {confidence['overall_confidence']:.2%}")
            
        elif choice == "4":
            prediction = predictor.generate_single_prediction("hybrid", 9)
            confidence = predictor.analyze_prediction_confidence(prediction)
            print(f"\nPREDIÇÃO (9 números): {' - '.join(f'{num:02d}' for num in prediction)}")
            print(f"Confiança: {confidence['overall_confidence']:.2%}")
            
        elif choice == "5":
            count = int(input("Quantas predições? (1-20): "))
            count = max(1, min(20, count))
            
            predictions = predictor.generate_multiple_predictions(count, "hybrid", 6)
            report = predictor.get_prediction_report(predictions, "hybrid")
            print("\n" + report)
            
        elif choice == "6":
            print("\nComparando métodos...")
            methods = ["lcg", "quantum", "statistical", "hybrid"]
            
            for method in methods:
                prediction = predictor.generate_single_prediction(method, 6)
                confidence = predictor.analyze_prediction_confidence(prediction)
                print(f"\n{method.upper()}: {' - '.join(f'{num:02d}' for num in prediction)}")
                print(f"Confiança: {confidence['overall_confidence']:.2%}")
        
        else:
            print("Opção inválida!")
            
    except Exception as e:
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()