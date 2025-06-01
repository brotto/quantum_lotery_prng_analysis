#!/usr/bin/env python3
"""
Script principal para descoberta do seed
Coordena todos os módulos e estratégias
"""

import os
import sys
import json
from datetime import datetime
from seed_discovery_engine import SeedDiscoveryEngine
from advanced_pattern_analyzer import AdvancedPatternAnalyzer

def print_banner():
    """Exibe banner do sistema"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          SISTEMA DE DESCOBERTA DE SEED - MEGA SENA           ║
║                  Baseado em Análise Quântica                  ║
║                                                               ║
║  Descobertas:                                                 ║
║  • Algoritmo: LCG (a=1, c=0-4, m=2^31-1)                    ║
║  • Correlação: 0.754 entre sorteios                         ║
║  • Pontos de mudança: 200, 2350, 2400                       ║
╚══════════════════════════════════════════════════════════════╝
    """)

def validate_environment():
    """Valida se o ambiente está configurado corretamente"""
    print("\n🔍 Validando ambiente...")
    
    # Verificar arquivo de dados
    data_path = "../data/MegaSena3.xlsx"
    if not os.path.exists(data_path):
        print(f"❌ Arquivo de dados não encontrado: {data_path}")
        print("   Por favor, copie MegaSena3.xlsx para a pasta data/")
        return False
    
    # Verificar diretórios de saída
    os.makedirs("../output", exist_ok=True)
    os.makedirs("../output/cache", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)
    
    print("✅ Ambiente validado com sucesso")
    return True

def run_complete_discovery():
    """Executa o processo completo de descoberta"""
    print_banner()
    
    if not validate_environment():
        return
    
    # Inicializar engine
    print("\n🚀 Inicializando sistema...")
    engine = SeedDiscoveryEngine()
    
    # Carregar dados
    data_path = "../data/MegaSena3.xlsx"
    engine.load_megasena_data(data_path)
    
    # Inicializar analisador
    analyzer = AdvancedPatternAnalyzer(engine)
    
    # FASE 1: Análise de padrões
    print("\n" + "="*60)
    print("FASE 1: ANÁLISE DE PADRÕES")
    print("="*60)
    
    temporal_patterns = analyzer.analyze_temporal_patterns()
    transition_patterns = analyzer.analyze_number_transitions()
    
    # FASE 2: Busca de seeds
    print("\n" + "="*60)
    print("FASE 2: BUSCA SISTEMÁTICA DE SEEDS")
    print("="*60)
    
    all_candidates = engine.parallel_seed_search()
    
    if not all_candidates:
        print("\n❌ Nenhum candidato encontrado na busca inicial")
        print("   Tentando estratégia alternativa...")
        
        # Estratégia alternativa: buscar em mais pontos
        extended_points = list(range(100, len(engine.historical_data), 100))
        for point in extended_points[:10]:
            candidates = engine.search_seed_at_change_point(point)
            all_candidates.extend(candidates)
    
    # FASE 3: Validação e refinamento
    print("\n" + "="*60)
    print("FASE 3: VALIDAÇÃO E REFINAMENTO")
    print("="*60)
    
    if all_candidates:
        # Encontrar assinaturas
        signatures = analyzer.find_seed_signature(all_candidates)
        
        if signatures:
            # Melhor candidato
            best_signature = signatures[0]
            best_candidate = None
            
            # Encontrar o candidato correspondente
            for candidate in all_candidates:
                if (candidate['seed'] == best_signature['seed'] and 
                    candidate['c'] == best_signature['c']):
                    best_candidate = candidate
                    break
            
            if best_candidate:
                print(f"\n🏆 MELHOR CANDIDATO IDENTIFICADO:")
                print(f"   Seed: {best_candidate['seed']}")
                print(f"   c: {best_candidate['c']}")
                print(f"   Score: {best_candidate['combined_score']:.3f}")
                
                # Validação completa
                print("\n📊 Validação contra histórico completo...")
                validation = engine.validate_recent_draws(
                    best_candidate['seed'], 
                    best_candidate['c'], 
                    min(1000, len(engine.historical_data))
                )
                
                if validation:
                    print(f"   Taxa de sucesso: {validation['success_rate']:.1%}")
                    print(f"   Média de acertos: {validation['average_hits']:.2f}")
                
                # Gerar relatório final
                report = analyzer.generate_validation_report(best_candidate)
                
                # Previsões finais
                print("\n🔮 PREVISÕES PARA PRÓXIMOS SORTEIOS:")
                predictions = engine.generate_predictions(
                    best_candidate['seed'], 
                    best_candidate['c'], 
                    10
                )
                
                for i, pred in enumerate(predictions[:5]):
                    print(f"   Sorteio +{i+1}: {pred['numbers']}")
                
                # Salvar previsões
                predictions_file = f"../output/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(predictions_file, 'w') as f:
                    json.dump({
                        'seed': best_candidate['seed'],
                        'c': best_candidate['c'],
                        'predictions': predictions,
                        'validation': validation
                    }, f, indent=2)
                
                print(f"\n💾 Previsões salvas em: {predictions_file}")
    
    # Gerar visualizações
    print("\n📊 Gerando visualizações finais...")
    analyzer.visualize_analysis()
    
    print("\n" + "="*60)
    print("✅ DESCOBERTA COMPLETA!")
    print("="*60)
    
    # Resumo final
    print("\n📋 RESUMO DOS RESULTADOS:")
    print(f"   Total de candidatos encontrados: {len(all_candidates)}")
    
    if all_candidates:
        print(f"   Melhor score obtido: {all_candidates[0]['combined_score']:.3f}")
        print(f"   Arquivos gerados em: output/")
        
        print("\n⚠️  IMPORTANTE:")
        print("   1. Valide as previsões com sorteios futuros")
        print("   2. O sistema pode ter mudado desde a análise")
        print("   3. Use os resultados de forma responsável")
    else:
        print("   ❌ Nenhum seed válido descoberto")
        print("   Possíveis razões:")
        print("   - O sistema usa um PRNG mais complexo")
        print("   - Os parâmetros LCG identificados mudaram")
        print("   - Necessária análise mais profunda")

if __name__ == "__main__":
    try:
        run_complete_discovery()
    except KeyboardInterrupt:
        print("\n\n⚠️  Processo interrompido pelo usuário")
    except Exception as e:
        print(f"\n\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()