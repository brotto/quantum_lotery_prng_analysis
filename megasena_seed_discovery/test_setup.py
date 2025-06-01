#!/usr/bin/env python3
"""
Testa se o ambiente está configurado corretamente
"""

print("🧪 Testando configuração...")

try:
    import numpy as np
    print("✅ NumPy instalado")
except:
    print("❌ NumPy não instalado")

try:
    import pandas as pd
    print("✅ Pandas instalado")
except:
    print("❌ Pandas não instalado")

try:
    import scipy
    print("✅ SciPy instalado")
except:
    print("❌ SciPy não instalado")

try:
    from numba import jit
    print("✅ Numba instalado")
except:
    print("❌ Numba não instalado")

try:
    from tqdm import tqdm
    print("✅ tqdm instalado")
except:
    print("❌ tqdm não instalado")

print("\n📁 Verificando estrutura de diretórios...")
import os

dirs = ['src', 'data', 'output', 'logs']
for d in dirs:
    if os.path.exists(d):
        print(f"✅ Diretório {d}/ existe")
    else:
        print(f"❌ Diretório {d}/ não existe")

print("\n✅ Teste concluído!")