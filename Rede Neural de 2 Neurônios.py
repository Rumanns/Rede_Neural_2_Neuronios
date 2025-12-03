# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 23:04:08 2025

@author: Rumanns
"""

import random

# Dados da porta AND
dados = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1)
]

# Inicializa pesos ALEATORIAMENTE
w1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)
b = random.uniform(-1, 1)
taxa_aprendizado = 0.1

print(f"Pesos INICIAIS aleatórios: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}")

# TREINAMENTO (o algoritmo REAL)
for epoca in range(100):  # Máximo 100 épocas
    erros = 0
    
    for entradas, target in dados:
        x1, x2 = entradas
        
        # 1. FAZ PREDIÇÃO (feedforward)
        z = w1*x1 + w2*x2 + b
        y = 1 if z >= 0 else 0
        
        # 2. VERIFICA SE ACERTOU
        if y != target:
            erros += 1
            
            # 3. AJUSTA PESOS (a parte IMPORTANTE!)
            erro = target - y  # será -1 ou 1
            
            w1 = w1 + taxa_aprendizado * erro * x1
            w2 = w2 + taxa_aprendizado * erro * x2
            b = b + taxa_aprendizado * erro
    
    # Se acertou tudo, PARA!
    if erros == 0:
        print(f"Convergiu na época {epoca+1}!")
        break

print(f"Pesos FINAIS: w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}")

# TESTE FINAL
print("\nTestando rede treinada:")
for entradas, target in dados:
    x1, x2 = entradas
    z = w1*x1 + w2*x2 + b
    y = 1 if z >= 0 else 0
    print(f"({x1},{x2}) → z={z:.1f} → y={y} (deveria ser {target}) {'✓' if y == target else '✗'}")