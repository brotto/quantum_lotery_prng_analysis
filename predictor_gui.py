#!/usr/bin/env python3
"""
Mega Sena Predictor - Interface Gráfica
======================================
Interface gráfica para o sistema de predição da Mega Sena
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from datetime import datetime
from mega_sena_predictor import MegaSenaPredictor

class MegaSenaPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mega Sena Quantum Predictor")
        self.root.geometry("800x600")
        self.root.configure(bg='#1e1e1e')
        
        # Variáveis
        self.predictor = None
        self.is_loading = False
        
        # Cores
        self.colors = {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'accent': '#0078d4',
            'success': '#28a745',
            'warning': '#ffc107',
            'error': '#dc3545',
            'card': '#2d2d2d'
        }
        
        self.setup_ui()
        self.load_predictor()
    
    def setup_ui(self):
        """Configura a interface do usuário"""
        # Título
        title_frame = tk.Frame(self.root, bg=self.colors['bg'])
        title_frame.pack(fill='x', padx=20, pady=10)
        
        title_label = tk.Label(
            title_frame, 
            text="🎯 MEGA SENA QUANTUM PREDICTOR",
            font=('Arial', 18, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['accent']
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Sistema de Predição Baseado em Análise Quântica",
            font=('Arial', 10),
            bg=self.colors['bg'],
            fg=self.colors['fg']
        )
        subtitle_label.pack()
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Frame de configurações
        config_frame = tk.LabelFrame(
            main_frame,
            text="Configurações de Predição",
            font=('Arial', 12, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['fg']
        )
        config_frame.pack(fill='x', pady=5)
        
        # Método de predição
        tk.Label(
            config_frame,
            text="Método:",
            font=('Arial', 10),
            bg=self.colors['card'],
            fg=self.colors['fg']
        ).grid(row=0, column=0, sticky='w', padx=10, pady=5)
        
        self.method_var = tk.StringVar(value="hybrid")
        method_combo = ttk.Combobox(
            config_frame,
            textvariable=self.method_var,
            values=["hybrid", "lcg", "quantum", "statistical"],
            state="readonly",
            width=15
        )
        method_combo.grid(row=0, column=1, padx=10, pady=5)
        
        # Quantidade de números
        tk.Label(
            config_frame,
            text="Números:",
            font=('Arial', 10),
            bg=self.colors['card'],
            fg=self.colors['fg']
        ).grid(row=0, column=2, sticky='w', padx=10, pady=5)
        
        self.numbers_var = tk.StringVar(value="6")
        numbers_combo = ttk.Combobox(
            config_frame,
            textvariable=self.numbers_var,
            values=["6", "7", "8", "9"],
            state="readonly",
            width=5
        )
        numbers_combo.grid(row=0, column=3, padx=10, pady=5)
        
        # Quantidade de predições
        tk.Label(
            config_frame,
            text="Predições:",
            font=('Arial', 10),
            bg=self.colors['card'],
            fg=self.colors['fg']
        ).grid(row=1, column=0, sticky='w', padx=10, pady=5)
        
        self.count_var = tk.StringVar(value="1")
        count_spinbox = tk.Spinbox(
            config_frame,
            from_=1,
            to=20,
            textvariable=self.count_var,
            width=5
        )
        count_spinbox.grid(row=1, column=1, padx=10, pady=5)
        
        # Botões
        button_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        button_frame.pack(fill='x', pady=10)
        
        self.predict_button = tk.Button(
            button_frame,
            text="🔮 GERAR PREDIÇÃO",
            font=('Arial', 12, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            command=self.generate_prediction,
            height=2,
            width=20
        )
        self.predict_button.pack(side='left', padx=5)
        
        self.compare_button = tk.Button(
            button_frame,
            text="📊 COMPARAR MÉTODOS",
            font=('Arial', 12, 'bold'),
            bg=self.colors['success'],
            fg='white',
            command=self.compare_methods,
            height=2,
            width=20
        )
        self.compare_button.pack(side='left', padx=5)
        
        self.clear_button = tk.Button(
            button_frame,
            text="🗑️ LIMPAR",
            font=('Arial', 12, 'bold'),
            bg=self.colors['error'],
            fg='white',
            command=self.clear_results,
            height=2,
            width=15
        )
        self.clear_button.pack(side='right', padx=5)
        
        # Área de resultados
        results_frame = tk.LabelFrame(
            main_frame,
            text="Resultados",
            font=('Arial', 12, 'bold'),
            bg=self.colors['card'],
            fg=self.colors['fg']
        )
        results_frame.pack(fill='both', expand=True, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            width=80,
            height=20,
            font=('Consolas', 10),
            bg='#000000',
            fg='#00ff00',
            insertbackground='#00ff00'
        )
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Pronto",
            font=('Arial', 9),
            bg=self.colors['card'],
            fg=self.colors['fg'],
            anchor='w'
        )
        self.status_bar.pack(fill='x', side='bottom')
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.root,
            mode='indeterminate'
        )
    
    def load_predictor(self):
        """Carrega o preditor em background"""
        def load():
            try:
                self.update_status("Carregando dados históricos...")
                self.progress.pack(fill='x', side='bottom')
                self.progress.start()
                
                file_path = '/Users/alebrotto/Downloads/quantum_mega_pseudo-aleatorio/Mega-Sena-3.xlsx'
                self.predictor = MegaSenaPredictor(file_path)
                
                self.progress.stop()
                self.progress.pack_forget()
                self.update_status("Sistema carregado e pronto para uso!")
                
                # Ativar botões
                self.predict_button.config(state='normal')
                self.compare_button.config(state='normal')
                
            except Exception as e:
                self.progress.stop()
                self.progress.pack_forget()
                self.update_status(f"Erro ao carregar: {str(e)}")
                messagebox.showerror("Erro", f"Erro ao carregar o sistema:\n{str(e)}")
        
        # Desativar botões durante carregamento
        self.predict_button.config(state='disabled')
        self.compare_button.config(state='disabled')
        
        # Executar em thread separada
        threading.Thread(target=load, daemon=True).start()
    
    def update_status(self, message):
        """Atualiza a barra de status"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def generate_prediction(self):
        """Gera predição com base nas configurações"""
        if not self.predictor:
            messagebox.showerror("Erro", "Sistema não carregado!")
            return
        
        try:
            method = self.method_var.get()
            numbers_count = int(self.numbers_var.get())
            prediction_count = int(self.count_var.get())
            
            self.update_status("Gerando predições...")
            
            # Executar predição em thread separada
            def predict():
                try:
                    if prediction_count == 1:
                        predictions = [self.predictor.generate_single_prediction(method, numbers_count)]
                    else:
                        predictions = self.predictor.generate_multiple_predictions(
                            prediction_count, method, numbers_count
                        )
                    
                    # Gerar relatório
                    report = self.predictor.get_prediction_report(predictions, method)
                    
                    # Atualizar interface na thread principal
                    self.root.after(0, lambda: self.display_results(report))
                    self.root.after(0, lambda: self.update_status("Predições geradas com sucesso!"))
                    
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro na predição:\n{str(e)}"))
                    self.root.after(0, lambda: self.update_status("Erro na predição"))
            
            threading.Thread(target=predict, daemon=True).start()
            
        except ValueError as e:
            messagebox.showerror("Erro", f"Configuração inválida:\n{str(e)}")
    
    def compare_methods(self):
        """Compara todos os métodos de predição"""
        if not self.predictor:
            messagebox.showerror("Erro", "Sistema não carregado!")
            return
        
        try:
            numbers_count = int(self.numbers_var.get())
            
            self.update_status("Comparando métodos...")
            
            def compare():
                try:
                    methods = ["lcg", "quantum", "statistical", "hybrid"]
                    comparison_results = []
                    
                    comparison_results.append("=" * 80)
                    comparison_results.append("COMPARAÇÃO DE MÉTODOS DE PREDIÇÃO")
                    comparison_results.append("=" * 80)
                    comparison_results.append(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
                    comparison_results.append(f"Números por predição: {numbers_count}")
                    comparison_results.append("")
                    
                    for method in methods:
                        prediction = self.predictor.generate_single_prediction(method, numbers_count)
                        confidence = self.predictor.analyze_prediction_confidence(prediction)
                        
                        comparison_results.append(f"MÉTODO: {method.upper()}")
                        comparison_results.append(f"  Números: {' - '.join(f'{num:02d}' for num in prediction)}")
                        comparison_results.append(f"  Confiança Geral: {confidence['overall_confidence']:.2%}")
                        comparison_results.append(f"  Score Quântico: {confidence['quantum_score']:.4f}")
                        comparison_results.append(f"  Score de Frequência: {confidence['frequency_score']['balance']:.2%}")
                        comparison_results.append(f"  Score de Soma: {confidence['sum_score']:.2%}")
                        comparison_results.append("")
                    
                    comparison_results.append("RECOMENDAÇÃO:")
                    comparison_results.append("O método HYBRID combina todos os algoritmos para")
                    comparison_results.append("obter a melhor predição possível.")
                    comparison_results.append("=" * 80)
                    
                    result_text = "\n".join(comparison_results)
                    
                    # Atualizar interface
                    self.root.after(0, lambda: self.display_results(result_text))
                    self.root.after(0, lambda: self.update_status("Comparação concluída!"))
                    
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro na comparação:\n{str(e)}"))
                    self.root.after(0, lambda: self.update_status("Erro na comparação"))
            
            threading.Thread(target=compare, daemon=True).start()
            
        except ValueError as e:
            messagebox.showerror("Erro", f"Configuração inválida:\n{str(e)}")
    
    def display_results(self, text):
        """Exibe resultados na área de texto"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)
    
    def clear_results(self):
        """Limpa a área de resultados"""
        self.results_text.delete(1.0, tk.END)
        self.update_status("Resultados limpos")

def main():
    """Função principal da GUI"""
    root = tk.Tk()
    app = MegaSenaPredictorGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nSaindo...")

if __name__ == "__main__":
    main()