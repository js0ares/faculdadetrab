import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path


class ModeloRegressaoLinear:
    """Classe para análise de regressão linear simples"""
    
    def __init__(self):
        self.dados_x = None
        self.dados_y = None
        self.coef_angular = None
        self.intercepto = None
        self.r_quadrado = None
        self.predicoes = None
    
    def carregar_dados(self, caminho_x='X.txt', caminho_y='y.txt'):
        """Carrega os dados dos arquivos de texto"""
        try:
            self.dados_x = np.loadtxt(caminho_x)
            self.dados_y = np.loadtxt(caminho_y)
            
            print("📂 Dados carregados com sucesso!")
            print(f"   Total de observações: {len(self.dados_x)}")
            print(f"   Variável X: [{self.dados_x.min():.2f}, {self.dados_x.max():.2f}]")
            print(f"   Variável Y: [{self.dados_y.min():.2f}, {self.dados_y.max():.2f}]")
            
            return True
            
        except FileNotFoundError:
            print("❌ Erro: Um ou mais arquivos não foram encontrados!")
            return False
        except Exception as erro:
            print(f"❌ Erro ao carregar dados: {erro}")
            return False
    
    def treinar_modelo(self):
        """Calcula os coeficientes da regressão linear usando mínimos quadrados"""
        if self.dados_x is None or self.dados_y is None:
            print("❌ Carregue os dados primeiro!")
            return False
        
        print("\n🔧 Treinando modelo de regressão linear...")
        
        # Construir matriz de design
        n_observacoes = len(self.dados_x)
        matriz_design = np.column_stack([np.ones(n_observacoes), self.dados_x])
        
        # Método dos mínimos quadrados: β = (X'X)^(-1) X'y
        xtx = matriz_design.T @ matriz_design
        xty = matriz_design.T @ self.dados_y
        coeficientes = np.linalg.solve(xtx, xty)
        
        self.intercepto = coeficientes[0]
        self.coef_angular = coeficientes[1]
        
        # Calcular predições
        self.predicoes = self.intercepto + self.coef_angular * self.dados_x
        
        print("✅ Modelo treinado com sucesso!\n")
        
        return True
    
    def calcular_metricas(self):
        """Calcula métricas de qualidade do modelo"""
        if self.predicoes is None:
            print("❌ Treine o modelo primeiro!")
            return
        
        # R² (Coeficiente de determinação)
        residuos = self.dados_y - self.predicoes
        soma_quad_residuos = np.sum(residuos ** 2)
        soma_quad_total = np.sum((self.dados_y - np.mean(self.dados_y)) ** 2)
        self.r_quadrado = 1 - (soma_quad_residuos / soma_quad_total)
        
        # MSE (Erro Quadrático Médio)
        mse = soma_quad_residuos / len(self.dados_y)
        rmse = np.sqrt(mse)
        
        print("📊 MÉTRICAS DO MODELO:")
        print("─" * 50)
        print(f"   Intercepto (β₀):      {self.intercepto:>10.4f}")
        print(f"   Coef. Angular (β₁):   {self.coef_angular:>10.4f}")
        print(f"   R² (R-squared):       {self.r_quadrado:>10.4f}")
        print(f"   RMSE:                 {rmse:>10.4f}")
        print("─" * 50)
        print(f"   Equação: ŷ = {self.intercepto:.4f} + {self.coef_angular:.4f}·x")
        print()
    
    def plotar_resultados(self, salvar_como='analise_regressao.html'):
        """Gera visualização interativa dos resultados"""
        if self.predicoes is None:
            print("❌ Treine o modelo primeiro!")
            return None
        
        print("📈 Gerando visualização...")
        
        # Criar figura
        grafico = go.Figure()
        
        # Adicionar pontos observados
        grafico.add_trace(go.Scatter(
            x=self.dados_x,
            y=self.dados_y,
            mode='markers',
            name='Dados Reais',
            marker=dict(
                color='#3498db',
                size=7,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>X:</b> %{x:.2f}<br><b>Y:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Adicionar linha de regressão
        # Ordenar para melhor visualização
        indices_ordenados = np.argsort(self.dados_x)
        x_ordenado = self.dados_x[indices_ordenados]
        y_pred_ordenado = self.predicoes[indices_ordenados]
        
        grafico.add_trace(go.Scatter(
            x=x_ordenado,
            y=y_pred_ordenado,
            mode='lines',
            name=f'Modelo (R²={self.r_quadrado:.4f})',
            line=dict(color='#e74c3c', width=3),
            hovertemplate='<b>Predição:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Configurar layout
        grafico.update_layout(
            title={
                'text': 'ANÁLISE DE REGRESSÃO LINEAR<br><sub>Anos de Estudo × Salário</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            xaxis=dict(
                title='Anos de Estudo',
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False
            ),
            yaxis=dict(
                title='Salário (R$)',
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=False
            ),
            plot_bgcolor='#f8f9fa',
            hovermode='closest',
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            ),
            height=600,
            margin=dict(l=80, r=50, t=120, b=80)
        )
        
        # Salvar arquivo
        grafico.write_html(salvar_como, include_plotlyjs='cdn')
        
        print(f"✅ Visualização salva em: {salvar_como}\n")
        
        return salvar_como
    
    def executar_analise_completa(self, arquivo_x='X.txt', arquivo_y='y.txt', 
                                   arquivo_saida='analise_regressao.html'):
        """Executa pipeline completo de análise"""
        if not self.carregar_dados(arquivo_x, arquivo_y):
            return False
        
        if not self.treinar_modelo():
            return False
        
        self.calcular_metricas()
        self.plotar_resultados(arquivo_saida)
        
        return True


def main():
    """Função principal"""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "SISTEMA DE ANÁLISE DE REGRESSÃO LINEAR" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝\n")
    
    # Criar instância do modelo
    modelo = ModeloRegressaoLinear()
    
    # Executar análise
    sucesso = modelo.executar_analise_completa(
        arquivo_x='X.txt',
        arquivo_y='y.txt',
        arquivo_saida='analise_regressao.html'
    )
    
    if sucesso:
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 20 + "✓ ANÁLISE CONCLUÍDA COM SUCESSO!" + " " * 17 + "║")
        print("╚" + "═" * 68 + "╝\n")
    else:
        print("\n❌ A análise não pôde ser concluída.\n")
    
    input("Pressione ENTER para finalizar...")


if __name__ == "__main__":
    main()
