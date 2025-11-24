import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import calendar


class ConsultadorDolarPTAX:
    """Classe para consultar e visualizar cotações do dólar"""
    
    def __init__(self, periodo: str):
        self.periodo = periodo
        self.url_api = "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
        self.dados_processados = None
        
    def _converter_periodo(self):
        """Converte string MMYYYY para objeto datetime"""
        try:
            return datetime.strptime(self.periodo, "%m%Y")
        except ValueError:
            raise ValueError("Período deve estar no formato MMYYYY")
    
    def _montar_url_consulta(self, dt_objeto):
        """Monta URL de consulta da API"""
        primeiro_dia = 1
        ultimo_dia = calendar.monthrange(dt_objeto.year, dt_objeto.month)[1]
        
        dt_inicio = f"{dt_objeto.month:02d}-{primeiro_dia:02d}-{dt_objeto.year}"
        dt_fim = f"{dt_objeto.month:02d}-{ultimo_dia:02d}-{dt_objeto.year}"
        
        endpoint = (
            f"{self.url_api}CotacaoDolarPeriodo(dataInicial=@dataInicial,"
            f"dataFinalCotacao=@dataFinalCotacao)?"
            f"@dataInicial='{dt_inicio}'&@dataFinalCotacao='{dt_fim}'&$format=json"
        )
        
        return endpoint, primeiro_dia, ultimo_dia
    
    def buscar_cotacoes(self):
        """Realiza a busca das cotações na API do BCB"""
        print(f"🔍 Buscando cotações para {self.periodo}...")
        
        dt_referencia = self._converter_periodo()
        url_requisicao, dia_ini, dia_fim = self._montar_url_consulta(dt_referencia)
        
        try:
            resposta = requests.get(url_requisicao, timeout=10)
            resposta.raise_for_status()
            dados_json = resposta.json()
            
            if not dados_json.get('value'):
                print(f"⚠️  Nenhuma cotação encontrada para {self.periodo}")
                return None
            
            # Processar dados da API
            dataframe = pd.DataFrame(dados_json['value'])
            dataframe['dataHoraCotacao'] = pd.to_datetime(dataframe['dataHoraCotacao'])
            dataframe['dia'] = dataframe['dataHoraCotacao'].dt.date
            dataframe = dataframe[['dia', 'cotacaoVenda']].sort_values('dia')
            
            print(f"✅ {len(dataframe)} registros obtidos")
            
            # Preencher todos os dias do mês
            dataframe_completo = self._preencher_dias_faltantes(
                dataframe, dt_referencia, dia_ini, dia_fim
            )
            
            self.dados_processados = dataframe_completo
            return dataframe_completo
            
        except requests.exceptions.Timeout:
            print("❌ Tempo esgotado ao conectar com a API")
            return None
        except requests.exceptions.RequestException as erro:
            print(f"❌ Erro na requisição: {erro}")
            return None
    
    def _preencher_dias_faltantes(self, df_original, dt_ref, dia_inicial, dia_final):
        """Preenche dias sem cotação com o último valor disponível"""
        data_inicio = datetime(dt_ref.year, dt_ref.month, dia_inicial).date()
        data_fim = datetime(dt_ref.year, dt_ref.month, dia_final).date()
        
        # Criar range com todos os dias
        todos_dias = pd.date_range(start=data_inicio, end=data_fim, freq='D').date
        df_todos_dias = pd.DataFrame({'dia': todos_dias})
        
        # Fazer merge e preencher valores faltantes
        df_final = df_todos_dias.merge(df_original, on='dia', how='left')
        df_final['cotacaoVenda'].fillna(method='ffill', inplace=True)
        df_final.rename(columns={'cotacaoVenda': 'valor_cotacao'}, inplace=True)
        
        dias_preenchidos = df_final['valor_cotacao'].notna().sum() - len(df_original)
        if dias_preenchidos > 0:
            print(f"📝 {dias_preenchidos} dias foram preenchidos com valores anteriores")
        
        return df_final
    
    def gerar_grafico(self):
        """Gera gráfico interativo das cotações"""
        if self.dados_processados is None:
            print("❌ Não há dados para gerar o gráfico")
            return None
        
        df = self.dados_processados
        dt_ref = self._converter_periodo()
        nome_mes = dt_ref.strftime("%B").upper()
        
        # Criar gráfico com plotly graph_objects
        figura = go.Figure()
        
        figura.add_trace(go.Scatter(
            x=df['dia'],
            y=df['valor_cotacao'],
            mode='lines+markers',
            name='Cotação',
            line=dict(color='#00a86b', width=2.5),
            marker=dict(size=5, color='#006b4e'),
            hovertemplate='<b>Data:</b> %{x}<br><b>Cotação:</b> R$ %{y:.4f}<extra></extra>'
        ))
        
        figura.update_layout(
            title={
                'text': f'COTAÇÃO PTAX DO DÓLAR - {nome_mes}/{dt_ref.year}',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Helvetica, sans-serif', 'color': '#2c3e50'}
            },
            xaxis=dict(
                title='Período',
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='Valor em Reais (R$)',
                showgrid=True,
                gridcolor='lightgray'
            ),
            plot_bgcolor='white',
            hovermode='x unified',
            height=550,
            margin=dict(l=80, r=50, t=100, b=80)
        )
        
        # Salvar gráfico
        nome_arquivo = f"grafico_dolar_ptax_{self.periodo}.html"
        figura.write_html(nome_arquivo, include_plotlyjs='cdn')
        
        print(f"\n✅ Gráfico salvo: {nome_arquivo}")
        print(f"💡 Abra o arquivo para visualização interativa")
        
        return nome_arquivo


def executar_consulta(mes_ano: str):
    """Função principal para executar a consulta"""
    try:
        if len(mes_ano) != 6 or not mes_ano.isdigit():
            print("\n❌ Formato incorreto! Utilize MMYYYY (exemplo: 072016)")
            return
        
        print("\n" + "="*75)
        print("SISTEMA DE CONSULTA - COTAÇÃO DÓLAR PTAX (BCB)")
        print("="*75 + "\n")
        
        consultor = ConsultadorDolarPTAX(mes_ano)
        
        # Buscar cotações
        dados = consultor.buscar_cotacoes()
        
        if dados is not None:
            # Gerar gráfico
            consultor.gerar_grafico()
            
            # Exibir estatísticas
            print("\n📊 ESTATÍSTICAS DO PERÍODO:")
            print(f"   • Valor mínimo: R$ {dados['valor_cotacao'].min():.4f}")
            print(f"   • Valor máximo: R$ {dados['valor_cotacao'].max():.4f}")
            print(f"   • Valor médio: R$ {dados['valor_cotacao'].mean():.4f}")
        
        print("\n" + "="*75)
        
    except Exception as erro:
        print(f"\n❌ Erro durante execução: {erro}")


if __name__ == "__main__":
    periodo_consulta = "022019"
    executar_consulta(periodo_consulta)
    
    print("\n")
    input("Pressione ENTER para sair...")
