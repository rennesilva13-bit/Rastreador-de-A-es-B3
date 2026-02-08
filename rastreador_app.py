"""
📊 Rastreador de Ações B3 - Enhanced Edition
Sistema inteligente de análise quantitativa de ações da B3 com foco em:
Valuation (P/L, P/VP, Graham)
Qualidade (ROE, Margem Líquida, Liquidez)
Rentabilidade (Dividend Yield, Dividendos)
Análise Técnica (RSI, Volatilidade)
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import warnings
import time
import requests

warnings.filterwarnings('ignore')

# ===== CONFIGURAÇÃO DA PÁGINA =====
st.set_page_config(
    page_title="Rastreador B3 Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== ESTILOS PERSONALIZADOS =====
st.markdown("""
<style>
    /* Paleta moderna: Azul-profundo + tons neutros */
    :root {
        --primary: #1e3a8a;
        --secondary: #0f172a;
        --accent: #3b82f6;
        --success: #10b981; 
        --warning: #f59e0b;
        --danger: #ef4444;
        --light: #f8fafc;
        --dark: #0f172a;
        --gray: #64748b;
    }
    
    /* Fundo gradiente sutil */
    .stApp {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    }
    
    /* Cards métricas */
    div[data-testid="stMetric"] {
        background: white;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease;
        border-left: 4px solid var(--accent);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.12);
    }
    
    /* Botões modernos */ 
    .stButton>button {
        background: var(--accent);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #2563eb;
        transform: scale(1.03);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    
    /* Tabela estilizada */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }
    .dataframe th {
        background-color: var(--primary) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    .dataframe td {
        background-color: white !important;
    }
    
    /* Sidebar elegante */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: white;
    }
    [data-testid="stSidebar"] .stSelectbox, 
    [data-testid="stSidebar"] .stNumberInput,
    [data-testid="stSidebar"] .stTextArea {
        background-color: rgba(30, 41, 59, 0.7);
        border-radius: 10px;
    }
    
    /* Badges informativos */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .badge-success {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
    }
    .badge-warning {
        background: linear-gradient(90deg, #f59e0b, #d97706);
        color: white;
    }
    .badge-info {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
    }
    
    /* Progress bar customizada */ 
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# ===== FUNÇÕES DE ANÁLISE TÉCNICA =====
def calcular_rsi(precos: pd.Series, periodo: int = 14) -> float:
    """Calcula o Índice de Força Relativa (RSI)"""
    try:
        if len(precos) < periodo + 1:
            return 50.0
        
        delta = precos.diff()
        ganhos = delta.where(delta > 0, 0).rolling(window=periodo).mean()
        perdas = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        
        # Evita divisão por zero
        perdas = perdas.replace(0, 0.0001)
        
        rs = ganhos / perdas
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    except Exception:
        return 50.0

def calcular_volatilidade(precos: pd.Series, janela: int = 30) -> float:
    """Calcula a volatilidade histórica (desvio padrão dos retornos)"""
    try:
        if len(precos) < janela:
            return 0.0
        
        retornos = precos.pct_change().dropna()
        volatilidade = retornos.rolling(window=janela).std().iloc[-1]
        
        return float(volatilidade * 100) if not pd.isna(volatilidade) else 0.0
    except Exception:
        return 0.0

def calcular_graham_value(lpa: float, vpa: float) -> float:
    """Calcula o Valor Intrínseco pela fórmula de Benjamin Graham"""
    try:
        if lpa > 0 and vpa > 0:
            # Fórmula: √(22.5 × LPA × VPA)
            graham = np.sqrt(22.5 * lpa * vpa)
            return float(graham) if not np.isnan(graham) and np.isfinite(graham) else 0.0
        return 0.0
    except Exception:
        return 0.0

# ===== VERIFICAÇÃO DE CONECTIVIDADE =====
def check_yfinance_connection() -> bool:
    """Verifica se há conexão com a API do Yahoo Finance"""
    try:
        response = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/PETR4.SA", timeout=5)
        return response.status_code == 200
    except Exception:
        return False

# ===== CACHE DE DADOS OTIMIZADO =====
@st.cache_data(ttl=900, show_spinner=False)
def get_yahoo_data_enhanced(ticker: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Função otimizada para coleta e processamento de dados da B3 via Yahoo Finance
    Returns:
        Tuple[Dict ou None, str ou None]: (dados_processados, mensagem_erro)
    """
    ticker_clean = ticker.strip().upper().replace('.SA', '')
    yahoo_ticker = f"{ticker_clean}.SA"

    try:
        acao = yf.Ticker(yahoo_ticker)
        
        # 1. PREÇO ATUAL (com fallback robusto)
        preco_atual = 0.0
        
        try:
            # Tenta fast_info primeiro (mais rápido)
            if hasattr(acao, 'fast_info'):
                last_price = acao.fast_info.get('last_price')
                if last_price and last_price > 0:
                    preco_atual = float(last_price)
        except Exception:
            pass
        
        # Fallback para histórico
        if preco_atual <= 0:
            try:
                hist = acao.history(period="5d")
                if not hist.empty:
                    preco_atual = float(hist['Close'].iloc[-1])
            except Exception:
                pass
        
        if preco_atual <= 0:
            return None, f"{ticker_clean}: Preço não disponível"

        # 2. FUNDAMENTOS
        try:
            info = acao.info
        except Exception as e:
            return None, f"{ticker_clean}: Erro ao obter fundamentos"

        if not info or not isinstance(info, dict):
            return None, f"{ticker_clean}: Informações vazias"

        # 3. HISTÓRICO PARA ANÁLISE TÉCNICA
        hist_precos = pd.Series([])
        try:
            hist_full = acao.history(period="3mo")
            if not hist_full.empty:
                hist_precos = hist_full['Close']
        except Exception:
            pass

        # 4. PROCESSAMENTO DE DADOS
        def safe_get(key: str, default=0.0, multiplier: float = 1.0) -> float:
            """Extrai valor com segurança"""
            try:
                val = info.get(key, default)
                if val is None:
                    return default
                result = float(val) * multiplier
                return result if np.isfinite(result) else default
            except Exception:
                return default

        # Dividend Yield (ajuste de escala)
        dy_raw = safe_get('dividendYield', 0.0)
        dividend_yield = dy_raw * 100 if 0 < dy_raw < 1 else dy_raw
        
        # LPA e VPA
        lpa = safe_get('trailingEps', 0.0)
        vpa = safe_get('bookValue', 0.0)
        
        # Valuation
        pl = safe_get('trailingPE', 0.0)
        pvp = safe_get('priceToBook', 0.0)
        
        # Correções matemáticas
        if pl == 0 and lpa > 0:
            pl = preco_atual / lpa
        if pvp == 0 and vpa > 0:
            pvp = preco_atual / vpa
        if lpa == 0 and pl > 0:
            lpa = preco_atual / pl
        if vpa == 0 and pvp > 0:
            vpa = preco_atual / pvp
        
        # Qualidade
        roe = safe_get('returnOnEquity', 0.0, 100)
        margem_liq = safe_get('profitMargins', 0.0, 100)
        margem_bruta = safe_get('grossMargins', 0.0, 100)
        
        # Liquidez e Dívida
        liquidez_corr = safe_get('currentRatio', 0.0)
        divida_patrimonio = safe_get('debtToEquity', 0.0)
        
        # Volume e Liquidez
        volume_medio = safe_get('averageVolume', 0.0)
        
        # Setor
        setor_raw = info.get('sector', 'N/A')
        setor_map = {
            'Financial Services': 'Financeiro',
            'Basic Materials': 'Materiais Básicos',
            'Energy': 'Energia',
            'Utilities': 'Utilidades Públicas',
            'Industrials': 'Industrial',
            'Consumer Cyclical': 'Consumo Cíclico',
            'Consumer Defensive': 'Consumo Não Cíclico',
            'Technology': 'Tecnologia',
            'Healthcare': 'Saúde',
            'Real Estate': 'Imobiliário',
            'Communication Services': 'Comunicação'
        }
        setor = setor_map.get(setor_raw, setor_raw if setor_raw else 'Outros')
        
        # 5. ANÁLISE TÉCNICA
        rsi = calcular_rsi(hist_precos) if not hist_precos.empty else 50.0
        volatilidade = calcular_volatilidade(hist_precos) if not hist_precos.empty else 0.0
        
        # 6. VALOR DE GRAHAM
        graham_value = calcular_graham_value(lpa, vpa)
        
        # 7. SCORE DE QUALIDADE (0-100)
        score = 0.0
        
        # ROE (30 pontos)
        if roe >= 20:
            score += 30
        elif roe >= 15:
            score += 25
        elif roe >= 10:
            score += 15
        elif roe >= 5:
            score += 5
        
        # P/L (25 pontos)
        if 0 < pl <= 10:
            score += 25
        elif 10 < pl <= 15:
            score += 20
        elif 15 < pl <= 20:
            score += 10
        
        # Margem Líquida (20 pontos)
        if margem_liq >= 20:
            score += 20
        elif margem_liq >= 10:
            score += 15
        elif margem_liq >= 5:
            score += 10
        
        # Dividend Yield (15 pontos)
        if dividend_yield >= 8:
            score += 15
        elif dividend_yield >= 5:
            score += 12
        elif dividend_yield >= 3:
            score += 8
        
        # Liquidez (10 pontos)
        if liquidez_corr >= 1.5:
            score += 10
        elif liquidez_corr >= 1.0:
            score += 5
        
        # 8. COMPILAÇÃO FINAL
        dados = {
            "Ação": ticker_clean,
            "Preço": round(preco_atual, 2),
            "DY %": round(dividend_yield, 2),
            "LPA": round(lpa, 2),
            "VPA": round(vpa, 2),
            "V. Graham": round(graham_value, 2),
            "ROE": round(roe, 2),
            "Margem_Liq": round(margem_liq, 2),
            "Margem_Bruta": round(margem_bruta, 2),
            "Liquidez_Corr": round(liquidez_corr, 2),
            "P/L": round(pl, 2),
            "P/VP": round(pvp, 2),
            "Divida/PL": round(divida_patrimonio, 2),
            "Volume_Medio": int(volume_medio),
            "RSI": round(rsi, 1),
            "Volatilidade": round(volatilidade, 2),
            "Score": round(score, 1),
            "Setor": setor,
        }
        
        # Cálculos derivados
        dados["Div_Anual"] = round(dados["Preço"] * (dados["DY %"] / 100), 2)
        dados["Margem_Seg_Graham"] = round(((graham_value - preco_atual) / preco_atual) * 100, 1) if graham_value > 0 else 0.0
        
        return dados, None

    except Exception as e:
        return None, f"{ticker_clean}: {str(e)[:50]}"

def processar_em_paralelo(tickers: List[str], max_workers: int = 5) -> Tuple[List[Dict], List[str]]:
    """Processa múltiplos tickers em paralelo"""
    dados_coletados = []
    erros = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(get_yahoo_data_enhanced, ticker): ticker 
            for ticker in tickers
        }
        
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                dados, erro = future.result(timeout=15)
                if dados:
                    dados_coletados.append(dados)
                elif erro:
                    erros.append(erro)
            except Exception as e:
                erros.append(f"{ticker}: Timeout ou erro crítico")

    return dados_coletados, erros

# ===== INICIALIZAÇÃO DO SESSION STATE =====
if 'df_resultados' not in st.session_state:
    st.session_state['df_resultados'] = None
if 'timestamp_analise' not in st.session_state:
    st.session_state['timestamp_analise'] = None

# ===== SIDEBAR - CONTROLES =====
with st.sidebar:
    st.markdown("### 🔍 Painel de Controle")
    
    # Setores B3
    setores_b3 = [
        "Todos", "Financeiro", "Materiais Básicos", "Energia", 
        "Utilidades Públicas", "Industrial", "Consumo Cíclico",
        "Consumo Não Cíclico", "Tecnologia", "Saúde", "Imobiliário", 
        "Comunicação", "Outros"
    ]
    setor_selecionado = st.selectbox("📊 Filtrar por Setor", setores_b3, index=0)

    st.markdown("<hr style='border: 1px solid #334155; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # Filtros Quantitativos
    st.markdown("**🎯 Filtros de Valuation**")
    col1, col2 = st.columns(2)
    with col1:
        min_dy = st.number_input("DY Mín (%)", min_value=0.0, max_value=30.0, value=4.0, step=0.5)
        min_roe = st.number_input("ROE Mín (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
        min_score = st.number_input("Score Mín", min_value=0, max_value=100, value=40, step=5)
    with col2:
        max_pl = st.number_input("P/L Máx", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        max_pvp = st.number_input("P/VP Máx", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
        max_divida = st.number_input("Dív/PL Máx", min_value=0.0, max_value=5.0, value=2.0, step=0.1)

    st.markdown("<hr style='border: 1px solid #334155; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # Lista de Tickers
    st.markdown("**📝 Tickers B3**")
    tickers_default = """PETR4
VALE3
ITUB4
BBDC4
B3SA3
MGLU3
WEGE3
ABEV3
BBAS3
RENT3
EGIE3
TAEE11
SAPR11
CPLE6
TRPL4
VIVT3
PRIO3
RDOR3
CSAN3
BEEF3"""
    tickers_input = st.text_area(
        "Digite um ticker por linha:", 
        value=tickers_default, 
        height=200,
        help="Insira os códigos das ações sem o .SA"
    )

    tickers_lista = [t.strip().upper() for t in tickers_input.strip().split("\n") if t.strip()]

    st.info(f"✅ {len(tickers_lista)} tickers listados")

    st.markdown("<hr style='border: 1px solid #334155; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # Configurações Avançadas
    with st.expander("⚙️ Configurações Avançadas"):
        usar_paralelo = st.checkbox("Processamento Paralelo", value=True, help="Mais rápido, mas usa mais recursos")
        max_workers = st.slider("Threads Paralelas", 3, 10, 5) if usar_paralelo else 1
        mostrar_erros = st.checkbox("Mostrar Erros Detalhados", value=False)

    st.markdown("""
    <div style='text-align: center; margin-top: 2rem; color: #94a3b8; font-size: 0.85rem;'>
        📊 Yahoo Finance API <br>
        ⚡ Cache: 15 minutos <br>
        🔒 Dados em tempo real
    </div>
    """, unsafe_allow_html=True)

# ===== CABEÇALHO PRINCIPAL =====
st.markdown("""
# 📈 Rastreador de Ações B3 Pro

Análise quantitativa inteligente • Valuation • Qualidade • Dividend Yield
""")

# ===== VERIFICAÇÃO DE CONEXÃO =====
if not check_yfinance_connection():
    st.error("❌ Conexão com a API do Yahoo Finance não está disponível. Verifique sua conexão ou tente mais tarde.")
    st.stop()

# ===== BOTÃO DE ANÁLISE =====
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analisar = st.button("🚀 Executar Análise Completa", use_container_width=True, type="primary")

# ===== PROCESSAMENTO =====
if analisar and tickers_lista:
    with st.spinner("🔍 Coletando e processando dados..."):
        inicio = time.time()
        
        # Barra de progresso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if usar_paralelo:
            # Processamento paralelo
            status_text.markdown(
                f"<div style='text-align: center; font-weight: 500; color: #3b82f6;'>"
                f"⚡ Processamento paralelo ativado ({max_workers} threads)...</div>", 
                unsafe_allow_html=True
            )
            
            dados_coletados, erros = processar_em_paralelo(tickers_lista, max_workers)
            progress_bar.progress(1.0)
            
        else:
            # Processamento sequencial
            dados_coletados = []
            erros = []
            total = len(tickers_lista)
            
            for idx, ticker in enumerate(tickers_lista):
                progress = (idx + 1) / total
                progress_bar.progress(progress)
                status_text.markdown(
                    f"<div style='text-align: center; font-weight: 500; color: #3b82f6;'>"
                    f"Analisando {ticker} ({idx+1}/{total})</div>", 
                    unsafe_allow_html=True
                )
                
                dados, erro = get_yahoo_data_enhanced(ticker)
                if dados:
                    dados_coletados.append(dados)
                elif erro:
                    erros.append(erro)
                
                time.sleep(0.2)  # Rate limiting
        
        tempo_decorrido = time.time() - inicio
        
        progress_bar.empty()
        status_text.empty()
        
        # Mostra erros se solicitado
        if erros and mostrar_erros:
            with st.expander(f"⚠️ Erros Encontrados ({len(erros)})"):
                for erro in erros:
                    st.caption(erro)

    # ===== PROCESSAMENTO DOS RESULTADOS =====
    if dados_coletados:
        df = pd.DataFrame(dados_coletados)
        
        # Aplica filtros
        df_filtrado = df[
            (df["DY %"] >= min_dy) &
            (df["ROE"] >= min_roe) &
            (df["P/L"] > 0) &
            (df["P/L"] <= max_pl) &
            (df["P/VP"] > 0) &
            (df["P/VP"] <= max_pvp) &
            (df["Divida/PL"] <= max_divida) &
            (df["Score"] >= min_score)
        ]
        
        if setor_selecionado != "Todos":
            df_filtrado = df_filtrado[df_filtrado["Setor"] == setor_selecionado]
        
        # Salva no session state
        st.session_state['df_resultados'] = df_filtrado
        st.session_state['timestamp_analise'] = datetime.now()
        
        # ===== RESUMO EXECUTIVO =====
        st.markdown("## 📊 Resumo da Análise")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Analisadas", len(df), help="Total de ações processadas")
        with col2:
            oportunidades = len(df_filtrado)
            percentual = (oportunidades / len(df) * 100) if len(df) > 0 else 0
            st.metric(
                "Oportunidades", 
                oportunidades, 
                delta=f"{percentual:.1f}%",
                help="Ações que passaram nos filtros"
            )
        with col3:
            if not df_filtrado.empty:
                st.metric("DY Médio", f"{df_filtrado['DY %'].mean():.2f}%")
        with col4:
            if not df_filtrado.empty:
                st.metric("ROE Médio", f"{df_filtrado['ROE'].mean():.2f}%")
        with col5:
            if not df_filtrado.empty:
                st.metric("Score Médio", f"{df_filtrado['Score'].mean():.1f}")
        
        st.success(f"✅ Análise concluída em {tempo_decorrido:.1f} segundos!")
        
        # ===== TABELA DE RESULTADOS =====
        if not df_filtrado.empty:
            st.markdown("---")
            st.markdown("## 🏆 Oportunidades Identificadas")
            
            # Ordena por Score
            df_display = df_filtrado.sort_values("Score", ascending=False).reset_index(drop=True)
            df_display.index = df_display.index + 1
            
            # Seleciona colunas para exibição
            colunas_exibir = [
                "Ação", "Setor", "Preço", "V. Graham", "Margem_Seg_Graham",
                "DY %", "Div_Anual", "ROE", "P/L", "P/VP", 
                "Margem_Liq", "Liquidez_Corr", "RSI", "Score"
            ]
            
            # Formata a tabela
            st.dataframe(
                df_display[colunas_exibir],
                column_config={
                    "Preço": st.column_config.NumberColumn(format="R$ %.2f"),
                    "V. Graham": st.column_config.NumberColumn("V. Graham", format="R$ %.2f"),
                    "Margem_Seg_Graham": st.column_config.NumberColumn("MS Graham %", format="%.1f%%"),
                    "DY %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Div_Anual": st.column_config.NumberColumn("Div. Anual", format="R$ %.2f"),
                    "ROE": st.column_config.NumberColumn(format="%.2f%%"),
                    "P/L": st.column_config.NumberColumn(format="%.1f"),
                    "P/VP": st.column_config.NumberColumn(format="%.2f"),
                    "Margem_Liq": st.column_config.NumberColumn("Mrg Líq %", format="%.2f%%"),
                    "Liquidez_Corr": st.column_config.NumberColumn("Liq. Corr", format="%.2f"),
                    "RSI": st.column_config.NumberColumn(format="%.1f"),
                    "Score": st.column_config.ProgressColumn("Score", format="%.0f", min_value=0, max_value=100),
                },
                use_container_width=True,
                height=400
            )
            
            # ===== VISUALIZAÇÕES =====
            st.markdown("---")
            st.markdown("## 📈 Análises Visuais")
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 ROE vs DY", 
                "💰 Top 10 DY", 
                "🏅 Score de Qualidade",
                "📊 Análise Setorial"
            ])
            
            with tab1:
                # Gráfico de dispersão ROE x DY
                fig_scatter = px.scatter(
                    df_filtrado,
                    x="DY %",
                    y="ROE",
                    size="Score",
                    color="Setor",
                    hover_name="Ação",
                    hover_data={
                        "P/L": ":.1f",
                        "P/VP": ":.2f",
                        "Preço": ":,.2f",
                        "Score": ":.1f"
                    },
                    title="Qualidade (ROE) vs Rentabilidade (DY) - Tamanho = Score",
                    labels={"ROE": "ROE (%)", "DY %": "Dividend Yield (%)"},
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    height=500
                )
                fig_scatter.update_traces(marker=dict(line=dict(width=1.5, color='white')))
                fig_scatter.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12),
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with tab2:
                # Top 10 Dividend Yield
                top_dy = df_filtrado.nlargest(10, "DY %").sort_values("DY %", ascending=True)
                
                fig_bar = go.Figure(go.Bar(
                    x=top_dy["DY %"],
                    y=top_dy["Ação"],
                    orientation='h',
                    marker=dict(
                        color=top_dy["DY %"],
                        colorscale='Blues',
                        line=dict(color='white', width=1.5),
                        colorbar=dict(title="DY %")
                    ),
                    text=[f"{val:.2f}%" for val in top_dy["DY %"]],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>DY: %{x:.2f}%<extra></extra>'
                ))
                
                fig_bar.update_layout(
                    title="Top 10 Ações por Dividend Yield",
                    xaxis_title="Dividend Yield (%)",
                    yaxis_title="",
                    plot_bgcolor='white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=450,
                    font=dict(size=12)
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with tab3:
                # Top 10 Score de Qualidade
                top_score = df_filtrado.nlargest(10, "Score").sort_values("Score", ascending=True)
                
                fig_score = go.Figure(go.Bar(
                    x=top_score["Score"],
                    y=top_score["Ação"],
                    orientation='h',
                    marker=dict(
                        color=top_score["Score"],
                        colorscale='Viridis',
                        line=dict(color='white', width=1.5),
                        colorbar=dict(title="Score")
                    ),
                    text=[f"{val:.1f}" for val in top_score["Score"]],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>'
                ))
                
                fig_score.update_layout(
                    title="Top 10 Ações por Score de Qualidade (0-100)",
                    xaxis_title="Score",
                    yaxis_title="",
                    plot_bgcolor='white',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=450,
                    font=dict(size=12)
                )
                st.plotly_chart(fig_score, use_container_width=True)
            
            with tab4:
                # Análise por Setor
                if len(df_filtrado['Setor'].unique()) > 1:
                    setor_stats = df_filtrado.groupby('Setor').agg({
                        'Ação': 'count',
                        'DY %': 'mean',
                        'ROE': 'mean',
                        'Score': 'mean'
                    }).reset_index()
                    setor_stats.columns = ['Setor', 'Quantidade', 'DY Médio', 'ROE Médio', 'Score Médio']
                    setor_stats = setor_stats.sort_values('Score Médio', ascending=False)
                    
                    fig_setor = go.Figure()
                    
                    fig_setor.add_trace(go.Bar(
                        name='Score Médio',
                        x=setor_stats['Setor'],
                        y=setor_stats['Score Médio'],
                        marker_color='#3b82f6',
                        text=[f"{val:.1f}" for val in setor_stats['Score Médio']],
                        textposition='outside'
                    ))
                    
                    fig_setor.update_layout(
                        title='Score Médio por Setor',
                        xaxis_title='Setor',
                        yaxis_title='Score Médio',
                        plot_bgcolor='white',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=450,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_setor, use_container_width=True)
                    
                    # Tabela de estatísticas setoriais
                    st.markdown("#### Estatísticas por Setor")
                    st.dataframe(
                        setor_stats,
                        column_config={
                            "DY Médio": st.column_config.NumberColumn(format="%.2f%%"),
                            "ROE Médio": st.column_config.NumberColumn(format="%.2f%%"),
                            "Score Médio": st.column_config.NumberColumn(format="%.1f"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("Não há dados suficientes para análise setorial com os filtros atuais.")
            
            # ===== DOWNLOAD =====
            st.markdown("---")
            st.markdown("## 💾 Exportar Resultados")
            
            col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
            with col_d2:
                csv = df_filtrado.to_csv(index=False, encoding='utf-8-sig')
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                st.download_button(
                    label="📥 Baixar Análise Completa (CSV)",
                    data=csv,
                    file_name=f"rastreador_b3_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )
        
        else:
            st.warning("⚠️ Nenhuma ação atende aos critérios selecionados. Tente ajustar os filtros na barra lateral.")
            
            # Mostra estatísticas básicas mesmo sem filtros
            st.markdown("#### Estatísticas Gerais (sem filtros)")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            
            with col_s1:
                st.metric("DY Médio Geral", f"{df['DY %'].mean():.2f}%")
            with col_s2:
                st.metric("ROE Médio Geral", f"{df['ROE'].mean():.2f}%")
            with col_s3:
                st.metric("P/L Mediano", f"{df[df['P/L'] > 0]['P/L'].median():.1f}")
            with col_s4:
                st.metric("Score Médio", f"{df['Score'].mean():.1f}")

    else:
        st.error("❌ Nenhum dado foi coletado. Verifique:")
        st.markdown("""
        - Os tickers estão corretos e sem espaços extras?
        - A API do Yahoo Finance está acessível (teste com PETR4, VALE3)?
        - Há limite de requisições sendo atingido?
        """)
        
        if 'erros' in locals() and erros and mostrar_erros:
            with st.expander("Ver detalhes dos erros"):
                for erro in erros:
                    st.caption(erro)

# ===== RODAPÉ =====
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #64748b; font-size: 0.9rem;'>
💡 Aviso Legal: Este aplicativo utiliza dados do Yahoo Finance e é destinado 
exclusivamente para fins educacionais e de análise pessoal.<br>
Não constitui recomendação de investimento. 
Sempre faça sua própria análise e consulte um profissional certificado antes de investir.<br><br>
🚀 Desenvolvido com Streamlit • Última atualização: {datetime.now().strftime("%d/%m/%Y às %H:%M")}
</div>
""", unsafe_allow_html=True)
