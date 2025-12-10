"""
Monte Carlo Simulation with Real-time Analytics
"""

import numpy as np
import scipy.stats as stats
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="Quantum Option Pricing Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM STYLING ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5D6D7E;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .info-box {
        background-color: #EBF5FB;
        border-left: 5px solid #2E86C1;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .stMetric {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ========== ENUMERATIONS ==========
class ContractType(Enum):
    """Financial contract classification"""
    CALL = 1.0
    PUT = -1.0
    STRADDLE = 2.0 

class OptionCategory(Enum):
    """Option classification categories"""
    EUROPEAN = "European"
    BINARY = "Binary"
    BARRIER = "Barrier"

# ========== CORE PRICING MODELS ==========
class QuantitativePricingEngine:
    """Pricing engine with multiple models"""
    
    @staticmethod
    def calculate_standard_option_value(contract_type, spot_price, strike, volatility, time_horizon, interest_rate):
        """Compute Black-Scholes-Merton valuation"""
        # FIX: Handle both scalar and array inputs
        if np.isscalar(strike):
            strike_array = np.array([strike]).reshape(1, 1)
        else:
            strike_array = np.array(strike).reshape(-1, 1)
        
        d1_numerator = np.log(spot_price / strike_array) + (interest_rate + 0.5 * volatility**2) * time_horizon
        d1 = d1_numerator / (volatility * np.sqrt(time_horizon))
        d2 = d1 - volatility * np.sqrt(time_horizon)
        
        if contract_type == ContractType.CALL:
            valuation = (stats.norm.cdf(d1) * spot_price) - \
                       (strike_array * np.exp(-interest_rate * time_horizon) * stats.norm.cdf(d2))
        elif contract_type == ContractType.PUT:
            valuation = (strike_array * np.exp(-interest_rate * time_horizon) * stats.norm.cdf(-d2)) - \
                       (stats.norm.cdf(-d1) * spot_price)
        elif contract_type == ContractType.STRADDLE: 
            call_price = (stats.norm.cdf(d1) * spot_price) - \
                        (strike_array * np.exp(-interest_rate * time_horizon) * stats.norm.cdf(d2))
            put_price = (strike_array * np.exp(-interest_rate * time_horizon) * stats.norm.cdf(-d2)) - \
                       (stats.norm.cdf(-d1) * spot_price)
            valuation = call_price + put_price
        
        return valuation if valuation.size > 1 else float(valuation[0])
    
    @staticmethod
    def calculate_binary_option_value(contract_type, spot_price, strike, volatility, time_horizon, interest_rate):
        """Compute cash-or-nothing digital option valuation"""
        # FIX: Handle both scalar and array inputs
        if np.isscalar(strike):
            strike_array = np.array([strike]).reshape(1, 1)
        else:
            strike_array = np.array(strike).reshape(-1, 1)
        
        d1 = (np.log(spot_price / strike_array) + (interest_rate + 0.5 * volatility**2) * time_horizon) / \
             (volatility * np.sqrt(time_horizon))
        d2 = d1 - volatility * np.sqrt(time_horizon)
        
        if contract_type == ContractType.CALL:
            valuation = strike_array * np.exp(-interest_rate * time_horizon) * stats.norm.cdf(d2)
        elif contract_type == ContractType.PUT:
            valuation = strike_array * np.exp(-interest_rate * time_horizon) * (1.0 - stats.norm.cdf(d2))
        
        return valuation if valuation.size > 1 else float(valuation[0])

# ========== PATH GENERATORS ==========
class StochasticPathGenerator:
    """Path simulation with multiple schemes"""
    
    @staticmethod
    @st.cache_data(show_spinner="Generating Euler paths...")
    def simulate_euler_paths(num_simulations, time_intervals, maturity_time, interest, vol, initial_price):
        """Generate paths using Euler-Maruyama discretization"""
        random_shocks = np.random.normal(0.0, 1.0, [num_simulations, time_intervals])
        brownian_motion = np.zeros([num_simulations, time_intervals + 1])
        price_paths = np.zeros([num_simulations, time_intervals + 1])
        price_paths[:, 0] = initial_price
        
        time_grid = np.zeros([time_intervals + 1])
        delta_t = maturity_time / float(time_intervals)
        
        for step in range(time_intervals):
            # Antithetic variates for variance reduction
            if num_simulations > 1:
                random_shocks[:, step] = (random_shocks[:, step] - np.mean(random_shocks[:, step])) / \
                                         np.std(random_shocks[:, step])
            
            # Generate antithetic paths 
            brownian_motion[:, step + 1] = brownian_motion[:, step] + np.sqrt(delta_t) * random_shocks[:, step]
            
            price_paths[:, step + 1] = price_paths[:, step] + \
                                      interest * price_paths[:, step] * delta_t + \
                                      vol * price_paths[:, step] * (brownian_motion[:, step + 1] - brownian_motion[:, step])
            
            time_grid[step + 1] = time_grid[step] + delta_t
        
        return {"timeline": time_grid, "price_trajectories": price_paths}
    
    @staticmethod
    @st.cache_data(show_spinner="Generating Milstein paths...")
    def simulate_milstein_paths(num_simulations, time_intervals, maturity_time, interest, vol, initial_price):
        """Generate paths using Milstein higher-order discretization"""
        random_shocks = np.random.normal(0.0, 1.0, [num_simulations, time_intervals])
        brownian_motion = np.zeros([num_simulations, time_intervals + 1])
        price_paths = np.zeros([num_simulations, time_intervals + 1])
        price_paths[:, 0] = initial_price
        
        time_grid = np.zeros([time_intervals + 1])
        delta_t = maturity_time / float(time_intervals)
        
        for step in range(time_intervals):
            if num_simulations > 1:
                random_shocks[:, step] = (random_shocks[:, step] - np.mean(random_shocks[:, step])) / \
                                         np.std(random_shocks[:, step])
            
            brownian_motion[:, step + 1] = brownian_motion[:, step] + np.sqrt(delta_t) * random_shocks[:, step]
            
            brownian_increment = brownian_motion[:, step + 1] - brownian_motion[:, step]
            
            # Milstein scheme with correction term
            price_paths[:, step + 1] = price_paths[:, step] + \
                                      interest * price_paths[:, step] * delta_t + \
                                      vol * price_paths[:, step] * brownian_increment + \
                                      0.5 * vol**2.0 * price_paths[:, step] * (brownian_increment**2 - delta_t)
            
            time_grid[step + 1] = time_grid[step] + delta_t
        
        return {"timeline": time_grid, "price_trajectories": price_paths}

# ========== MONTE CARLO VALUATORS ==========
class MonteCarloValuator:
    """Monte Carlo option pricing"""
    
    @staticmethod
    def price_european_from_simulation(contract_type, terminal_prices, strike, maturity, interest):
        """Calculate European option value from simulated terminal prices"""
        if contract_type == ContractType.CALL:
            return np.exp(-interest * maturity) * np.mean(np.maximum(terminal_prices - strike, 0.0))
        elif contract_type == ContractType.PUT:
            return np.exp(-interest * maturity) * np.mean(np.maximum(strike - terminal_prices, 0.0))
        elif contract_type == ContractType.STRADDLE:  
            call_value = np.exp(-interest * maturity) * np.mean(np.maximum(terminal_prices - strike, 0.0))
            put_value = np.exp(-interest * maturity) * np.mean(np.maximum(strike - terminal_prices, 0.0))
            return call_value + put_value
    
    @staticmethod
    def price_binary_from_simulation(contract_type, terminal_prices, strike, maturity, interest):
        """Calculate binary option value from simulated terminal prices"""
        if contract_type == ContractType.CALL:
            return np.exp(-interest * maturity) * strike * np.mean((terminal_prices > strike))
        elif contract_type == ContractType.PUT:
            return np.exp(-interest * maturity) * strike * np.mean((terminal_prices <= strike))

# ========== VISUALIZATION ==========
class FinancialVisualizer:
    """visualization engine with Plotly"""
    
    @staticmethod
    def create_interactive_path_comparison(euler_paths, milstein_paths, num_paths_display=20):
        """Create interactive comparison of simulated paths"""
        fig = go.Figure()
        
        # Euler paths
        for i in range(min(num_paths_display, euler_paths["price_trajectories"].shape[0])):
            fig.add_trace(go.Scatter(
                x=euler_paths["timeline"],
                y=euler_paths["price_trajectories"][i, :],
                mode='lines',
                name=f'Euler Path {i+1}' if i < 3 else None,
                line=dict(width=1, color='rgba(100, 149, 237, 0.6)'),
                showlegend=True if i < 3 else False,
                hoverinfo='skip'
            ))
        
        # Milstein paths
        for i in range(min(num_paths_display, milstein_paths["price_trajectories"].shape[0])):
            fig.add_trace(go.Scatter(
                x=milstein_paths["timeline"],
                y=milstein_paths["price_trajectories"][i, :],
                mode='lines',
                name=f'Milstein Path {i+1}' if i < 3 else None,
                line=dict(width=1, color='rgba(220, 20, 60, 0.6)'),
                showlegend=True if i < 3 else False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title="Price Trajectory Simulations",
            xaxis_title="Time (Years)",
            yaxis_title="Asset Price",
            hovermode='x unified',
            template="plotly_white",
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_clean_path_visualizations(euler_paths, milstein_paths, num_paths_display=10):
        """Create cleaner, separated path visualizations"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Euler Scheme - First 5 Paths', 
                           'Milstein Scheme - First 5 Paths',
                           'Euler Scheme - Paths 6-10', 
                           'Milstein Scheme - Paths 6-10'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Euler paths 1-5
        for i in range(min(5, euler_paths["price_trajectories"].shape[0])):
            fig.add_trace(
                go.Scatter(
                    x=euler_paths["timeline"],
                    y=euler_paths["price_trajectories"][i, :],
                    mode='lines',
                    name=f'Euler {i+1}',
                    line=dict(width=1.5, color=px.colors.qualitative.Set1[i]),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Milstein paths 1-5
        for i in range(min(5, milstein_paths["price_trajectories"].shape[0])):
            fig.add_trace(
                go.Scatter(
                    x=milstein_paths["timeline"],
                    y=milstein_paths["price_trajectories"][i, :],
                    mode='lines',
                    name=f'Milstein {i+1}',
                    line=dict(width=1.5, color=px.colors.qualitative.Set2[i]),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Euler paths 6-10
        for i in range(5, min(10, euler_paths["price_trajectories"].shape[0])):
            fig.add_trace(
                go.Scatter(
                    x=euler_paths["timeline"],
                    y=euler_paths["price_trajectories"][i, :],
                    mode='lines',
                    name=f'Euler {i+1}',
                    line=dict(width=1.5, color=px.colors.qualitative.Set1[i-5]),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Milstein paths 6-10
        for i in range(5, min(10, milstein_paths["price_trajectories"].shape[0])):
            fig.add_trace(
                go.Scatter(
                    x=milstein_paths["timeline"],
                    y=milstein_paths["price_trajectories"][i, :],
                    mode='lines',
                    name=f'Milstein {i+1}',
                    line=dict(width=1.5, color=px.colors.qualitative.Set2[i-5]),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            template="plotly_white",
            title_text="Path Simulations - Clean View",
            title_font=dict(size=20)
        )
        
        # Update axes labels
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Time (Years)", row=i, col=j)
                fig.update_yaxes(title_text="Asset Price", row=i, col=j)
        
        return fig
    
    @staticmethod
    def create_statistical_path_summary(euler_paths, milstein_paths):
        """Create statistical summary of paths"""
        euler_terminal = euler_paths["price_trajectories"][:, -1]
        milstein_terminal = milstein_paths["price_trajectories"][:, -1]
        
        fig = go.Figure()
        
        # Euler distribution
        fig.add_trace(go.Histogram(
            x=euler_terminal,
            name='Euler Distribution',
            marker_color='rgba(100, 149, 237, 0.7)',
            opacity=0.7,
            nbinsx=30
        ))
        
        # Milstein distribution
        fig.add_trace(go.Histogram(
            x=milstein_terminal,
            name='Milstein Distribution',
            marker_color='rgba(220, 20, 60, 0.7)',
            opacity=0.7,
            nbinsx=30
        ))
        
        fig.update_layout(
            title="Terminal Price Distribution Comparison",
            xaxis_title="Price at Maturity",
            yaxis_title="Frequency",
            barmode='overlay',
            template="plotly_white",
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        return fig
    
    @staticmethod
    def create_convergence_dashboard(path_counts, euler_prices, milstein_prices, analytical_price):
        """Create convergence analysis dashboard"""
        fig = go.Figure()
        
        # Euler convergence
        fig.add_trace(go.Scatter(
            x=path_counts,
            y=euler_prices,
            mode='lines+markers',
            name='Euler Scheme',
            line=dict(color='royalblue', width=3),
            marker=dict(size=8)
        ))
        
        # Milstein convergence
        fig.add_trace(go.Scatter(
            x=path_counts,
            y=milstein_prices,
            mode='lines+markers',
            name='Milstein Scheme',
            line=dict(color='crimson', width=3, dash='dash'),
            marker=dict(size=8, symbol='square')
        ))
        
        # Analytical price reference
        fig.add_hline(
            y=analytical_price,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Analytical Price: {analytical_price:.4f}",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            title="Monte Carlo Convergence Analysis",
            xaxis_title="Number of Simulation Paths",
            yaxis_title="Option Value",
            hovermode='x unified',
            template="plotly_white",
            height=500,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_error_comparison(path_counts, euler_errors, milstein_errors):
        """Create error comparison visualization"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=path_counts,
            y=euler_errors,
            mode='lines+markers',
            name='Euler Error',
            line=dict(color='orange', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=path_counts,
            y=milstein_errors,
            mode='lines+markers',
            name='Milstein Error',
            line=dict(color='purple', width=2, dash='dot'),
            marker=dict(size=6, symbol='diamond')
        ))
        
        # Add theoretical convergence line
        theoretical_convergence = 1 / np.sqrt(path_counts)
        theoretical_convergence = theoretical_convergence / theoretical_convergence[0] * max(euler_errors[0], milstein_errors[0])
        
        fig.add_trace(go.Scatter(
            x=path_counts,
            y=theoretical_convergence,
            mode='lines',
            name='Theoretical 1/√N',
            line=dict(color='gray', width=2, dash='dash'),
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Numerical Error Comparison",
            xaxis_title="Number of Simulation Paths",
            yaxis_title="Absolute Error",
            hovermode='x unified',
            template="plotly_white",
            height=500,
            yaxis_type="log",
            xaxis_type="log",
            showlegend=True
        )
        
        return fig

# ========== RISK ANALYTICS ==========
class RiskAnalyzer:
    """Risk analytics and Greeks calculation"""
    
    @staticmethod
    def calculate_greeks(contract_type, spot_price, strike, volatility, time_horizon, interest_rate):
        """Calculate option Greeks (Delta, Gamma, Vega, Theta, Rho)"""
        # FIX: Handle both scalar and array inputs
        if np.isscalar(strike):
            strike_array = np.array([strike]).reshape(1, 1)
        else:
            strike_array = np.array(strike).reshape(-1, 1)
        
        d1 = (np.log(spot_price / strike_array) + (interest_rate + 0.5 * volatility**2) * time_horizon) / \
             (volatility * np.sqrt(time_horizon))
        d2 = d1 - volatility * np.sqrt(time_horizon)
        
        # Delta
        if contract_type == ContractType.CALL:
            delta = stats.norm.cdf(d1)
        else:  # PUT
            delta = stats.norm.cdf(d1) - 1
        
        # Gamma (same for call and put)
        gamma = stats.norm.pdf(d1) / (spot_price * volatility * np.sqrt(time_horizon))
        
        # Vega (same for call and put)
        vega = spot_price * stats.norm.pdf(d1) * np.sqrt(time_horizon)
        
        # Theta
        if contract_type == ContractType.CALL:
            theta = -spot_price * stats.norm.pdf(d1) * volatility / (2 * np.sqrt(time_horizon)) - \
                    interest_rate * strike_array * np.exp(-interest_rate * time_horizon) * stats.norm.cdf(d2)
        else:  # PUT
            theta = -spot_price * stats.norm.pdf(d1) * volatility / (2 * np.sqrt(time_horizon)) + \
                    interest_rate * strike_array * np.exp(-interest_rate * time_horizon) * stats.norm.cdf(-d2)
        
        # Rho
        if contract_type == ContractType.CALL:
            rho = strike_array * time_horizon * np.exp(-interest_rate * time_horizon) * stats.norm.cdf(d2)
        else:  # PUT
            rho = -strike_array * time_horizon * np.exp(-interest_rate * time_horizon) * stats.norm.cdf(-d2)
        
        return {
            'Delta': float(delta[0] if hasattr(delta, '__len__') else delta),
            'Gamma': float(gamma[0] if hasattr(gamma, '__len__') else gamma),
            'Vega': float(vega[0] if hasattr(vega, '__len__') else vega),
            'Theta': float(theta[0] if hasattr(theta, '__len__') else theta),
            'Rho': float(rho[0] if hasattr(rho, '__len__') else rho)
        }

# ========== MAIN APPLICATION ==========
def execute_quantitative_analysis():
    """Main application orchestrator"""
    
    # ========== HEADER SECTION ==========
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">⚡ Quantum Option Pricing Platform</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Monte Carlo Simulation & Risk Analytics</p>', unsafe_allow_html=True)
    
    # ========== SIDEBAR CONTROLS ==========
    with st.sidebar:
        st.markdown("### 🎮 Simulation Controls")
        
        # Market Parameters
        st.markdown("#### Market Parameters")
        spot_price = st.number_input("Initial Asset Price ($S_0$)", value=100.0, min_value=1.0, max_value=1000.0, step=10.0)
        
        strike_input = st.text_input("Strike Price(s) - Comma Separated", value="100.0")
        try:
            strike_values = [float(k.strip()) for k in strike_input.split(",")]
            strike_prices = np.array(strike_values)
        except:
            st.error("Invalid strike price format")
            strike_prices = np.array([100.0])
        
        maturity_time = st.slider("Time to Maturity (Years)", 0.1, 5.0, 1.0, 0.1)
        risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.1) / 100
        volatility = st.slider("Annual Volatility (%)", 1.0, 100.0, 20.0, 1.0) / 100
        
        # Simulation Parameters
        st.markdown("#### Simulation Parameters")
        time_intervals = st.selectbox("Time Discretization Steps", [100, 500, 1000, 2000], index=2)
        
        # Multiple path configurations with presets
        simulation_presets = st.selectbox("Simulation Presets", 
                                         ["Quick Test", "Standard", "High Precision", "Custom"])
        
        if simulation_presets == "Quick Test":
            path_counts = [100, 500]
        elif simulation_presets == "Standard":
            path_counts = [100, 1000, 5000]
        elif simulation_presets == "High Precision":
            path_counts = [1000, 5000, 10000, 25000]
        else:  # Custom
            custom_paths = st.text_input("Custom Path Counts (comma separated)", "100,1000,5000,10000")
            path_counts = [int(p.strip()) for p in custom_paths.split(",")]
        
        # Contract Selection
        st.markdown("#### Contract Configuration")
        contract_type = st.radio("Option Type", 
                                [ContractType.CALL, ContractType.PUT, ContractType.STRADDLE])
        
        option_category = st.radio("Option Style", 
                                  [OptionCategory.EUROPEAN, OptionCategory.BINARY])
        
        # Visualization Settings
        st.markdown("#### Visualization")
        paths_to_display = st.slider("Paths to Visualize", 5, 100, 25)
        
        # Random seed for reproducibility
        use_fixed_seed = st.checkbox("Use Fixed Random Seed", value=True)
        if use_fixed_seed:
            np.random.seed(42)
    
    # ========== MAIN DASHBOARD ==========
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Pricing Analysis", "📈 Convergence", "⚖️ Risk Metrics", "🔍 Path Visualizations"])
    
    if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
        
        with st.spinner("Performing quantitative analysis..."):
            
            # ========== ANALYTICAL PRICING ==========
            with tab1:
                st.markdown("### Analytical Valuation")
                
                # Calculate analytical prices - FIX: Use strike_prices[0] as scalar
                try:
                    selected_strike = strike_prices[0] if len(strike_prices) > 0 else 100.0
                    
                    if option_category == OptionCategory.EUROPEAN:
                        analytical_price = QuantitativePricingEngine.calculate_standard_option_value(
                            contract_type, spot_price, selected_strike, volatility, maturity_time, risk_free_rate
                        )
                    else:  # Binary
                        analytical_price = QuantitativePricingEngine.calculate_binary_option_value(
                            contract_type, spot_price, selected_strike, volatility, maturity_time, risk_free_rate
                        )
                    
                    # Ensure analytical_price is a scalar
                    if hasattr(analytical_price, '__len__'):
                        analytical_price = float(analytical_price[0] if analytical_price.size > 0 else analytical_price)
                    else:
                        analytical_price = float(analytical_price)
                        
                except Exception as e:
                    st.error(f"Error calculating analytical price: {str(e)}")
                    return
                
                # Display metrics in clean cards
                st.markdown("#### Key Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-title">Analytical Price</div>
                        <div class="metric-value">${analytical_price:.4f}</div>
                        <div style="font-size: 0.8rem; opacity: 0.9;">Black-Scholes Value</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    moneyness = "ATM" if abs(spot_price - selected_strike)/spot_price < 0.05 else \
                               "ITM" if (contract_type == ContractType.CALL and spot_price > selected_strike) or \
                                       (contract_type == ContractType.PUT and spot_price < selected_strike) else "OTM"
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-title">Moneyness</div>
                        <div class="metric-value">{moneyness}</div>
                        <div style="font-size: 0.8rem; opacity: 0.9;">Option Position</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col3:
                    time_value = analytical_price - max(spot_price - selected_strike, 0) if contract_type == ContractType.CALL else \
                                analytical_price - max(selected_strike - spot_price, 0)
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-title">Time Value</div>
                        <div class="metric-value">${float(time_value):.4f}</div>
                        <div style="font-size: 0.8rem; opacity: 0.9;">Extrinsic Value</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col4:
                    intrinsic_value = max(spot_price - selected_strike, 0) if contract_type == ContractType.CALL else \
                                     max(selected_strike - spot_price, 0)
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-title">Intrinsic Value</div>
                        <div class="metric-value">${float(intrinsic_value):.4f}</div>
                        <div style="font-size: 0.8rem; opacity: 0.9;">Immediate Exercise</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Monte Carlo Results
                st.markdown("### Monte Carlo Simulation Results")
                
                euler_results, milstein_results = [], []
                all_euler_paths, all_milstein_paths = None, None
                
                progress_bar = st.progress(0)
                for idx, path_count in enumerate(path_counts):
                    try:
                        # Generate paths
                        euler_paths = StochasticPathGenerator.simulate_euler_paths(
                            path_count, time_intervals, maturity_time, risk_free_rate, volatility, spot_price
                        )
                        milstein_paths = StochasticPathGenerator.simulate_milstein_paths(
                            path_count, time_intervals, maturity_time, risk_free_rate, volatility, spot_price
                        )
                        
                        # Store last paths for visualization
                        if idx == len(path_counts) - 1:
                            all_euler_paths = euler_paths
                            all_milstein_paths = milstein_paths
                        
                        # Calculate prices
                        if option_category == OptionCategory.EUROPEAN:
                            euler_price = MonteCarloValuator.price_european_from_simulation(
                                contract_type, euler_paths["price_trajectories"][:, -1], 
                                selected_strike, maturity_time, risk_free_rate
                            )
                            milstein_price = MonteCarloValuator.price_european_from_simulation(
                                contract_type, milstein_paths["price_trajectories"][:, -1], 
                                selected_strike, maturity_time, risk_free_rate
                            )
                        else:  # Binary
                            euler_price = MonteCarloValuator.price_binary_from_simulation(
                                contract_type, euler_paths["price_trajectories"][:, -1], 
                                selected_strike, maturity_time, risk_free_rate
                            )
                            milstein_price = MonteCarloValuator.price_binary_from_simulation(
                                contract_type, milstein_paths["price_trajectories"][:, -1], 
                                selected_strike, maturity_time, risk_free_rate
                            )
                        
                        euler_results.append(float(euler_price))
                        milstein_results.append(float(milstein_price))
                        
                    except Exception as e:
                        st.error(f"Error in simulation for {path_count} paths: {str(e)}")
                        euler_results.append(0.0)
                        milstein_results.append(0.0)
                    
                    progress_bar.progress((idx + 1) / len(path_counts))
                
                # Display results table
                results_data = {
                    "Paths": path_counts,
                    "Euler Price": [f"${p:.4f}" for p in euler_results],
                    "Milstein Price": [f"${p:.4f}" for p in milstein_results],
                    "Euler Error": [f"${abs(p - analytical_price):.4f}" for p in euler_results],
                    "Milstein Error": [f"${abs(p - analytical_price):.4f}" for p in milstein_results]
                }
                
                st.dataframe(results_data, use_container_width=True, hide_index=True)
            
            # ========== CONVERGENCE ANALYSIS ==========
            with tab2:
                st.markdown("### Convergence Analysis")
                
                # Create convergence plot
                convergence_fig = FinancialVisualizer.create_convergence_dashboard(
                    path_counts, euler_results, milstein_results, analytical_price
                )
                st.plotly_chart(convergence_fig, use_container_width=True)
                
                # Error comparison
                euler_errors = [abs(p - analytical_price) for p in euler_results]
                milstein_errors = [abs(p - analytical_price) for p in milstein_results]
                
                error_fig = FinancialVisualizer.create_error_comparison(
                    path_counts, euler_errors, milstein_errors
                )
                st.plotly_chart(error_fig, use_container_width=True)
            
            # ========== RISK METRICS ==========
            with tab3:
                st.markdown("### Risk Analytics & Greeks")
                
                # Calculate Greeks
                if option_category == OptionCategory.EUROPEAN:
                    try:
                        greeks = RiskAnalyzer.calculate_greeks(
                            contract_type, spot_price, selected_strike, volatility, 
                            maturity_time, risk_free_rate
                        )
                        
                        # Display Greeks in clean cards
                        st.markdown("#### Option Greeks")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        greek_cards = [
                            ("Δ Delta", greeks['Delta'], "#667eea", "Price sensitivity"),
                            ("Γ Gamma", greeks['Gamma'], "#764ba2", "Delta sensitivity"),
                            ("ν Vega", greeks['Vega'], "#e53e3e", "Volatility sensitivity"),
                            ("θ Theta", greeks['Theta'], "#38a169", "Time decay (per year)"),
                            ("ρ Rho", greeks['Rho'], "#3182ce", "Interest rate sensitivity")
                        ]
                        
                        for idx, (name, value, color, desc) in enumerate(greek_cards):
                            with [col1, col2, col3, col4, col5][idx]:
                                st.markdown(f'''
                                <div class="metric-card" style="background: linear-gradient(135deg, {color} 0%, {color}99 100%);">
                                    <div class="metric-title">{name}</div>
                                    <div class="metric-value">{value:.4f}</div>
                                    <div style="font-size: 0.7rem; opacity: 0.9;">{desc}</div>
                                </div>
                                ''', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error calculating Greeks: {str(e)}")
                
                # Probability Distribution Analysis
                st.markdown("### Probability Analysis")
                
                if all_euler_paths is not None:
                    terminal_prices = all_euler_paths["price_trajectories"][:, -1]
                    
                    # Create distribution plot
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=terminal_prices,
                        nbinsx=50,
                        name='Terminal Price Distribution',
                        marker_color='cornflowerblue',
                        opacity=0.7,
                        histnorm='probability density'
                    ))
                    
                    # Add strike line
                    fig_dist.add_vline(
                        x=selected_strike,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Strike: {selected_strike:.2f}",
                        annotation_position="top right"
                    )
                    
                    # Add spot line
                    fig_dist.add_vline(
                        x=spot_price,
                        line_dash="dot",
                        line_color="green",
                        annotation_text=f"Spot: {spot_price:.2f}",
                        annotation_position="top left"
                    )
                    
                    fig_dist.update_layout(
                        title="Terminal Price Distribution",
                        xaxis_title="Price at Maturity",
                        yaxis_title="Probability Density",
                        template="plotly_white",
                        height=400
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Calculate probabilities
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        prob_itm = np.mean(terminal_prices > selected_strike) if contract_type == ContractType.CALL else \
                                  np.mean(terminal_prices < selected_strike)
                        st.markdown(f'''
                        <div class="metric-card" style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);">
                            <div class="metric-title">Probability ITM</div>
                            <div class="metric-value">{prob_itm:.2%}</div>
                            <div style="font-size: 0.8rem; opacity: 0.9;">In-the-Money</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col2:
                        prob_otm = 1 - prob_itm
                        st.markdown(f'''
                        <div class="metric-card" style="background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);">
                            <div class="metric-title">Probability OTM</div>
                            <div class="metric-value">{prob_otm:.2%}</div>
                            <div style="font-size: 0.8rem; opacity: 0.9;">Out-of-the-Money</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col3:
                        expected_payoff = np.mean(np.maximum(terminal_prices - selected_strike, 0)) if contract_type == ContractType.CALL else \
                                         np.mean(np.maximum(selected_strike - terminal_prices, 0))
                        st.markdown(f'''
                        <div class="metric-card" style="background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);">
                            <div class="metric-title">Expected Payoff</div>
                            <div class="metric-value">${expected_payoff:.4f}</div>
                            <div style="font-size: 0.8rem; opacity: 0.9;">Discounted Expected Value</div>
                        </div>
                        ''', unsafe_allow_html=True)
            
            # ========== PATH VISUALIZATIONS ==========
            with tab4:
                st.markdown("### Path Simulations")
                
                if all_euler_paths is not None and all_milstein_paths is not None:
                    # Create cleaner path visualizations
                    from plotly.subplots import make_subplots
                    
                    # Clean separated path view
                    clean_fig = FinancialVisualizer.create_clean_path_visualizations(
                        all_euler_paths, all_milstein_paths, min(10, paths_to_display)
                    )
                    st.plotly_chart(clean_fig, use_container_width=True)
                    
                    # Statistical summary
                    st.markdown("#### Distribution Comparison")
                    stats_fig = FinancialVisualizer.create_statistical_path_summary(
                        all_euler_paths, all_milstein_paths
                    )
                    st.plotly_chart(stats_fig, use_container_width=True)
                    
                    # Statistics comparison
                    st.markdown("#### Simulation Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Euler Scheme Statistics")
                        euler_terminal = all_euler_paths["price_trajectories"][:, -1]
                        stats_data = {
                            "Mean": np.mean(euler_terminal),
                            "Std Dev": np.std(euler_terminal),
                            "Min": np.min(euler_terminal),
                            "Max": np.max(euler_terminal),
                            "Skewness": stats.skew(euler_terminal),
                            "Kurtosis": stats.kurtosis(euler_terminal)
                        }
                        import pandas as pd
                        st.dataframe(
                            pd.DataFrame.from_dict(stats_data, orient='index', columns=['Value']).round(4),
                            use_container_width=True,
                            height=250
                        )
                    
                    with col2:
                        st.markdown("##### Milstein Scheme Statistics")
                        milstein_terminal = all_milstein_paths["price_trajectories"][:, -1]
                        stats_data = {
                            "Mean": np.mean(milstein_terminal),
                            "Std Dev": np.std(milstein_terminal),
                            "Min": np.min(milstein_terminal),
                            "Max": np.max(milstein_terminal),
                            "Skewness": stats.skew(milstein_terminal),
                            "Kurtosis": stats.kurtosis(milstein_terminal)
                        }
                        st.dataframe(
                            pd.DataFrame.from_dict(stats_data, orient='index', columns=['Value']).round(4),
                            use_container_width=True,
                            height=250
                        )
    
    else:
        # Initial state - show instructions
        st.info("👈 Configure parameters in the sidebar and click 'Run Simulation' to begin analysis")
        
        # Show feature highlights
        with st.expander("🚀 Platform Features", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">Option Types</div>
                    <div class="metric-value">3+</div>
                    <div style="font-size: 0.8rem; opacity: 0.9;">Call, Put, Straddle</div>
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">Schemes</div>
                    <div class="metric-value">2</div>
                    <div style="font-size: 0.8rem; opacity: 0.9;">Euler & Milstein</div>
                </div>
                ''', unsafe_allow_html=True)
            with col3:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">Analytics</div>
                    <div class="metric-value">5+</div>
                    <div style="font-size: 0.8rem; opacity: 0.9;">Greeks, Convergence, Risk</div>
                </div>
                ''', unsafe_allow_html=True)
            with col4:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">Visualizations</div>
                    <div class="metric-value">4+</div>
                    <div style="font-size: 0.8rem; opacity: 0.9;">Interactive plots</div>
                </div>
                ''', unsafe_allow_html=True)
        
        # Quick example
        st.markdown("### Quick Example Configuration")
        example_config = {
            "Asset Price": "$100",
            "Strike Price": "$100",
            "Maturity": "1 year",
            "Volatility": "20%",
            "Simulation Paths": "1,000 - 10,000"
        }
        st.json(example_config)

# ========== APPLICATION ENTRY POINT ==========
if __name__ == "__main__":
    import pandas as pd
    from plotly.subplots import make_subplots
    execute_quantitative_analysis()