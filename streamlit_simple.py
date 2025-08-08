"""Simple Streamlit web application for viewing experiment results."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="NorthAU Experiment Viewer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database connection
DB_PATH = Path(__file__).parent / "data" / "autotuning.db"

def get_database_connection():
    """Get database connection."""
    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}")
        st.info("Please run some experiments first to create the database.")
        return None
    return sqlite3.connect(str(DB_PATH))

@st.cache_data
def load_sessions():
    """Load all sessions from database."""
    conn = get_database_connection()
    if conn is None:
        return []
    
    try:
        query = """
        SELECT id, name, mode, dataset_name, dataset_version, status, 
               total_experiments, best_score, total_time_seconds, created_at, completed_at
        FROM sessions
        ORDER BY created_at DESC
        """
        df = pd.read_sql_query(query, conn)
        return df.to_dict('records')
    except Exception as e:
        st.error(f"Error loading sessions: {e}")
        return []
    finally:
        conn.close()

@st.cache_data
def load_experiments(session_id: str):
    """Load experiments for a session."""
    conn = get_database_connection()
    if conn is None:
        return []
    
    try:
        query = """
        SELECT id, session_id, config_id, config_data, dataset_items, status,
               execution_time_seconds, overall_score, metric_scores, token_usage,
               api_calls, cost_usd, created_at
        FROM experiments
        WHERE session_id = ?
        ORDER BY overall_score DESC NULLS LAST
        """
        df = pd.read_sql_query(query, conn, params=[session_id])
        
        # Parse JSON fields
        for index, row in df.iterrows():
            for json_col in ['config_data', 'metric_scores', 'token_usage']:
                if pd.notna(row[json_col]):
                    try:
                        df.at[index, json_col] = json.loads(row[json_col])
                    except json.JSONDecodeError:
                        df.at[index, json_col] = {}
                else:
                    df.at[index, json_col] = {}
        
        return df.to_dict('records')
    except Exception as e:
        st.error(f"Error loading experiments: {e}")
        return []
    finally:
        conn.close()

@st.cache_data
def load_item_results(experiment_id: str):
    """Load item results for an experiment."""
    conn = get_database_connection()
    if conn is None:
        return []
    
    try:
        query = """
        SELECT id, experiment_id, item_id, score, metric_scores, agent_output,
               execution_time, token_usage, evaluation_feedback, error_message, created_at
        FROM item_results
        WHERE experiment_id = ?
        ORDER BY score DESC NULLS LAST
        """
        df = pd.read_sql_query(query, conn, params=[experiment_id])
        
        # Parse JSON fields
        for index, row in df.iterrows():
            for json_col in ['metric_scores', 'token_usage']:
                if pd.notna(row[json_col]):
                    try:
                        df.at[index, json_col] = json.loads(row[json_col])
                    except json.JSONDecodeError:
                        df.at[index, json_col] = {}
                else:
                    df.at[index, json_col] = {}
        
        return df.to_dict('records')
    except Exception as e:
        st.error(f"Error loading item results: {e}")
        return []
    finally:
        conn.close()

@st.cache_data
def load_dataset_items(dataset_name: str, dataset_version: str = None):
    """Load dataset items from files or database if available."""
    # Try to find dataset files in common locations
    dataset_paths = [
        # JSON files
        Path(__file__).parent / "data" / f"{dataset_name}.json",
        Path(__file__).parent / "datasets" / f"{dataset_name}.json", 
        Path(__file__).parent / f"{dataset_name}.json",
        # JSONL files
        Path(__file__).parent / "data" / f"{dataset_name}.jsonl",
        Path(__file__).parent / "datasets" / f"{dataset_name}.jsonl",
        Path(__file__).parent / f"{dataset_name}.jsonl",
        # Specific examples path for prompt_hacking
        Path(__file__).parent / "examples" / "prompt_hacking" / "sample_dataset.jsonl",
        Path(__file__).parent / "examples" / f"{dataset_name}" / "sample_dataset.jsonl",
    ]
    
    for path in dataset_paths:
        if path.exists():
            try:
                if path.suffix == '.jsonl':
                    # Handle JSONL files (one JSON object per line)
                    items = []
                    with open(path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    items.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
                    return items
                else:
                    # Handle regular JSON files
                    with open(path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            return data
                        elif isinstance(data, dict) and 'items' in data:
                            return data['items']
            except Exception as e:
                st.warning(f"Error loading dataset from {path}: {e}")
                continue
    
    # If no dataset files found, return empty list
    return []

def format_datetime(dt_str):
    """Format datetime string for display."""
    if dt_str:
        try:
            if dt_str.endswith('Z'):
                dt_str = dt_str[:-1] + '+00:00'
            dt = datetime.fromisoformat(dt_str)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return dt_str
    return "N/A"

def format_duration(seconds):
    """Format duration in seconds to human readable."""
    if seconds is None or pd.isna(seconds):
        return "N/A"
    try:
        seconds = int(float(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except:
        return "N/A"

def create_score_distribution_chart(experiments):
    """Create score distribution chart."""
    if not experiments:
        return None
    
    scores = []
    for exp in experiments:
        score = exp.get('overall_score')
        if score is not None and not pd.isna(score):
            scores.append(float(score))
    
    if not scores:
        return None
    
    fig = px.histogram(
        x=scores,
        nbins=min(20, len(scores)),
        title="Score Distribution Across Experiments",
        labels={'x': 'Overall Score', 'y': 'Count'}
    )
    fig.update_layout(showlegend=False)
    return fig

def create_metrics_comparison_chart(experiments):
    """Create metrics comparison chart."""
    if not experiments:
        return None
    
    # Extract all metric names
    all_metrics = set()
    for exp in experiments:
        if exp.get('metric_scores'):
            all_metrics.update(exp['metric_scores'].keys())
    
    if not all_metrics:
        return None
    
    # Create metrics data
    metrics_data = []
    for exp in experiments:
        config_id = exp.get('config_id', 'Unknown')
        metric_scores = exp.get('metric_scores', {})
        
        for metric in all_metrics:
            if metric in metric_scores:
                metrics_data.append({
                    'Config': config_id,
                    'Metric': metric,
                    'Score': metric_scores[metric]
                })
    
    if not metrics_data:
        return None
    
    df = pd.DataFrame(metrics_data)
    fig = px.bar(
        df,
        x='Config',
        y='Score',
        color='Metric',
        title="Metrics Comparison Across Configurations",
        barmode='group'
    )
    fig.update_xaxes(tickangle=45)
    return fig

def render_session_overview():
    """Render session overview page."""
    st.title("🧪 NorthAU Experiment Viewer")
    
    # Add refresh button at the top
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 Refresh Data", type="secondary"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        st.info("Click refresh if status seems outdated")
    
    st.markdown("---")
    
    # Load sessions
    sessions = load_sessions()
    
    if not sessions:
        st.warning("No evaluation sessions found in the database.")
        st.info("Run some experiments first to see results here!")
        return
    
    st.subheader(f"📊 Sessions Overview ({len(sessions)} total)")
    
    # Create sessions dataframe for display
    sessions_data = []
    for session in sessions:
        sessions_data.append({
            'Session ID': str(session.get('id', ''))[:8] + '...',
            'Name': session.get('name', 'Unnamed'),
            'Mode': session.get('mode', 'Unknown'),
            'Dataset': f"{session.get('dataset_name', 'Unknown')} v{session.get('dataset_version', '?')}",
            'Status': session.get('status', 'Unknown'),
            'Experiments': session.get('total_experiments', 0) or 0,
            'Best Score': f"{float(session.get('best_score', 0)):.4f}" if session.get('best_score') is not None else "N/A",
            'Duration': format_duration(session.get('total_time_seconds')),
            'Created': format_datetime(session.get('created_at'))
        })
    
    sessions_df = pd.DataFrame(sessions_data)
    
    # Display sessions table
    st.dataframe(
        sessions_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sessions", len(sessions))
    with col2:
        completed_sessions = len([s for s in sessions if s.get('status') == 'completed'])
        st.metric("Completed", completed_sessions)
    with col3:
        total_experiments = sum(s.get('total_experiments', 0) or 0 for s in sessions)
        st.metric("Total Experiments", total_experiments)
    with col4:
        best_scores = [float(s.get('best_score', 0)) for s in sessions if s.get('best_score') is not None]
        avg_best_score = sum(best_scores) / len(best_scores) if best_scores else 0
        st.metric("Avg Best Score", f"{avg_best_score:.4f}")

def render_session_details(session_id: str):
    """Render detailed session view."""
    sessions = load_sessions()
    session = next((s for s in sessions if s['id'] == session_id), None)
    
    if not session:
        st.error("Session not found!")
        return
    
    st.title(f"📋 Session Details: {session.get('name', 'Unnamed')}")
    st.markdown("---")
    
    # Session info
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Session Information")
        st.write(f"**ID:** `{session['id']}`")
        st.write(f"**Name:** {session.get('name', 'Unnamed')}")
        st.write(f"**Mode:** {session.get('mode', 'Unknown')}")
        st.write(f"**Status:** {session.get('status', 'Unknown')}")
        st.write(f"**Dataset:** {session.get('dataset_name', 'Unknown')} v{session.get('dataset_version', '?')}")
    
    with col2:
        st.subheader("Performance Metrics")
        st.write(f"**Total Experiments:** {session.get('total_experiments', 0) or 0}")
        if session.get('best_score') is not None:
            st.write(f"**Best Score:** {float(session.get('best_score')):.4f}")
        else:
            st.write("**Best Score:** N/A")
        st.write(f"**Total Duration:** {format_duration(session.get('total_time_seconds'))}")
        st.write(f"**Created:** {format_datetime(session.get('created_at'))}")
        if session.get('completed_at'):
            st.write(f"**Completed:** {format_datetime(session.get('completed_at'))}")
    
    # Load experiments for this session
    experiments = load_experiments(session_id)
    
    if not experiments:
        st.warning("No experiments found for this session.")
        return
    
    st.markdown("---")
    st.subheader(f"🔬 Experiments ({len(experiments)} total)")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Analytics", "📋 Details", "💬 Item I/O"])
    
    with tab1:
        render_experiments_overview(experiments)
    
    with tab2:
        render_experiments_analytics(experiments)
    
    with tab3:
        render_experiments_details(experiments)
    
    with tab4:
        render_item_io_tab(experiments)

def render_experiments_overview(experiments):
    """Render experiments overview tab."""
    # Create experiments dataframe
    experiments_data = []
    for exp in experiments:
        token_usage = exp.get('token_usage', {})
        total_tokens = token_usage.get('total_tokens', 0) if isinstance(token_usage, dict) else 0
        
        experiments_data.append({
            'Config ID': exp.get('config_id', 'Unknown'),
            'Status': exp.get('status', 'Unknown'),
            'Overall Score': f"{float(exp.get('overall_score', 0)):.4f}" if exp.get('overall_score') is not None else "N/A",
            'Dataset Items': exp.get('dataset_items', 0) or 0,
            'Execution Time': format_duration(exp.get('execution_time_seconds')),
            'API Calls': exp.get('api_calls', 0) or 0,
            'Token Usage': total_tokens,
            'Cost (USD)': f"${float(exp.get('cost_usd', 0)):.4f}" if exp.get('cost_usd') is not None else "$0.0000",
            'Created': format_datetime(exp.get('created_at'))
        })
    
    experiments_df = pd.DataFrame(experiments_data)
    
    st.dataframe(
        experiments_df,
        use_container_width=True,
        hide_index=True
    )

def render_experiments_analytics(experiments):
    """Render experiments analytics tab."""
    # Score distribution chart
    score_chart = create_score_distribution_chart(experiments)
    if score_chart:
        st.plotly_chart(score_chart, use_container_width=True)
    else:
        st.info("No score data available for visualization.")
    
    # Metrics comparison chart
    metrics_chart = create_metrics_comparison_chart(experiments)
    if metrics_chart:
        st.plotly_chart(metrics_chart, use_container_width=True)
    else:
        st.info("No metrics data available for comparison.")

def render_experiments_details(experiments):
    """Render detailed experiments results."""
    if not experiments:
        st.warning("No experiments to display.")
        return
    
    # Select experiment to view details
    config_ids = [exp.get('config_id', f"Unknown_{i}") for i, exp in enumerate(experiments)]
    selected_config = st.selectbox("Select Experiment to View Details", config_ids)
    
    if selected_config:
        selected_exp = next((exp for exp in experiments if exp.get('config_id') == selected_config), None)
        if selected_exp:
            render_single_experiment_details(selected_exp)

def render_single_experiment_details(experiment):
    """Render details for a single experiment."""
    st.subheader(f"🔍 Experiment: {experiment.get('config_id', 'Unknown')}")
    
    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        if experiment.get('overall_score') is not None:
            st.metric("Overall Score", f"{float(experiment.get('overall_score')):.4f}")
        else:
            st.metric("Overall Score", "N/A")
    with col2:
        st.metric("Execution Time", format_duration(experiment.get('execution_time_seconds')))
    with col3:
        st.metric("Dataset Items", experiment.get('dataset_items', 0) or 0)
    
    # Configuration details
    config_data = experiment.get('config_data', {})
    if config_data:
        with st.expander("📋 Configuration Details", expanded=False):
            # System prompt
            if config_data.get('system_prompts'):
                if isinstance(config_data['system_prompts'], dict):
                    for key, value in config_data['system_prompts'].items():
                        st.subheader(f"System Prompt {key}")
                        st.text_area(f"System Prompt: {key}", value, height=200, disabled=True, key=f"prompt_{experiment.get('id')}", label_visibility="collapsed")
                else:
                    st.text_area("System Prompt", config_data['system_prompts'], height=200, disabled=True, key=f"prompt_{experiment.get('id')}", label_visibility="collapsed")
            
            # LLM config
            if config_data.get('llm_config'):
                st.subheader("LLM Configuration")
                st.json(config_data['llm_config'])
    
    # Metrics breakdown
    metric_scores = experiment.get('metric_scores', {})
    if metric_scores:
        st.subheader("📊 Metrics Breakdown")
        
        # Create metrics chart
        metrics_df = pd.DataFrame([
            {'Metric': k, 'Score': v} 
            for k, v in metric_scores.items()
        ])
        
        fig = px.bar(
            metrics_df,
            x='Metric',
            y='Score',
            title="Individual Metric Scores"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics table
        st.dataframe(
            metrics_df.set_index('Metric'),
            use_container_width=True
        )
    
    # Item results
    st.subheader("📝 Item Results")
    item_results = load_item_results(experiment['id'])
    
    if item_results:
        # Summary metrics
        successful_items = len([item for item in item_results if not item.get('error_message')])
        success_rate = successful_items / len(item_results) if item_results else 0
        
        scores = [float(item.get('score', 0)) for item in item_results if item.get('score') is not None]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        exec_times = [float(item.get('execution_time', 0)) for item in item_results if item.get('execution_time') is not None]
        avg_execution_time = sum(exec_times) / len(exec_times) if exec_times else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Items", len(item_results))
        with col2:
            st.metric("Success Rate", f"{success_rate:.2%}")
        with col3:
            st.metric("Avg Score", f"{avg_score:.4f}")
        with col4:
            st.metric("Avg Time", f"{avg_execution_time:.2f}s")
        
        # Item results table
        items_data = []
        for item in item_results:
            items_data.append({
                'Item ID': item.get('item_id', 'Unknown'),
                'Score': f"{float(item.get('score', 0)):.4f}" if item.get('score') is not None else "N/A",
                'Execution Time': f"{float(item.get('execution_time', 0)):.2f}s" if item.get('execution_time') is not None else "N/A",
                'Output Length': len(item.get('agent_output', '')),
                'Has Error': 'Yes' if item.get('error_message') else 'No'
            })
        
        items_df = pd.DataFrame(items_data)
        
        st.dataframe(
            items_df,
            use_container_width=True,
            hide_index=True
        )
        
    else:
        st.warning("No item results found for this experiment.")

def render_item_io_tab(experiments):
    """Render the Item Input/Output tab showing detailed I/O for each item."""
    if not experiments:
        st.warning("No experiments to display.")
        return
    
    st.subheader("💬 Item Input & Output Details")
    
    # Select experiment
    config_ids = [exp.get('config_id', f"Unknown_{i}") for i, exp in enumerate(experiments)]
    selected_config = st.selectbox(
        "Select Experiment", 
        config_ids,
        key="io_experiment_select"
    )
    
    if not selected_config:
        return
    
    selected_exp = next((exp for exp in experiments if exp.get('config_id') == selected_config), None)
    if not selected_exp:
        return
    
    # Load item results for selected experiment
    item_results = load_item_results(selected_exp['id'])
    
    if not item_results:
        st.warning("No item results found for this experiment.")
        return
    
    # Try to load original dataset items for input data
    sessions = load_sessions()
    current_session = None
    for session in sessions:
        # Find the session this experiment belongs to
        session_experiments = load_experiments(session['id'])
        if any(exp['id'] == selected_exp['id'] for exp in session_experiments):
            current_session = session
            break
    
    dataset_items = []
    if current_session:
        dataset_name = current_session.get('dataset_name')
        dataset_version = current_session.get('dataset_version')
        if dataset_name:
            dataset_items = load_dataset_items(dataset_name, dataset_version)
    
    # Create a mapping from item_id to original dataset item
    dataset_map = {}
    for item in dataset_items:
        if isinstance(item, dict):
            # Try different ID field names
            item_id = item.get('id') or item.get('item_id') or item.get('identifier')
            if item_id:
                dataset_map[str(item_id)] = item
    
    st.info(f"Found {len(item_results)} item results. " + 
            (f"Original dataset has {len(dataset_items)} items." if dataset_items else 
             "No original dataset found - showing outputs only."))
    
    # Debug info
    if dataset_items:
        first_item = dataset_items[0] if dataset_items else {}
        st.write(f"**Dataset info:** Found fields: {list(first_item.keys()) if first_item else 'None'}")
    else:
        if current_session:
            st.write(f"**Looking for dataset:** {current_session.get('dataset_name')} v{current_session.get('dataset_version')}")
        st.write("**Searched paths:** examples/prompt_hacking/sample_dataset.jsonl, data/{dataset_name}.jsonl, etc.")
    
    # Display options
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        # Filter and sort options
        sort_by = st.selectbox(
            "Sort by",
            ["Score (High to Low)", "Score (Low to High)", "Item ID", "Execution Time"],
            key="io_sort"
        )
    with col2:
        show_errors_only = st.checkbox("Show errors only", key="io_errors_only")
    with col3:
        max_items = st.number_input(
            "Max items to show", 
            min_value=1, 
            max_value=len(item_results), 
            value=min(10, len(item_results)),
            key="io_max_items"
        )
    
    # Apply filters and sorting
    filtered_results = item_results.copy()
    
    if show_errors_only:
        filtered_results = [item for item in filtered_results if item.get('error_message')]
    
    # Sort items
    if sort_by == "Score (High to Low)":
        filtered_results.sort(key=lambda x: x.get('score', 0) or 0, reverse=True)
    elif sort_by == "Score (Low to High)":
        filtered_results.sort(key=lambda x: x.get('score', 0) or 0)
    elif sort_by == "Item ID":
        filtered_results.sort(key=lambda x: x.get('item_id', ''))
    elif sort_by == "Execution Time":
        filtered_results.sort(key=lambda x: x.get('execution_time', 0) or 0, reverse=True)
    
    # Limit results
    filtered_results = filtered_results[:max_items]
    
    if not filtered_results:
        st.warning("No items match the current filters.")
        return
    
    # Display items
    for i, item in enumerate(filtered_results):
        with st.expander(
            f"📄 Item {item.get('item_id', 'Unknown')} - Score: {item.get('score', 0):.4f}",
            expanded=(i == 0)  # Expand first item by default
        ):
            # Item metadata
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Score", f"{item.get('score', 0):.4f}")
            with col2:
                st.metric("Execution Time", f"{item.get('execution_time', 0):.2f}s")
            with col3:
                has_error = "Yes" if item.get('error_message') else "No"
                st.metric("Has Error", has_error)
            with col4:
                output_length = len(item.get('agent_output', ''))
                st.metric("Output Length", f"{output_length:,} chars")
            
            # Show original input if available
            item_id = str(item.get('item_id', ''))
            original_item = dataset_map.get(item_id)
            
            if original_item:
                st.subheader("📥 Original Input")
                
                # Display different input fields based on what's available
                input_fields = ['message', 'input_message', 'input_data', 'input', 'question', 'prompt', 'text', 'content']
                input_found = False
                
                for field in input_fields:
                    if field in original_item:
                        st.text_area(
                            f"Input ({field})",
                            original_item[field],
                            height=150,
                            disabled=True,
                            key=f"input_{item['id']}_{field}",
                            label_visibility="collapsed"
                        )
                        input_found = True
                        break
                
                if not input_found:
                    # Show the entire original item as JSON
                    st.json(original_item)
                
                # Show expected output if available  
                expected_fields = ['expected', 'expected_output', 'answer', 'target', 'label']
                for field in expected_fields:
                    if field in original_item:
                        st.subheader(f"🎯 Expected Output ({field})")
                        st.text_area(
                            f"Expected ({field})",
                            str(original_item[field]),
                            height=100,
                            disabled=True,
                            key=f"expected_{item['id']}_{field}",
                            label_visibility="collapsed"
                        )
                        break
            else:
                st.info(f"Original input for item '{item_id}' not found in dataset.")
            
            # Agent output
            st.subheader("🤖 Agent Output")
            agent_output = item.get('agent_output', '')
            if agent_output:
                st.text_area(
                    "Agent Output",
                    agent_output,
                    height=200,
                    disabled=True,
                    key=f"output_{item['id']}",
                    label_visibility="collapsed"
                )
            else:
                st.warning("No agent output available")
            
            # Error message
            if item.get('error_message'):
                st.subheader("❌ Error Message")
                st.error(item['error_message'])
            
            # Evaluation feedback
            if item.get('evaluation_feedback'):
                st.subheader("📝 Evaluation Feedback")
                st.text_area(
                    "Evaluation Feedback",
                    item['evaluation_feedback'],
                    height=100,
                    disabled=True,
                    key=f"feedback_{item['id']}",
                    label_visibility="collapsed"
                )
            
            # Individual metric scores
            if item.get('metric_scores'):
                st.subheader("📊 Individual Metric Scores")
                metrics_data = [
                    {"Metric": k, "Score": f"{v:.4f}"}
                    for k, v in item['metric_scores'].items()
                ]
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(
                        metrics_df.set_index('Metric'),
                        use_container_width=True
                    )
            
            # Token usage details
            if item.get('token_usage'):
                st.subheader("🔢 Token Usage")
                st.json(item['token_usage'])

def main():
    """Main Streamlit application."""
    # Check database first
    if not DB_PATH.exists():
        st.error(f"Database not found at {DB_PATH}")
        st.info("Please run some experiments first to create the database.")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("🧪 Navigation")
    
    # Navigation options
    page = st.sidebar.selectbox(
        "Select View",
        ["Sessions Overview", "Session Details"]
    )
    
    if page == "Sessions Overview":
        render_session_overview()
    
    elif page == "Session Details":
        # Session selection
        sessions = load_sessions()
        if sessions:
            session_options = {
                f"{session.get('name', 'Unnamed')} ({str(session['id'])[:8]}...)": session['id']
                for session in sessions
            }
            
            selected_session = st.sidebar.selectbox(
                "Select Session",
                list(session_options.keys())
            )
            
            if selected_session:
                session_id = session_options[selected_session]
                render_session_details(session_id)
        else:
            st.sidebar.warning("No sessions available")
            render_session_overview()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Refresh Data")
    if st.sidebar.button("Clear Cache & Refresh", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("**💡 Tip:** If experiment status shows 'running' but experiments are complete, click 'Clear Cache & Refresh' to see updated data.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    ### 📚 About
    This dashboard shows results from NorthAU evaluation sessions.
    
    **Database:** `{DB_PATH}`
    
    **Features:**
    - Session overview and filtering
    - Experiment comparison and analytics
    - Detailed configuration viewing
    - Item-level result inspection
    """)

if __name__ == "__main__":
    main()