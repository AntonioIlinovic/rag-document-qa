"""Configuration section component for user settings.

Provides toggle switches for user-configurable settings like QA engine and NER.
"""

import streamlit as st
from typing import Optional


def render_config_section() -> None:
    """Render the configuration section in the sidebar.
    
    This section allows users to toggle between QA engines and enable/disable NER.
    Settings are stored in session state only and reset to defaults on app restart.
    """
    with st.expander("⚙️ Settings", expanded=False):
        st.markdown("### Model Configuration")
        
        # Initialize session state for settings if not exists (using frontend defaults)
        if "qa_engine" not in st.session_state or "ner_enabled" not in st.session_state:
            st.session_state.qa_engine = "cloud"  # Frontend default
            st.session_state.ner_enabled = True   # Frontend default
        
        # QA Engine Toggle
        qa_engine_help = """
        **QA (Question Answering) Engine Selection:**
        - **Local**: Uses local DistilBERT QA model
        - **Cloud**: Uses OpenAI gpt-4o-mini model for QA
        """
        
        qa_engine = st.radio(
            "QA Engine",
            options=["local", "cloud"],
            index=0 if st.session_state.qa_engine == "local" else 1,
            format_func=lambda x: "Local" if x == "local" else "Cloud",
            help=qa_engine_help,
            key="qa_engine_radio"
        )
        
        # Update session state
        st.session_state.qa_engine = qa_engine
        
        # NER Toggle
        ner_help = """
        **Named Entity Recognition (NER):**
        
        When enabled, the system will:
        - Identify and highlight entities (people, organizations, locations, etc.)
        - Show entity types with color coding
        - Display entity summary in expandable section
        
        Disable for cleaner text display or if entity detection is not needed.
        """
        
        ner_enabled = st.toggle(
            "Enable NER",
            value=st.session_state.ner_enabled,
            help=ner_help,
            key="ner_enabled_toggle"
        )
        
        # Update session state
        st.session_state.ner_enabled = ner_enabled
        
        # Note about persistence
        st.caption("Settings reset to defaults when you refresh the page")


def get_qa_engine() -> str:
    """Get the current QA engine setting from session state.
    
    Returns:
        Current QA engine ('local' or 'cloud')
        Note: This should always be set by the config section
    """
    return st.session_state.get("qa_engine", "cloud")


def is_ner_enabled() -> bool:
    """Get the current NER enabled setting from session state.
    
    Returns:
        True if NER is enabled, False otherwise
        Note: This should always be set by the config section
    """
    return st.session_state.get("ner_enabled", True)
