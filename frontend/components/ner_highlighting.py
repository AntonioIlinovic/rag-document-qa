"""NER (Named Entity Recognition) highlighting component.

Provides functions to highlight text with named entities and render entity summaries.
"""

from typing import List, Dict, Optional
import streamlit as st
import os


def load_ner_css():
    """Load NER CSS styles."""
    css_path = os.path.join(os.path.dirname(__file__), "..", "static", "css", "ner.css")
    try:
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback CSS if file not found
        st.markdown("""
        <style>
        [data-entity-type] {
            display: inline-block;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: 500;
            margin: 1px 0;
        }
        .entity-label {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 0.9em;
            margin-right: 8px;
            margin-bottom: 4px;
        }
        </style>
        """, unsafe_allow_html=True)


def highlight_text_with_entities(text: str, entities: List[Dict]) -> str:
    """Generate HTML with highlighted entities from raw entity data.
    
    Args:
        text: Original text to highlight
        entities: List of entity dictionaries with 'text', 'label', 'start', 'end'
        
    Returns:
        HTML string with entities highlighted
    """
    if not entities:
        return text
    
    # Sort entities by start position (reverse order to avoid offset issues)
    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    # Insert HTML markers
    highlighted_text = text
    for entity in sorted_entities:
        start = entity['start']
        end = entity['end']
        entity_text = entity['text']
        entity_type = entity['label']
        
        # Insert highlighting HTML
        highlighted_entity = f'<span data-entity-type="{entity_type}">{entity_text}</span>'
        highlighted_text = highlighted_text[:start] + highlighted_entity + highlighted_text[end:]
    
    return highlighted_text


def render_answer_with_entities(answer: str, entities: List[Dict]) -> None:
    """Render answer with NER highlighting and entity summary.
    
    Args:
        answer: The answer text to render
        entities: List of entity dictionaries
    """
    # Load CSS styles
    load_ner_css()
    
    if entities:
        # Generate highlighted answer
        highlighted_answer = highlight_text_with_entities(answer, entities)
        st.markdown(highlighted_answer, unsafe_allow_html=True)
        
        # Show entity summary
        with st.expander("Named Entities Found"):
            entity_counts = {}
            for entity in entities:
                label = entity['label']
                if label not in entity_counts:
                    entity_counts[label] = []
                entity_counts[label].append(entity['text'])
            
            for label, texts in entity_counts.items():
                # Use colored label that matches entity highlighting
                colored_label = f'<span class="entity-label" data-entity-type="{label}">{label}</span>'
                st.markdown(f"{colored_label}: {', '.join(texts)}", unsafe_allow_html=True)
    else:
        # Fallback to plain text
        st.write(answer)


def render_entity_summary(entities: List[Dict]) -> None:
    """Render just the entity summary section.
    
    Args:
        entities: List of entity dictionaries
    """
    if not entities:
        return
    
    load_ner_css()
    
    with st.expander("Named Entities Found"):
        entity_counts = {}
        for entity in entities:
            label = entity['label']
            if label not in entity_counts:
                entity_counts[label] = []
            entity_counts[label].append(entity['text'])
        
        for label, texts in entity_counts.items():
            # Use colored label that matches entity highlighting
            colored_label = f'<span class="entity-label" data-entity-type="{label}">{label}</span>'
            st.markdown(f"{colored_label}: {', '.join(texts)}", unsafe_allow_html=True)
