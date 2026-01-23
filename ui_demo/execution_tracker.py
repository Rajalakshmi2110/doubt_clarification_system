"""
Review Execution Tracker - Shows step-by-step module execution with real outputs
"""

import streamlit as st
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    st.set_page_config(page_title="Review Execution Tracker", layout="wide")
    st.title("🎓 Academic Doubt Clarification System - Review Execution")
    
    # Module execution status
    modules = {
        "Module 1": {"file": "data/processed/knowledge_chunks_primary_textbook_clean.json", "size": "6,866 chunks"},
        "Module 2": {"file": "data/processed/textbook_faiss.index", "size": "6,850 vectors"},
        "Module 3": {"file": "data/processed/mcp_train.jsonl", "size": "48 examples (150+ available)"},
        "Module 4": {"file": "models/flan_t5_mcp/model.safetensors", "size": "307MB model"},
        "Module 5A": {"file": "modules/module5a_question_validation/enhanced_validator.py", "size": "70+ rules"}
    }
    
    st.header("📊 Real-time Execution Status")
    
    for module, info in modules.items():
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.write(f"**{module}**")
        
        with col2:
            if Path(info["file"]).exists():
                st.success(f"✅ {info['size']}")
            else:
                st.error("❌ Not found")
        
        with col3:
            if st.button(f"Run {module}", key=module):
                run_module(module)
    
    # Live demo section
    st.header("🎯 Live System Demo")
    
    if st.button("🚀 Launch Interactive Demo"):
        st.info("Starting Streamlit demo...")
        try:
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ui_demo/review_demo.py"])
            st.success("Demo launched! Check new browser tab.")
        except Exception as e:
            st.error(f"Error: {e}")

def run_module(module_name):
    """Execute specific module and show progress"""
    st.write(f"Executing {module_name}...")
    
    module_commands = {
        "Module 1": "python modules/module1_knowledge_ingestion/knowledge_ingestion.py",
        "Module 2": "python modules/module2_semantic_indexing/semantic_indexing.py", 
        "Module 3": "python modules/module3_qa_system/mcp_dataset_generator.py",
        "Module 4": "python modules/module4_finetuning/finetune_flan_t5.py",
        "Module 5A": "python modules/module5a_question_validation/enhanced_validator.py"
    }
    
    if module_name in module_commands:
        with st.spinner(f"Running {module_name}..."):
            try:
                result = subprocess.run(module_commands[module_name].split(), 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    st.success(f"{module_name} completed successfully!")
                    st.code(result.stdout)
                else:
                    st.error(f"{module_name} failed!")
                    st.code(result.stderr)
            except subprocess.TimeoutExpired:
                st.warning(f"{module_name} is taking longer than expected...")
            except Exception as e:
                st.error(f"Error running {module_name}: {e}")

if __name__ == "__main__":
    main()