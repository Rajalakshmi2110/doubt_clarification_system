"""
First Review Demo Launcher
Run this script to start the demo interface for the first review
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages for UI demo"""
    print("📦 Installing UI requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "ui_requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements. Please install manually:")
        print("pip install streamlit pandas numpy plotly")
        return False
    return True

def launch_demo():
    """Launch the Streamlit demo"""
    print("🚀 Launching First Review Demo...")
    print("📍 Demo will open in your browser at: http://localhost:8501")
    print("🔄 Starting Streamlit server...")
    
    # Change to project root directory
    os.chdir('..')
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ui_demo/review_demo.py"])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error launching demo: {e}")

def main():
    print("=" * 60)
    print("🎓 ACADEMIC DOUBT CLARIFICATION SYSTEM")
    print("📋 First Review Demo (30% Completion)")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("../modules"):
        print("❌ Error: Please run this script from the ui_demo directory")
        print("📁 Expected structure: project/ui_demo/launch_review_demo.py")
        return
    
    print("📊 Project Status:")
    print("✅ Module 1: Knowledge Ingestion (6,866 chunks)")
    print("✅ Module 2: Semantic Indexing (FAISS vector DB)")
    print("✅ Module 3: Dataset Generation (MCP format)")
    print("✅ Module 4: Model Fine-tuning (FLAN-T5)")
    print("✅ Module 5A: Question Validation (Enhanced)")
    print()
    
    # Install requirements
    if not install_requirements():
        return
    
    print()
    print("🎯 Demo Features:")
    print("• Step-by-step module execution")
    print("• Real-time data visualization") 
    print("• Interactive question validation")
    print("• Live system demonstration")
    print()
    
    input("Press Enter to launch the demo interface...")
    launch_demo()

if __name__ == "__main__":
    main()