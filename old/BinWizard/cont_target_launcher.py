
import os
import subprocess

def contGUI():
    # Construir la ruta completa al script dentro del paquete
    script_path = os.path.join(os.path.dirname(__file__), 'continuous_target.py')
    
    if os.path.exists(script_path):
        try:
            subprocess.run(["streamlit", "run", script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error al intentar ejecutar Streamlit para app1: {e}")
    else:
        raise FileNotFoundError(f"Invalid value: File does not exist: {script_path}")
