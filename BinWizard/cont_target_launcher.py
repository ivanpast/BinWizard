import subprocess

def contGUI():
    """
    Función que lanza la primera aplicación Streamlit desde la línea de comandos.
    """
    try:
        subprocess.run(["streamlit", "run", "continuous_target.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error al intentar ejecutar Streamlit para app1: {e}")
