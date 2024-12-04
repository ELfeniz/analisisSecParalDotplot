import os
import numpy as np
import matplotlib.pyplot as plt

def cargar_tiempos():
    """
    Carga los tiempos de ejecución de los métodos desde los archivos .txt generados por el código principal.
    """
    tiempos = {}

    archivos = [
        "tiempos/tiempo_secuencial.txt", 
        "tiempos/tiempo_multiprocessing.txt", 
        "tiempos/tiempo_mpi.txt", 
        "tiempos/tiempo_pycuda.txt"
    ]

    for archivo in archivos:
        if os.path.exists(archivo):
            with open(archivo, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        key, value = line.strip().split(": ")
                        tiempos[key] = float(value.replace(" segundos", ""))
        else:
            print(f"Archivo {archivo} no encontrado.")

    return tiempos

def calcular_metricas(tiempos):
    """
    Calcula las métricas de rendimiento como aceleración, eficiencia y tiempo muerto.
    """
    # Tiempos de cálculo de dotplot
    secuencial = tiempos["Tiempo calculo dotplot secuencial"]
    multiprocessing = tiempos["Tiempo calculo dotplot multiprocessing"]
    mpi = tiempos["Tiempo calculo dotplot mpi"]
    pycuda = tiempos["Tiempo calculo dotplot pycuda"]

    # Tiempos de graficado
    secuencial_graficas = tiempos["Tiempo Graficas secuencial"]
    multiprocessing_graficas = tiempos["Tiempo Graficas multiprocessing"]
    mpi_graficas = tiempos["Tiempo Graficas mpi"]
    pycuda_graficas = tiempos["Tiempo Graficas pycuda"]

    # Aceleración
    aceleracion_multiprocessing = secuencial / multiprocessing
    aceleracion_mpi = secuencial / mpi
    aceleracion_pycuda = secuencial / pycuda
    
    # Eficiencia (considerando 4 procesadores en multiprocessing y mpi)
    eficiencia_multiprocessing = aceleracion_multiprocessing / 4
    eficiencia_mpi = aceleracion_mpi / 4
    eficiencia_pycuda = aceleracion_pycuda  # Asumiendo eficiencia total de la GPU

    # Tiempo muerto (resta entre el tiempo total y el tiempo útil de ejecución)
    tiempo_muerto_multiprocessing = (tiempos["Tiempo filtrado diagonales multiprocessing"] + 
                                     tiempos["Tiempo Graficas multiprocessing"]) - tiempos["Tiempo calculo dotplot multiprocessing"]
    tiempo_muerto_mpi = (tiempos["Tiempo filtrado diagonales mpi"] + 
                         tiempos["Tiempo Graficas mpi"]) - tiempos["Tiempo calculo dotplot mpi"]
    tiempo_muerto_pycuda = (tiempos["Tiempo filtrado diagonales pycuda"] + 
                            tiempos["Tiempo Graficas pycuda"]) - tiempos["Tiempo calculo dotplot pycuda"]


    # Tiempo total (cálculo dotplot + graficado)
    tiempo_total_multiprocessing = tiempos["Tiempo calculo dotplot multiprocessing"] + tiempos["Tiempo Graficas multiprocessing"]
    tiempo_total_mpi = tiempos["Tiempo calculo dotplot mpi"] + tiempos["Tiempo Graficas mpi"]
    tiempo_total_pycuda = tiempos["Tiempo calculo dotplot pycuda"] + tiempos["Tiempo Graficas pycuda"]
    tiempo_total_secuencial = tiempos["Tiempo calculo dotplot secuencial"] + tiempos["Tiempo Graficas secuencial"]

    return {
        "aceleracion_multiprocessing": aceleracion_multiprocessing,
        "aceleracion_mpi": aceleracion_mpi,
        "aceleracion_pycuda": aceleracion_pycuda,
        "eficiencia_multiprocessing": eficiencia_multiprocessing,
        "eficiencia_mpi": eficiencia_mpi,
        "eficiencia_pycuda": eficiencia_pycuda,
        "tiempo_muerto_multiprocessing": tiempo_muerto_multiprocessing,
        "tiempo_muerto_mpi": tiempo_muerto_mpi,
        "tiempo_muerto_pycuda": tiempo_muerto_pycuda,
        "tiempo_total_secuencial": tiempo_total_secuencial,
        "tiempo_total_multiprocessing": tiempo_total_multiprocessing,
        "tiempo_total_mpi": tiempo_total_mpi,
        "tiempo_total_pycuda": tiempo_total_pycuda
    }

def generar_graficas(tiempos, metricas):
    """
    Genera las gráficas de desempeño, aceleración, eficiencia, tiempo muerto y tiempo total.
    """
    # Definir las etiquetas de los métodos
    labels = ['Secuencial', 'Multiprocessing', 'MPI', 'PyCUDA']
    tiempos_ejecucion = [
        tiempos["Tiempo calculo dotplot secuencial"],
        tiempos["Tiempo calculo dotplot multiprocessing"],
        tiempos["Tiempo calculo dotplot mpi"],
        tiempos["Tiempo calculo dotplot pycuda"]
    ]
    
    # Gráfico de desempeño (tiempos de ejecución)
    plt.figure(figsize=(10, 6))
    plt.bar(labels, tiempos_ejecucion, color=['blue', 'green', 'red', 'orange'])
    plt.xlabel('Método')
    plt.ylabel('Tiempo de ejecución (segundos)')
    plt.title('Comparación de Tiempos de Ejecución')
    plt.tight_layout()
    plt.savefig('graficas_metricas/desempeno.png')
    plt.show()

    # Gráfico de aceleración
    aceleraciones = [
        metricas["aceleracion_multiprocessing"],
        metricas["aceleracion_mpi"],
        metricas["aceleracion_pycuda"]
    ]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels[1:], aceleraciones, color=['green', 'red', 'orange'])
    plt.xlabel('Método')
    plt.ylabel('Aceleración')
    plt.title('Comparación de Aceleración entre Métodos')
    plt.tight_layout()
    plt.savefig('graficas_metricas/aceleracion.png')
    plt.show()

    # Gráfico de eficiencia
    eficiencias = [
        metricas["eficiencia_multiprocessing"],
        metricas["eficiencia_mpi"],
        metricas["eficiencia_pycuda"]
    ]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels[1:], eficiencias, color=['green', 'red', 'orange'])
    plt.xlabel('Método')
    plt.ylabel('Eficiencia')
    plt.title('Comparación de Eficiencia entre Métodos')
    plt.tight_layout()
    plt.savefig('graficas_metricas/eficiencia.png')
    plt.show()

    # Gráfico de tiempo muerto
    tiempos_muertos = [
        metricas["tiempo_muerto_multiprocessing"],
        metricas["tiempo_muerto_mpi"],
        metricas["tiempo_muerto_pycuda"]
    ]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels[1:], tiempos_muertos, color=['green', 'red', 'orange'])
    plt.xlabel('Método')
    plt.ylabel('Tiempo Muerto (segundos)')
    plt.title('Comparación de Tiempo Muerto entre Métodos')
    plt.tight_layout()
    plt.savefig('graficas_metricas/tiempo_muerto.png')
    plt.show()

    # Gráfico de tiempo total
    tiempos_totales = [
        metricas["tiempo_total_secuencial"],
        metricas["tiempo_total_multiprocessing"],
        metricas["tiempo_total_mpi"],
        metricas["tiempo_total_pycuda"]
    ]
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, tiempos_totales, color=['blue', 'green', 'red', 'orange'])
    plt.xlabel('Método')
    plt.ylabel('Tiempo Total (segundos)')
    plt.title('Comparación de Tiempo Total de Ejecución')
    plt.tight_layout()
    plt.savefig('graficas_metricas/tiempo_total.png')
    plt.show()

def imprimir_metricas(metricas):
    """
    Imprime las métricas por consola.
    """
    print("\nMétricas calculadas:")
    print(f"Aceleración Multiprocessing: {metricas['aceleracion_multiprocessing']:.4f}")
    print(f"Aceleración MPI: {metricas['aceleracion_mpi']:.4f}")
    print(f"Aceleración PyCUDA: {metricas['aceleracion_pycuda']:.4f}")
    print(f"Eficiencia Multiprocessing: {metricas['eficiencia_multiprocessing']:.4f}")
    print(f"Eficiencia MPI: {metricas['eficiencia_mpi']:.4f}")
    print(f"Eficiencia PyCUDA: {metricas['eficiencia_pycuda']:.4f}")
    print(f"Tiempo Muerto Multiprocessing: {metricas['tiempo_muerto_multiprocessing']:.4f}")
    print(f"Tiempo Muerto MPI: {metricas['tiempo_muerto_mpi']:.4f}")
    print(f"Tiempo Muerto PyCUDA: {metricas['tiempo_muerto_pycuda']:.4f}")
    print(f"Tiempo Total Secuencial: {metricas['tiempo_total_secuencial']:.4f}")
    print(f"Tiempo Total Multiprocessing: {metricas['tiempo_total_multiprocessing']:.4f}")
    print(f"Tiempo Total MPI: {metricas['tiempo_total_mpi']:.4f}")
    print(f"Tiempo Total PyCUDA: {metricas['tiempo_total_pycuda']:.4f}")

if __name__ == "__main__":
    # Cargar los tiempos desde los archivos generados
    tiempos = cargar_tiempos()

    # Calcular las métricas de rendimiento
    metricas = calcular_metricas(tiempos)

    # Imprimir las métricas por consola
    imprimir_metricas(metricas)

    # Generar las gráficas
    generar_graficas(tiempos, metricas)
