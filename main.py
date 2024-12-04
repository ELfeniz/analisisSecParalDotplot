import argparse
from dotplot.fasta_utils import cargar_secuencias_fasta
from dotplot.dotplot_sequential import dotplot_secuencial_memmap
from dotplot.dotplot_multiprocessing import dotplot_multiprocessing_memmap
from dotplot.dotplot_mpi import dotplot_mpi_memmap
from dotplot.dotplot_pycuda import dotplot_pycuda_memmap as dotplot_pycuda
import pycuda.gpuarray as gpuarray
from tqdm import tqdm
import numpy as np
from pycuda.compiler import SourceModule
from dotplot.image_filter import filtrar_diagonales
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
import os


# Código CUDA para el dotplot y detección de diagonales
mod = SourceModule("""
__global__ void generar_dotplot(char *sec1, char *sec2, int *dotplot, int len1, int len2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < len1 && j < len2) {
        dotplot[i * len2 + j] = (sec1[i] == sec2[j]) ? 1 : 0;
    }
}

__global__ void detectar_diagonales(int *dotplot, int *resultado, int len1, int len2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < len1 - 1 && j < len2 - 1) {
        if (dotplot[i * len2 + j] == 1 && dotplot[(i + 1) * len2 + (j + 1)] == 1) {
            resultado[i * len2 + j] = 1;
        } else {
            resultado[i * len2 + j] = 0;
        }
    }
}
""")

# Crear la carpeta 'tiempos' si no existe
if not os.path.exists('tiempos'):
    os.makedirs('tiempos')

def guardar_tiempos(nombre_archivo, tiempos):
    # Definir la ruta completa del archivo
    archivo_tiempo = f'tiempos/{nombre_archivo}.txt'
    
    # Guardar los tiempos en el archivo (sobrescribirá en cada ejecución)
    with open(archivo_tiempo, 'w') as file:
        for key, value in tiempos.items():
            file.write(f"{key}: {value:.2f} segundos\n")
    print(f"Tiempos guardados en {archivo_tiempo}")

def generar_dotplot(dotplot, nombre_salida, bloque_tamano=500):
    """
    Genera un dotplot visualizando la matriz por bloques para evitar problemas de memoria.

    Args:
        dotplot (np.ndarray): Matriz del dotplot.
        nombre_salida (str): Nombre del archivo de salida.
        bloque_tamano (int): Tamaño del bloque para la visualización.
    """
    filas, columnas = dotplot.shape
    plt.figure(figsize=(10, 10))

    # Visualizar por bloques
    for i in range(0, filas, bloque_tamano):
        for j in range(0, columnas, bloque_tamano):
            submatriz = dotplot[i:i+bloque_tamano, j:j+bloque_tamano]
            plt.imshow(submatriz, cmap='Greys', aspect='auto', extent=(j, j+submatriz.shape[1], i+submatriz.shape[0], i))

    plt.xlabel("Secuencia 2")
    plt.ylabel("Secuencia 1")
    plt.savefig(nombre_salida)
    plt.close()

# Función para filtrar diagonales en la matriz
def filtrar_diagonales(dotplot_memmap, output_file, bloque_tamano=500):
    len1, len2 = dotplot_memmap.shape
    resultado_memmap = np.memmap(output_file, dtype=np.int32, mode='w+', shape=(len1, len2))
    resultado_memmap[:] = 0

    block_size = (32, 32, 1)
    total_bloques = (len1 // bloque_tamano + (1 if len1 % bloque_tamano != 0 else 0))
    with tqdm(total=total_bloques, desc="Filtrando diagonales", unit="bloques") as pbar:
        for i in range(0, len1, bloque_tamano):
            bloque_len1 = min(bloque_tamano, len1 - i)
            bloque = dotplot_memmap[i:i + bloque_len1, :]

            dotplot_gpu = gpuarray.to_gpu(bloque.flatten())
            resultado_gpu = gpuarray.zeros((bloque_len1, len2), dtype=np.int32)

            grid_size = (
                (bloque_len1 + block_size[0] - 1) // block_size[0],
                (len2 + block_size[1] - 1) // block_size[1],
            )

            func = mod.get_function("detectar_diagonales")
            func(
                dotplot_gpu, resultado_gpu,
                np.int32(bloque_len1), np.int32(len2),
                block=block_size, grid=grid_size,
            )

            resultado_bloque = resultado_gpu.get().reshape(bloque_len1, len2)
            resultado_memmap[i:i + bloque_len1, :] = resultado_bloque

            del dotplot_gpu, resultado_gpu
            pbar.update(1)

    resultado_memmap.flush()
    return resultado_memmap


def procesar_bloque_cpu(i, dotplot_memmap, len1, len2, bloque_tamano):
    """
    Procesa un bloque de la matriz para filtrar las diagonales en la versión CPU.
    Similar a la lógica de la GPU, se verifica la diagonal de cada bloque.
    """
    bloque_len1 = min(bloque_tamano, len1 - i)
    bloque = dotplot_memmap[i:i + bloque_len1, :]

    resultado_bloque = np.zeros_like(bloque, dtype=np.int32)

    # Aplicar el mismo filtro de diagonales que la GPU
    for idx1 in range(bloque_len1 - 1):  # Evitar indexación fuera de los límites
        for idx2 in range(len2 - 1):
            # Verificamos la diagonal: si hay coincidencia de 1s
            if bloque[idx1, idx2] == 1 and bloque[idx1 + 1, idx2 + 1] == 1:
                resultado_bloque[idx1, idx2] = 1

    return i, resultado_bloque

def filtrar_diagonales_cpu(dotplot_memmap, output_file, bloque_tamano=500):
    """
    Filtra las diagonales de la matriz dotplot utilizando procesamiento en paralelo en la CPU.
    Guarda el resultado en un archivo memmap.
    La lógica de filtrado es similar a la de la GPU para obtener resultados consistentes.
    """
    len1, len2 = dotplot_memmap.shape
    resultado_memmap = np.memmap(output_file, dtype=np.int32, mode='w+', shape=(len1, len2))
    resultado_memmap[:] = 0  # Inicializamos la matriz resultado en cero

    # Dividimos la tarea en bloques
    total_bloques = (len1 // bloque_tamano + (1 if len1 % bloque_tamano != 0 else 0))

    # Usamos multiprocessing para paralelizar el procesamiento de los bloques
    with mp.Pool(mp.cpu_count()) as pool:
        # Aplicamos procesamiento en paralelo para cada bloque
        resultados = list(tqdm(pool.starmap(procesar_bloque_cpu, 
                                           [(i, dotplot_memmap, len1, len2, bloque_tamano) for i in range(0, len1, bloque_tamano)]),
                              total=total_bloques, desc="Filtrando diagonales", unit="bloques"))

    # Almacenamos los resultados en el archivo memmap
    for i, resultado_bloque in resultados:
        bloque_len1 = min(bloque_tamano, len1 - i)
        resultado_memmap[i:i + bloque_len1, :] = resultado_bloque

    resultado_memmap.flush()
    return resultado_memmap

# Función para visualizar las matrices usando matplotlib
def visualizar_memmap_por_bloques(archivo_memmap, shape, nombre_salida, bloque_tamano=500):
    """
    Visualiza un archivo memmap (dotplot o diagonales) en bloques.

    Args:
        archivo_memmap (str): Ruta del archivo memmap.
        shape (tuple): Dimensiones de la matriz (filas, columnas).
        nombre_salida (str): Nombre del archivo de salida para la imagen.
        bloque_tamano (int): Tamaño de los bloques para la visualización.
    """
    # Cargar el archivo memmap
    matriz_memmap = np.memmap(archivo_memmap, dtype=np.int32, mode='r', shape=shape)
    filas, columnas = shape

    # Crear la figura
    plt.figure(figsize=(10, 10))

    # Visualizar por bloques
    for i in range(0, filas, bloque_tamano):
        for j in range(0, columnas, bloque_tamano):
            submatriz = matriz_memmap[i:i+bloque_tamano, j:j+bloque_tamano]
            plt.imshow(submatriz, cmap='Greys', aspect='auto', extent=(j, j+submatriz.shape[1], i+submatriz.shape[0], i))

    # Configurar ejes y guardar la imagen
    plt.xlabel("Secuencia 2")
    plt.ylabel("Secuencia 1")
    plt.savefig(nombre_salida)
    plt.close()

# Función para visualizar las matrices usando matplotlib
def visualizar_memmap_por_bloques2(archivo_memmap, shape, nombre_salida, bloque_tamano=500):
    """
    Visualiza un archivo memmap (dotplot o diagonales) en bloques.

    Args:
        archivo_memmap (str): Ruta del archivo memmap.
        shape (tuple): Dimensiones de la matriz (filas, columnas).
        nombre_salida (str): Nombre del archivo de salida para la imagen.
        bloque_tamano (int): Tamaño de los bloques para la visualización.
    """
    # Cargar el archivo memmap
    matriz_memmap = np.memmap(archivo_memmap, dtype=np.int8, mode='r', shape=shape)
    filas, columnas = shape

    # Crear la figura
    plt.figure(figsize=(10, 10))

    # Visualizar por bloques
    for i in range(0, filas, bloque_tamano):
        for j in range(0, columnas, bloque_tamano):
            submatriz = matriz_memmap[i:i+bloque_tamano, j:j+bloque_tamano]
            plt.imshow(submatriz, cmap='Greys', aspect='auto', extent=(j, j+submatriz.shape[1], i+submatriz.shape[0], i))

    # Configurar ejes y guardar la imagen
    plt.xlabel("Secuencia 2")
    plt.ylabel("Secuencia 1")
    plt.savefig(nombre_salida)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar un dotplot de dos secuencias.")
    parser.add_argument("--file1", required=True, help="Archivo FASTA 1.")
    parser.add_argument("--file2", required=True, help="Archivo FASTA 2.")
    parser.add_argument("--output", required=True, help="Nombre del archivo de salida.")
    parser.add_argument("--mode", choices=["secuencial", "multiprocessing", "mpi", "pycuda"],
                        required=True, help="Modo de ejecución.")

    args = parser.parse_args()

    # Cargar las secuencias y medir el tiempo
    start_time = time.time()
    secuencia1, secuencia2 = cargar_secuencias_fasta(args.file1, args.file2)
    end_time = time.time()
    tiempo_carga_datos = end_time - start_time

    temporal_resultado = {
        "Tiempo carga datos": tiempo_carga_datos,
    }

    if args.mode == "secuencial":
        # Ejecución secuencial
        start_time = time.time()
        dotplot = dotplot_secuencial_memmap(secuencia1, secuencia2)
        end_time = time.time()
        tiempo_calculo_dotplot = end_time - start_time

        start_time = time.time()
        diagonales = filtrar_diagonales_cpu(dotplot, output_file='diagonales_memmap_secuencial.dat')
        end_time = time.time()
        tiempo_filtrado_diagonales = end_time - start_time

        start_time = time.time()
        visualizar_memmap_por_bloques2('dotplot_memmap_secuencial.dat', (len(secuencia1), len(secuencia2)), f'{args.output}')
        aux = args.output[:-4]
        # Visualización de las diagonales desde el archivo memmap
        visualizar_memmap_por_bloques2('diagonales_memmap_secuencial.dat', (len(secuencia1), len(secuencia2)), f'{aux}_diagonales_.png')
        end_time = time.time()
        tiempo_Graficas = end_time - start_time

        temporal_resultado.update({
            "Tiempo calculo dotplot secuencial": tiempo_calculo_dotplot,
            "Tiempo filtrado diagonales secuencial": tiempo_filtrado_diagonales,
            "Tiempo Graficas secuencial": tiempo_Graficas
        })

        guardar_tiempos("tiempo_secuencial", temporal_resultado)

    elif args.mode == "multiprocessing":
        # Ejecución multiprocessing
        start_time = time.time()
        dotplot = dotplot_multiprocessing_memmap(secuencia1, secuencia2)
        end_time = time.time()
        tiempo_calculo_dotplot = end_time - start_time

        start_time = time.time()
        diagonales = filtrar_diagonales_cpu(dotplot, output_file='diagonales_memmap_multiprocessing.dat')
        end_time = time.time()
        tiempo_filtrado_diagonales = end_time - start_time

        start_time = time.time()
        visualizar_memmap_por_bloques2('dotplot_memmap_multiprocessing.dat', (len(secuencia1), len(secuencia2)), f'{args.output}')
        aux = args.output[:-4]
        # Visualización de las diagonales desde el archivo memmap
        visualizar_memmap_por_bloques2('diagonales_memmap_multiprocessing.dat', (len(secuencia1), len(secuencia2)), f'{aux}_diagonales_.png')
        end_time = time.time()
        tiempo_Graficas = end_time - start_time

        temporal_resultado.update({
            "Tiempo calculo dotplot multiprocessing": tiempo_calculo_dotplot,
            "Tiempo filtrado diagonales multiprocessing": tiempo_filtrado_diagonales,
            "Tiempo Graficas multiprocessing": tiempo_Graficas
        })

        guardar_tiempos("tiempo_multiprocessing", temporal_resultado)

    elif args.mode == "mpi":
        # Ejecución MPI
        start_time = time.time()
        dotplot = dotplot_mpi_memmap(secuencia1, secuencia2)
        end_time = time.time()
        tiempo_calculo_dotplot = end_time - start_time

        start_time = time.time()
        diagonales = filtrar_diagonales_cpu(dotplot, output_file='diagonales_memmap_mpi.dat')
        end_time = time.time()
        tiempo_filtrado_diagonales = end_time - start_time

        start_time = time.time()
        visualizar_memmap_por_bloques2('dotplot_memmap_mpi.dat', (len(secuencia1), len(secuencia2)), f'{args.output}')
        aux = args.output[:-4]
        # Visualización de las diagonales desde el archivo memmap
        visualizar_memmap_por_bloques2('diagonales_memmap_mpi.dat', (len(secuencia1), len(secuencia2)), f'{aux}_diagonales_.png')
        end_time = time.time()
        tiempo_Graficas = end_time - start_time

        temporal_resultado.update({
            "Tiempo calculo dotplot mpi": tiempo_calculo_dotplot,
            "Tiempo filtrado diagonales mpi": tiempo_filtrado_diagonales,
            "Tiempo Graficas mpi": tiempo_Graficas
        })

        guardar_tiempos("tiempo_mpi", temporal_resultado)

    elif args.mode == "pycuda":
        # Ejecución PyCUDA
        start_time = time.time()
        dotplot = dotplot_pycuda(secuencia1, secuencia2)
        end_time = time.time()
        tiempo_calculo_dotplot = end_time - start_time

        start_time = time.time()
        diagonales = filtrar_diagonales(dotplot, output_file='diagonales_memmap_pycuda.dat')
        end_time = time.time()
        tiempo_filtrado_diagonales = end_time - start_time

        start_time = time.time()
        visualizar_memmap_por_bloques('dotplot_memmap_pycuda.dat', (len(secuencia1), len(secuencia2)), f'{args.output}')
        aux = args.output[:-4]
        # Visualización de las diagonales desde el archivo memmap
        visualizar_memmap_por_bloques('diagonales_memmap_pycuda.dat', (len(secuencia1), len(secuencia2)), f'{aux}_diagonales_.png')
        end_time = time.time()
        tiempo_Graficas = end_time - start_time

        temporal_resultado.update({
            "Tiempo calculo dotplot pycuda": tiempo_calculo_dotplot,
            "Tiempo filtrado diagonales pycuda": tiempo_filtrado_diagonales,
            "Tiempo Graficas pycuda": tiempo_Graficas
        })

        guardar_tiempos("tiempo_pycuda", temporal_resultado)

    # Visualización de resultados
    print("Proceso completado: dotplot y diagonales generados.")

    ## python main.py --file1=data/elemento1.fasta --file2=data/elemento2.fasta --output=image/dotplot.png --mode=secuencial