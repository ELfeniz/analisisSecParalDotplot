import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def comparar_indices(args):
    """
    Compara un índice de la secuencia1 con todos los índices de la secuencia2.
    """
    i, secuencia1, secuencia2, dotplot_memmap = args
    # Comparar la secuencia1[i] con toda la secuencia2
    dotplot_memmap[i, :] = [1 if secuencia1[i] == secuencia2[j] else 0 for j in range(len(secuencia2))]
    return i

def dotplot_multiprocessing_memmap(secuencia1, secuencia2, output_file='dotplot_memmap_multiprocessing.dat', num_procesos=mp.cpu_count(), bloque_tamano=100):
    """
    Calcula el dotplot de dos secuencias utilizando multiprocessing y np.memmap,
    procesando en bloques para evitar saturar la memoria.

    Args:
        secuencia1 (str): Primera secuencia.
        secuencia2 (str): Segunda secuencia.
        output_file (str): Ruta del archivo memmap donde se guardará el dotplot.
        num_procesos (int): Número de procesos a utilizar.
        bloque_tamano (int): Tamaño del bloque para procesar las secuencias.

    Returns:
        np.ndarray: Matriz del dotplot almacenada en el archivo memmap.
    """
    len1, len2 = len(secuencia1), len(secuencia2)

    # Crear un archivo memmap para almacenar el dotplot
    dotplot_memmap = np.memmap(output_file, dtype=np.int32, mode='w+', shape=(len1, len2))

    # Dividir la secuencia1 en bloques para el procesamiento paralelo
    bloques = [(i, secuencia1, secuencia2, dotplot_memmap) for i in range(len(secuencia1))]

    # Crear la barra de progreso
    with tqdm(total=len(bloques), desc="Calculando Dotplot", unit="líneas") as pbar:
        # Usar multiprocessing para procesar los bloques
        with mp.Pool(num_procesos) as pool:
            # Usar imap para el procesamiento paralelo, actualizando la barra de progreso
            for _ in pool.imap(comparar_indices, bloques, chunksize=bloque_tamano):
                pbar.update(1)  # Actualizar la barra de progreso

    # Después de procesar, se debe hacer un flush para asegurar que los datos se escriben en el disco
    dotplot_memmap.flush()

    return dotplot_memmap