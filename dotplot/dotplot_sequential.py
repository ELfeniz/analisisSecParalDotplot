import numpy as np
from tqdm import tqdm

def dotplot_secuencial_memmap(secuencia1, secuencia2, output_file='dotplot_memmap_secuencial.dat', bloque_tamano=1000):
    """
    Calcula el dotplot de dos secuencias en bloques y almacena el resultado en un archivo memmap para evitar problemas de memoria.
    Usa np.memmap para no cargar todo el dotplot en memoria a la vez.

    Args:
        secuencia1 (str): Primera secuencia.
        secuencia2 (str): Segunda secuencia.
        output_file (str): Ruta del archivo memmap donde se guardará el dotplot.
        bloque_tamano (int): Tamaño del bloque para procesar las secuencias.

    Returns:
        np.ndarray: Matriz del dotplot almacenada en el archivo memmap.
    """
    len1, len2 = len(secuencia1), len(secuencia2)

    # Crear un archivo memmap para almacenar el dotplot
    dotplot = np.memmap(output_file, dtype=np.int32, mode='w+', shape=(len1, len2))

    # Calcular el número total de bloques para la barra de progreso
    total_bloques = (len1 // bloque_tamano + (1 if len1 % bloque_tamano != 0 else 0)) * \
                    (len2 // bloque_tamano + (1 if len2 % bloque_tamano != 0 else 0))

    # Procesar en bloques con barra de progreso
    with tqdm(total=total_bloques, desc="Calculando Dotplot", unit="bloques") as pbar:
        for i in range(0, len1, bloque_tamano):
            for j in range(0, len2, bloque_tamano):
                # Definir los límites del bloque
                bloque1 = secuencia1[i:i + bloque_tamano]
                bloque2 = secuencia2[j:j + bloque_tamano]

                # Crear una submatriz para el bloque actual
                submatriz = np.zeros((len(bloque1), len(bloque2)), dtype=np.int32)

                # Calcular el dotplot para el bloque actual
                for bi, base1 in enumerate(bloque1):
                    for bj, base2 in enumerate(bloque2):
                        submatriz[bi, bj] = 1 if base1 == base2 else 0

                # Insertar la submatriz en la matriz principal (memmap)
                dotplot[i:i + len(bloque1), j:j + len(bloque2)] = submatriz

                # Actualizar la barra de progreso
                pbar.update(1)

    return dotplot

