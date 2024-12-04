from mpi4py import MPI
import numpy as np
from tqdm import tqdm

def dotplot_mpi_memmap(secuencia1, secuencia2, output_file='dotplot_memmap_mpi.dat', bloque_tamano=100):
    """
    Calcula el dotplot de dos secuencias utilizando MPI y guarda directamente los resultados en un archivo memmap.

    Args:
        secuencia1 (str): Primera secuencia.
        secuencia2 (str): Segunda secuencia.
        output_file (str): Ruta del archivo memmap donde se guardará el dotplot.
        bloque_tamano (int): Tamaño del bloque para procesar las secuencias.

    Returns:
        np.memmap: Objeto memmap que representa el dotplot (solo en el proceso root).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    len1, len2 = len(secuencia1), len(secuencia2)

    # Crear archivo memmap solo en el proceso root
    if rank == 0:
        dotplot = np.memmap(output_file, dtype=np.int8, mode='w+', shape=(len1, len2))
    else:
        dotplot = None

    # Dividir la secuencia1 entre los procesos
    chunk_size = len1 // size
    inicio = rank * chunk_size
    fin = len1 if rank == size - 1 else (rank + 1) * chunk_size

    # Crear la matriz local para el dotplot
    dotplot_local = np.zeros((fin - inicio, len2), dtype=np.int8)

    # Barra de progreso para cada proceso
    total_bloques = (fin - inicio) // bloque_tamano + (1 if (fin - inicio) % bloque_tamano != 0 else 0)
    with tqdm(total=total_bloques, desc=f"Proceso {rank}", unit="bloques", position=rank) as pbar:
        for i in range(inicio, fin, bloque_tamano):
            bloque1 = secuencia1[i:i + bloque_tamano]
            for j in range(0, len2, bloque_tamano):
                bloque2 = secuencia2[j:j + bloque_tamano]

                # Crear una submatriz para el bloque actual
                submatriz = np.zeros((len(bloque1), len(bloque2)), dtype=np.int8)

                # Calcular el dotplot para el bloque actual
                for bi, base1 in enumerate(bloque1):
                    for bj, base2 in enumerate(bloque2):
                        submatriz[bi, bj] = 1 if base1 == base2 else 0

                # Determinar los índices finales para evitar desbordamientos
                end_i = min(i - inicio + len(bloque1), dotplot_local.shape[0])
                end_j = min(j + len(bloque2), dotplot_local.shape[1])

                # Insertar la submatriz en la matriz local
                dotplot_local[i - inicio:end_i, j:end_j] = submatriz

            # Actualizar la barra de progreso
            pbar.update(1)

    # Reunir los resultados en el archivo memmap en el proceso root
    if rank == 0:
        for r in range(size):
            if r == rank:
                dotplot[inicio:fin, :] = dotplot_local
            else:
                local_data = np.empty_like(dotplot_local)
                comm.Recv(local_data, source=r, tag=r)
                start_idx = r * chunk_size
                end_idx = len1 if r == size - 1 else (r + 1) * chunk_size
                dotplot[start_idx:end_idx, :] = local_data
    else:
        comm.Send(dotplot_local, dest=0, tag=rank)

    # Sincronizar procesos antes de finalizar
    comm.Barrier()

    # Solo el proceso root devuelve el archivo memmap
    if rank == 0:
        dotplot.flush()
        return dotplot
    return None
