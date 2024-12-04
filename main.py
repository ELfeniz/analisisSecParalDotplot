import argparse
from dotplot.fasta_utils import cargar_secuencias_fasta
from dotplot.dotplot_sequential import dotplot_secuencial
from dotplot.dotplot_multiprocessing import dotplot_multiprocessing
from dotplot.dotplot_mpi import dotplot_mpi
from dotplot.dotplot_pycuda import dotplot_pycuda_memmap as dotplot_pycuda

from dotplot.image_filter import filtrar_diagonales
import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generar un dotplot de dos secuencias.")
    parser.add_argument("--file1", required=True, help="Archivo FASTA 1.")
    parser.add_argument("--file2", required=True, help="Archivo FASTA 2.")
    parser.add_argument("--output", required=True, help="Nombre del archivo de salida.")
    parser.add_argument("--mode", choices=["secuencial", "multiprocessing", "mpi", "pycuda"],
                        required=True, help="Modo de ejecución.")

    args = parser.parse_args()

    secuencia1, secuencia2 = cargar_secuencias_fasta(args.file1, args.file2)

    if args.mode == "secuencial":
        dotplot = dotplot_secuencial(secuencia1, secuencia2)
    elif args.mode == "multiprocessing":
        dotplot = dotplot_multiprocessing(secuencia1, secuencia2)
    elif args.mode == "mpi":
        dotplot = dotplot_mpi(secuencia1, secuencia2)
    elif args.mode == "pycuda":
        dotplot = dotplot_pycuda(secuencia1, secuencia2)

    generar_dotplot(dotplot, args.output)


    ## python main.py --file1=data/elemento1.fasta --file2=data/elemento2.fasta --output=image/dotplot.png --mode=secuencial