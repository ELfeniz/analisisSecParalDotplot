import os
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from tqdm import tqdm

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

# Función para convertir secuencias a números
def convertir_secuencia_a_numeros(secuencia):
    mapa = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    return np.array([mapa[base] for base in secuencia], dtype=np.byte)

# Función principal para generar el dotplot y guardar en memmap
def dotplot_pycuda_memmap(secuencia1, secuencia2, output_file='dotplot_memmap_pycuda.dat', bloque_tamano=500, subbloque_tamano=100):
    sec1_numerica = convertir_secuencia_a_numeros(secuencia1)
    sec2_numerica = convertir_secuencia_a_numeros(secuencia2)
    len1, len2 = len(secuencia1), len(secuencia2)

    dotplot = np.memmap(output_file, dtype=np.int32, mode='w+', shape=(len1, len2))
    block_size = (32, 32, 1)

    total_bloques = (len1 // bloque_tamano + (1 if len1 % bloque_tamano != 0 else 0))
    with tqdm(total=total_bloques, desc="Calculando Dotplot", unit="bloques") as pbar:
        for i in range(0, len1, bloque_tamano):
            bloque1 = sec1_numerica[i:i + bloque_tamano]
            len_bloque1 = len(bloque1)

            for j in range(0, len2, subbloque_tamano):
                subbloque2 = sec2_numerica[j:j + subbloque_tamano]
                len_subbloque2 = len(subbloque2)

                sec1_gpu = gpuarray.to_gpu(bloque1)
                sec2_gpu = gpuarray.to_gpu(subbloque2)

                dotplot_gpu = gpuarray.zeros((len_bloque1, len_subbloque2), dtype=np.int32)
                grid_size = (
                    (len_bloque1 + block_size[0] - 1) // block_size[0],
                    (len_subbloque2 + block_size[1] - 1) // block_size[1],
                )

                func = mod.get_function("generar_dotplot")
                func(
                    sec1_gpu, sec2_gpu, dotplot_gpu,
                    np.int32(len_bloque1), np.int32(len_subbloque2),
                    block=block_size, grid=grid_size,
                )

                dotplot_bloque = dotplot_gpu.get()
                dotplot[i:i + len_bloque1, j:j + len_subbloque2] = dotplot_bloque

                del sec1_gpu, sec2_gpu, dotplot_gpu

            pbar.update(1)

    dotplot.flush()
    return dotplot
