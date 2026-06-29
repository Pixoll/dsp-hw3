#pragma once

// Punto de entrada de cada experimento, invocado desde src/main.cpp.
// Reciben el lote ya preprocesado en host: m imagenes x n componentes
// (n = ancho*alto, escala de grises), aplanado row-major y normalizado a [0,1].

// Experimento 1: implementacion tradicional en CUDA (Stream 0, copia sincrona).
int run_experiment1(const float *h_dataset, int m, int n);

// Experimento 2: orquestacion con CUDA Streams (pendiente, lo implementa el companiero).
// int run_exp2(const float *h_dataset, int m, int n, int num_streams);
