// ============================================================================
//  loader.h - Carga de imagenes (independiente del motor de computo CUDA).
//  Lo usa el Exp1 y se puede usar en el Exp2 en caso de ser necesario.
// ============================================================================
#ifndef LOADER_H
#define LOADER_H

#ifdef __cplusplus
extern "C"
{
#endif

    // Lee todos los .png de 'dir' (orden alfabetico), los pasa a escala de grises,
    // los redimensiona a width x height (interpolacion lineal) y los normaliza a
    // [0,1]. Construye un arreglo float lineal row-major de tamano m*n y lo
    // devuelve por *out_data (lo reserva con malloc; el llamador debe free()).
    //
    //   dir      : carpeta con las imagenes .png
    //   width,height : dimensiones destino (channels = 1, gris)
    //   max_m    : tope de imagenes a cargar (<=0 = todas las encontradas)
    //   out_data : (salida) puntero al arreglo float[m*n] reservado
    //   out_n    : (salida) n = width*height
    //
    // Retorna m (numero de imagenes cargadas), o 0 si hubo error.
    int cargarDataset(const char *dir, int width, int height, int max_m,
                      float **out_data, int *out_n);

#ifdef __cplusplus
}
#endif

#endif // LOADER_H
