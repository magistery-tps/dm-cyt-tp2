# UBA - Maestria en Explotación de Datos y Descubrimiento de Conocimiento - Datamining en Ciencia y Tecnologia


## Trabajos Prácticos 2

* [Consigna]()
* **Notebooks**
  * [Preprocesamiento](https://github.com/magistery-tps/dm-cyt-tp2/blob/main/notebooks/preprocessing.ipynb) 

## Pre-Requisitos

* [git](https://git-scm.com/downloads)
* [anaconda](https://www.anaconda.com/products/individual) / [minconda](https://docs.conda.io/en/latest/miniconda.html)

## Comenzando

### Video

[Paso a Paso (Windows)](https://www.youtube.com/watch?v=O8YXuHNdIIk)

### Pasos

**Paso 1**: Descargar el repositorio.

```bash
$ git clone https://github.com/magistery-tps/dm-cyt-tp2.git
$ cd dm-cyt-tp
```

**Paso 2**: Crear environment de dependencias para el proyecto (Parado en el directorio del proyecto).

```bash
$ conda env create -f environment.yml
```

**Paso 3**: Activamos el entorno donde se encuentran instaladas las dependencias del proyecto.

```bash
$ conda activate dm-cyt-tp2
```

**Paso 4**: Descomprimir el dataset:

```bash
$ cd datasets
$ unzip strength.SWOW-EN.R1.csv.zip
```

**Paso 5**: Sobre el directorio del proyecto levantamos jupyter lab.

```bash
$ jupyter lab

Jupyter Notebook 6.1.4 is running at:
http://localhost:8888/?token=45efe99607fa6......
```

**Paso 6**: Ir a http://localhost:8888.... como se indica en la consola.

## Tema Material Darker para Jupyter Lab

**Paso 1**: Instalar tema.

```bash
$ jupyter labextension install @oriolmirosa/jupyterlab_materialdarker
```

**Paso 2**: Reiniciar Jupyter Lab

