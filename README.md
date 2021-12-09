# UBA - Maestria en Explotación de Datos y Descubrimiento de Conocimiento - Datamining en Ciencia y Tecnologia


## Trabajos Prácticos 2

* [Enunciado](https://github.com/magistery-tps/dm-cyt-tp2/blob/main/docs/Enunciado.pdf)
* **Notebooks**
    * [Preprocessing](https://github.com/magistery-tps/dm-cyt-tp2/blob/main/notebooks/preprocessing.ipynb) 
    * [Analisys](https://github.com/magistery-tps/dm-cyt-tp2/blob/main/notebooks/analisys.ipynb) 

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

**Paso 5**: Descargar y descomprimir word embedding:

```bash
$ wget https://nlp.stanford.edu/data/glove.840B.300d.zip
$ unzip glove.840B.300d.zip
```
**Nota:** En algunas ocasiones el servidor de stanford no admite descargas. Como alternativa se puede  descargar desde [kaggle](https://www.kaggle.com/takuok/glove840b300dtxt).

**Paso 6**: Sobre el directorio del proyecto levantamos jupyter lab.

```bash
$ jupyter lab

Jupyter Notebook 6.1.4 is running at:
http://localhost:8888/?token=45efe99607fa6......
```

**Paso 7**: Ir a http://localhost:8888.... como se indica en la consola.


## Agregar una nueva depedencia

**Paso 1**: Agregar la nueva depedencia a ´environment.yml´

Conda tiene sus propios repositorios de paquetes pero en caso de tener algun problema siempre se puede usar los paquetes de pip.

```yaml
...
dependencies:
  - SOY_UN_PAQUETE_DE_CONDA
  - SOY_OTRO_PAQUETE_DE_CONDA
  - pip:
    - SOY_UN_PAQUETE_DE_PIP
    - SOY_OTRO_PAQUETE_DE_PIP
...
```

**Paso 2**: Una ques que agregamos el nombre del nuevo pquetes en ´environment.yml´ resta instalarlo. Para esto debemos **actializar** nuestro environment con con lso cambio que realizamos en  ´environment.yml´ de la siguiente forma:

```bash
$ conda env update -f environment.yml
```
**Paso 3**: Finalmente si teniamos abierto **jupyter lab**, debemos reinifiar el kernel donde estemso corriendo nuestra notebook para poder cargar la nueva libreria.

![image](https://user-images.githubusercontent.com/962480/145253730-365cb56b-ae26-41b0-a38d-41d505c9ea74.png)


## Tema Material Darker para Jupyter Lab

**Paso 1**: Instalar tema.

```bash
$ jupyter labextension install @oriolmirosa/jupyterlab_materialdarker
```

**Paso 2**: Reiniciar Jupyter Lab

