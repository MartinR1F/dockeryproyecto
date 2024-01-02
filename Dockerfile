# Usa una imagen oficial de Python como base
FROM python:3.8

# Establece el directorio de trabajo en el contenedor
WORKDIR /usr/src/app

# Copia el archivo de requisitos e instala las dependencias
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código fuente del proyecto al contenedor
COPY . .

# Define el comando para ejecutar la aplicación
CMD ["python", "./proyecto_ML.py"]
