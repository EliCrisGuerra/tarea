import numpy as np
import matplotlib.pyplot as plt
import csv

def graficar_polinomios(*polinomios):
    """
    Función para graficar múltiples polinomios de la forma ax^n
    
    Parámetros:
    *polinomios: Tuplas de la forma (a, n) donde:
                 a: coeficiente del polinomio
                 n: exponente del polinomio
    """
    # Definir el rango x
    x = np.linspace(-10, 10, 1000)
    
    # Crear la figura y el eje
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Graficar cada polinomio
    for i, (a, n) in enumerate(polinomios):
        y = a * (x ** n)
        ax.plot(x, y, label=f"{a}x^{n}")
    
    # Configurar la gráfica
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Gráfica de polinomios de la forma ax^n')
    ax.legend()
    
    # Mostrar la gráfica
    plt.show()

# Ejemplo de uso
# Polinomio 1: 2x^3
# Polinomio 2: -0.5x^2
# Polinomio 3: 5x^1
# Polinomio 4: 1x^4
if __name__ == "__main__":
    graficar_polinomios((2, 3), (-0.5, 2), (5, 1), (1, 4))

#Escritura de Datos en un Archivo CSV. Segundo punto:



frutas = [
    ["Manzana", "Roja", "Dulce"],
    ["Plátano", "Amarillo", "Dulce"],
    ["Lima", "Verde", "Ácida"]
]

# Abriendo el archivo en modo escritura ('w'), y asegurándonos de que se cree si no existe
with open("frutas.csv", mode="w", newline="") as archivo_csv:
    # Crear un escritor de CSV
    escritor_csv = csv.writer(archivo_csv)

    # Escribir cada sublista en una fila del archivo CSV
    for fruta in frutas:
        escritor_csv.writerow(fruta)

print("Datos guardados correctamente en 'frutas.csv'")


#Tercer punto Lista para almacenar los datos leídos del archivo
datos_frutas = []

# Abrir el archivo CSV en modo lectura
with open("frutas.csv", mode="r") as archivo_csv:
    # Crear un lector de CSV
    lector_csv = csv.reader(archivo_csv)

    # Leer las filas del archivo y almacenarlas en la lista datos_frutas
    for fila in lector_csv:
        datos_frutas.append(fila)

# Imprimir la lista de listas
print(datos_frutas)
