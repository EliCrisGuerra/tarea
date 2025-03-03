import csv
import numpy as np

def calcular_polinomios(x_valores, *polinomios):
    """
    Función para calcular valores de múltiples polinomios de la forma ax^n
    
    Parámetros:
    x_valores: Lista o array con los valores de x para evaluar los polinomios
    *polinomios: Tuplas de la forma (a, n) donde:
                 a: coeficiente del polinomio
                 n: exponente del polinomio
                 
    Retorna:
    Un diccionario donde las claves son strings que representan cada polinomio
    y los valores son arrays con los resultados evaluados en x_valores
    """
    resultados = {}
    
    # Calcular cada polinomio
    for a, n in polinomios:
        # Calcular y = ax^n para todos los valores de x
        y = a * (np.array(x_valores) ** n)
        
        # Guardar los resultados en el diccionario con un nombre descriptivo
        nombre_polinomio = f"{a}x^{n}"
        resultados[nombre_polinomio] = y
    
    return resultados

# Ejemplo de uso
if __name__ == "__main__":
    # Definir un rango de valores x (puede ser cualquier lista de valores)
    x = np.linspace(-5, 5, 11)  # 11 valores equidistantes entre -5 y 5
    
    # Calcular varios polinomios
    resultado = calcular_polinomios(x, (2, 3), (-0.5, 2), (5, 1), (1, 4))
    
    # Mostrar los resultados
    print("Valores de x:", x)
    print("\nResultados:")
    for polinomio, valores in resultado.items():
        print(f"\nPolinomio {polinomio}:")
        for i, (x_val, y_val) in enumerate(zip(x, valores)):
            print(f"  Para x = {x_val:.1f}, y = {y_val:.2f}")


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






 
