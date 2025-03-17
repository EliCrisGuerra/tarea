import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class Ejercicios:
    def __init__(self):
        # Crear un DataFrame vacío para almacenar los resultados
        self.df = pd.DataFrame(columns=["Ejercicio", "Descripción", "Resultado"])
        self.output_directory = "output" 
        os.makedirs(self.output_directory, exist_ok=True)

        
        
    # Ejercicio 1: Array de NumPy con valores desde 10 hasta 29
    def punto_1(self, inf=10, sup=30):
        """Genera un array de NumPy con valores desde 10 hasta 29."""
        array_10_29 = np.arange(inf, sup)
        
        # Guardar el resultado en el DataFrame - Corregido el método append
        nueva_fila = pd.DataFrame({
            "Ejercicio": [1],
            "Descripción": ["Array de NumPy con valores desde 10 hasta 29"],
            "Resultado": [str(array_10_29)]
        })
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)
        
        return array_10_29
    
    # Ejercicio 2: Suma de elementos en array 10x10 de unos
    def punto_2(self):
        """Calcula la suma de todos los elementos en un array de NumPy de tamaño 10x10, lleno de unos."""
        ones_array = np.ones((10, 10))
        suma_total = np.sum(ones_array)
        
        nueva_fila = pd.DataFrame({
            "Ejercicio": [2],
            "Descripción": ["Suma de elementos en array 10x10 de unos"],
            "Resultado": [suma_total]
        })
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)
        
        return suma_total
    
    # Ejercicio 3: Producto elemento a elemento
    def punto_3(self, size=5, min_val=1, max_val=11, seed=42):
        """Dados dos arrays de tamaño 5, llenos de números aleatorios desde 1 hasta 10, 
        realiza un producto elemento a elemento."""
        np.random.seed(seed)
        array_a = np.random.randint(min_val, max_val, size=size)
        array_b = np.random.randint(min_val, max_val, size=size)
        producto = array_a * array_b
        
        nuevas_filas = pd.DataFrame({
            "Ejercicio": [3, "3 (datos)", "3 (datos)"],
            "Descripción": [
                "Producto elemento a elemento de dos arrays aleatorios",
                "Array A",
                "Array B"
            ],
            "Resultado": [
                str(producto),
                str(array_a),
                str(array_b)
            ]
        })
        self.df = pd.concat([self.df, nuevas_filas], ignore_index=True)
        
        return producto
    
    # Ejercicio 4: Matriz i+j y su inversa
    def punto_4(self, size=4):
        """Crea una matriz de 4x4, donde cada elemento es igual a i+j 
        (con i y j siendo el índice de fila y columna, respectivamente) y calcula su inversa."""
        # Crear la matriz i+j
        matriz = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                matriz[i, j] = i + j
        
        nueva_fila = pd.DataFrame({
            "Ejercicio": [4],
            "Descripción": ["Matriz donde cada elemento es i+j"],
            "Resultado": [str(matriz)]
        })
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)
        
        # Intentar calcular la inversa
        try:
            inversa = np.linalg.inv(matriz)
            nueva_fila_resultado = pd.DataFrame({
                "Ejercicio": ["4 (resultado)"],
                "Descripción": ["Inversa de la matriz"],
                "Resultado": [str(inversa)]
            })
            self.df = pd.concat([self.df, nueva_fila_resultado], ignore_index=True)
            return matriz, inversa
        except np.linalg.LinAlgError:
            nueva_fila_resultado = pd.DataFrame({
                "Ejercicio": ["4 (resultado)"],
                "Descripción": ["Inversa de la matriz"],
                "Resultado": ["La matriz no es invertible (determinante = 0)"]
            })
            self.df = pd.concat([self.df, nueva_fila_resultado], ignore_index=True)
            return matriz, None
    
    # Ejercicio 5: Valores máximo y mínimo
    def punto_5(self, size=100, seed=42):
        """Encuentra los valores máximo y mínimo en un array de 100 elementos aleatorios 
        y muestra sus índices."""
        np.random.seed(seed)
        array_grande = np.random.rand(size)
        valor_max = np.max(array_grande)
        valor_min = np.min(array_grande)
        indice_max = np.argmax(array_grande)
        indice_min = np.argmin(array_grande)
        
        nuevas_filas = pd.DataFrame({
            "Ejercicio": [5, "5 (continuación)"],
            "Descripción": [
                "Valor máximo y su índice",
                "Valor mínimo y su índice"
            ],
            "Resultado": [
                f"Valor: {valor_max}, Índice: {indice_max}",
                f"Valor: {valor_min}, Índice: {indice_min}"
            ]
        })
        self.df = pd.concat([self.df, nuevas_filas], ignore_index=True)
        
        return (valor_max, indice_max), (valor_min, indice_min)
    
    # Ejercicio 6: Suma de arrays con broadcasting
    def punto_6(self):
        """Crea un array de tamaño 3x1 y uno de 1x3, y súmalos utilizando broadcasting 
        para obtener un array de 3x3."""
        array_3x1 = np.array([[1], [2], [3]])  # Array vertical 3x1
        array_1x3 = np.array([[10, 20, 30]])   # Array horizontal 1x3
        
        # Usar broadcasting para sumar
        resultado = array_3x1 + array_1x3
        
        nuevas_filas = pd.DataFrame({
            "Ejercicio": [6, "6 (datos)", "6 (datos)"],
            "Descripción": [
                "Suma de arrays mediante broadcasting",
                "Array 3x1",
                "Array 1x3"
            ],
            "Resultado": [
                str(resultado),
                str(array_3x1),
                str(array_1x3)
            ]
        })
        self.df = pd.concat([self.df, nuevas_filas], ignore_index=True)
        
        return resultado
    
    # Ejercicio 7: Extracción de submatriz
    def punto_7(self):
        """De una matriz 5x5, extrae una submatriz 2x2 que comience en la segunda fila y columna."""
        # Crear matriz 5x5 con valores de ejemplo
        matriz_5x5 = np.arange(25).reshape(5, 5)
        
        # Extraer submatriz 2x2 desde la segunda fila y columna (índices 1,1 hasta 2,2)
        submatriz = matriz_5x5[1:3, 1:3]
        
        nuevas_filas = pd.DataFrame({
            "Ejercicio": [7, "7 (datos)"],
            "Descripción": [
                "Submatriz 2x2 desde la segunda fila y columna",
                "Matriz original 5x5"
            ],
            "Resultado": [
                str(submatriz),
                str(matriz_5x5)
            ]
        })
        self.df = pd.concat([self.df, nuevas_filas], ignore_index=True)
        
        return submatriz
    
    # Ejercicio 8: Indexado para cambiar valores
    def punto_8(self):
        """Crea un array de ceros de tamaño 10 y usa indexado para cambiar el valor 
        de los elementos en el rango de índices 3 a 6 a 5."""
        # Crear array de ceros
        array_ceros = np.zeros(10)
        
        # Cambiar valores en el rango de índices 3 a 6 (inclusive) a 5
        array_ceros[3:7] = 5
        
        nueva_fila = pd.DataFrame({
            "Ejercicio": [8],
            "Descripción": ["Array con valores cambiados mediante indexado"],
            "Resultado": [str(array_ceros)]
        })
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)
        
        return array_ceros
    
    # Ejercicio 9: Invertir orden de filas
    def punto_9(self):
        """Dada una matriz de 3x3, invierte el orden de sus filas."""
        # Crear matriz 3x3 de ejemplo
        matriz_3x3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Invertir orden de filas
        matriz_invertida = matriz_3x3[::-1]
        
        nuevas_filas = pd.DataFrame({
            "Ejercicio": [9, "9 (datos)"],
            "Descripción": [
                "Matriz con filas en orden invertido",
                "Matriz original 3x3"
            ],
            "Resultado": [
                str(matriz_invertida),
                str(matriz_3x3)
            ]
        })
        self.df = pd.concat([self.df, nuevas_filas], ignore_index=True)
        
        return matriz_invertida
    
    # Ejercicio 10: Selección condicional
    def punto_10(self, size=10, seed=42):
        """Dado un array de números aleatorios de tamaño 10, selecciona y muestra 
        solo aquellos que sean mayores a 0.5."""
        # Establecer semilla para reproducibilidad
        np.random.seed(seed)
        
        # Crear array de números aleatorios
        array_aleatorio = np.random.rand(size)
        
        # Seleccionar elementos mayores a 0.5
        seleccionados = array_aleatorio[array_aleatorio > 0.5]
        
        nuevas_filas = pd.DataFrame({
            "Ejercicio": [10, "10 (datos)"],
            "Descripción": [
                "Elementos mayores a 0.5",
                "Array aleatorio original"
            ],
            "Resultado": [
                str(seleccionados),
                str(array_aleatorio)
            ]
        })
        self.df = pd.concat([self.df, nuevas_filas], ignore_index=True)
        
        return seleccionados
    
    # 11 Genera dos arrays de tamaño 100 con números aleatorios y crea un gráfico de dispersión.


    def punto_11(self):
        plt.figure(figsize=(10, 8))
        np.random.seed(42)
        x_random = np.random.rand(100)
        y_random = np.random.rand(100)

        plt.scatter(x_random, y_random)
        plt.title("Gráfico 100 números aleatorios")
        plt.xlabel("Eje X")
        plt.ylabel("Eje Y")
        plt.grid(True, alpha=0.3)

        fig_path = os.path.join(self.output_directory, "1_scatter_random.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        stats = {
            "Mean X": np.mean(x_random),
            "Mean Y": np.mean(y_random),
            "Min X": np.min(x_random),
            "Max X": np.max(x_random),
            "Min Y": np.min(y_random),
            "Max Y": np.max(y_random)
        }

        nueva_fila = pd.DataFrame({
            "Ejercicio": [11],
            "Descripción": ["Gráfico de dispersión con 100 puntos aleatorios"],
            "Resultado": [str(stats)]
        })
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)



#Punto 12 Genera un gráfico de dispersión las variables 𝑥 y 𝑦 = 𝑠𝑖𝑛(𝑥)+ ruido Gaussiano. 
# Donde x es un array con númereos entre -2𝜋 𝑦 2𝜋. 
# Grafica también los puntos 𝑦 = 𝑠𝑖𝑛(𝑥) en el mismo plot

    def punto_12(self):
        plt.figure(figsize=(10, 6))
        
        # Generar datos
        x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        y_sin = np.sin(x)
        y_noisy = y_sin + np.random.normal(0, 0.1, size=x.shape)  # Añadir ruido Gaussiano
        
        # Graficar
        plt.scatter(x, y_noisy, label=r'$y = \sin(x) + \text{ruido}$', alpha=0.5, color='blue')
        plt.plot(x, y_sin, label=r'$y = \sin(x)$', color='red', linewidth=2)
        
        # Etiquetas y título
        plt.xlabel("Eje X")
        plt.ylabel("Eje Y")
        plt.title("Gráfico de Dispersión")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Guardar figura
        fig_path = os.path.join(self.output_directory, "2_scatter_sin_noise.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Guardar en DataFrame
        nueva_fila = pd.DataFrame({
            "Ejercicio": [12],
            "Descripción": ["Gráfico de dispersión de y = sin(x) con ruido Gaussiano"],
            "Resultado": [fig_path]
        })
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)

    #Ejercicio 13
    def punto_13(self):
        x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        y = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.cos(X) + np.sin(Y)

        plt.figure(figsize=(8, 6))
        plt.contour(X, Y, Z, levels=20, cmap="viridis")
        plt.colorbar(label="Valor de Z")
        plt.xlabel("Eje X")
        plt.ylabel("Eje Y")
        plt.title("Gráfico de Contorno de Z = cos(X) + sin(Y)")

        fig_path = os.path.join(self.output_directory, "13_contour_plot.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        nueva_fila = pd.DataFrame({
            "Ejercicio": [13],
            "Descripción": ["Gráfico de contorno de Z = cos(X) + sin(Y)"],
            "Resultado": [fig_path]
        })
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)        

    #Ejercicio 14
    def punto_14(self):
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c=np.sqrt(x**2 + y**2), cmap="plasma", alpha=0.75)
        plt.colorbar(label="Densidad (Color por Magnitud)")
        plt.xlabel("Eje X")
        plt.ylabel("Eje Y")
        plt.title("Gráfico de Dispersión con Color por Densidad")

        fig_path = os.path.join(self.output_directory, "14_scatter_density.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        nueva_fila = pd.DataFrame({
            "Ejercicio": [14],
            "Descripción": ["Gráfico de dispersión con 1000 puntos aleatorios y color por densidad"],
            "Resultado": [fig_path]
        })
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)

    #Ejercicio 15
    def punto_15(self):
        x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        y = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.cos(X) + np.sin(Y)  # Misma función del punto 12

        plt.figure(figsize=(8, 6))
        contour_filled = plt.contourf(X, Y, Z, levels=50, cmap="viridis")  # Contorno lleno
        plt.colorbar(contour_filled, label="Valor de Z")
        plt.xlabel("Eje X")
        plt.ylabel("Eje Y")
        plt.title("Gráfico de Contorno Lleno")

        fig_path = os.path.join(self.output_directory, "15_filled_contour.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        nueva_fila = pd.DataFrame({
            "Ejercicio": [15],
            "Descripción": ["Gráfico de contorno lleno basado en la función cos(x) + sin(y)"],
            "Resultado": [fig_path]
        })
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)

    #Ejercicio 16

    def mejora_punto_12(self):
        np.random.seed(42)
        x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        y_sin = np.sin(x)
        y_noisy = y_sin + np.random.normal(scale=0.1, size=len(x))

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y_noisy, label=r"$y = \sin(x) + \text{ruido Gaussiano}$", color="blue", alpha=0.6)
        plt.plot(x, y_sin, label=r"$y = \sin(x)$", color="red", linewidth=2)

        plt.xlabel(r"\textbf{Eje X}")
        plt.ylabel(r"\textbf{Eje Y}")
        plt.title(r"\{Gráfico de Dispersión}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        fig_path = os.path.join(self.output_directory, "Mejora_punto12.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        nueva_fila = pd.DataFrame({
            "Ejercicio": [12.2],
            "Descripción": ["Gráfico de dispersión mejorado con etiquetas y leyendas en LaTeX"],
            "Resultado": [fig_path]
        })
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)

    #Histogramas 

    #16 Crea un histograma a partir de un array de 1000 números aleatorios generados con una distribución normal.
    def punto_16(self):
        np.random.seed(42)
        data = np.random.normal(loc=0, scale=1, size=1000)  # Media=0, Desviación=1

        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, color="blue", alpha=0.7, edgecolor="black")
        
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        plt.title("Histograma de 1000 números con distribución normal")
        
        fig_path = os.path.join(self.output_directory, "16_histograma_normal.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        nueva_fila = pd.DataFrame({
            "Ejercicio": [16],
            "Descripción": ["Histograma de 1000 números aleatorios con distribución normal"],
            "Resultado": [fig_path]
        })
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)

    #17 Genera dos sets de datos con distribuciones normales diferentes y muéstralos en el mismo histograma.
        

    def punto_17(self):
        np.random.seed(42)

        # Generamos dos conjuntos de datos con distribuciones normales diferentes
        data1 = np.random.normal(loc=0, scale=1, size=1000)   # Media=0, Desviación=1
        data2 = np.random.normal(loc=3, scale=1.5, size=1000) # Media=3, Desviación=1.5

        plt.figure(figsize=(10, 6))
        
        # Histograma del primer conjunto de datos
        plt.hist(data1, bins=30, color="blue", alpha=0.5, edgecolor="black", label="Media=0, Desv=1")
        
        # Histograma del segundo conjunto de datos
        plt.hist(data2, bins=30, color="red", alpha=0.5, edgecolor="black", label="Media=3, Desv=1.5")
        
        # Etiquetas y título
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        plt.title("Histogramas de dos distribuciones normales")
        plt.legend()  

        # Guardar el gráfico
        fig_path = os.path.join(self.output_directory, "18_histograma_doble.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Guardar resultados en el DataFrame
        nueva_fila = pd.DataFrame({
            "Ejercicio": [17],
            "Descripción": ["Histogramas de dos conjuntos de datos con distribuciones normales diferentes"],
            "Resultado": [fig_path]
        })
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)

    #18 Experimenta con diferentes valores de bins (por ejemplo, 10, 30, 50) en un histograma y observa cómo cambia la representación.


    def punto_18(self):
        np.random.seed(42)

        # Generamos datos con distribución normal
        data = np.random.normal(loc=0, scale=1, size=1000)

        # Definimos los valores de bins a probar
        bins_values = [10, 30, 50]
        
        plt.figure(figsize=(12, 6))

        for i, bins in enumerate(bins_values, 1):
            plt.subplot(1, 3, i)  # Crear subgráficos (3 en una fila)
            plt.hist(data, bins=bins, color="blue", alpha=0.7, edgecolor="black")
            plt.title(f"Histograma con {bins} bins")
            plt.xlabel("Valor")
            plt.ylabel("Frecuencia")

        plt.tight_layout()

        # Guardar el gráfico
        fig_path = os.path.join(self.output_directory, "19_histograma_bins.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Guardar resultados en el DataFrame
        nueva_fila = pd.DataFrame({
            "Ejercicio": [18],
            "Descripción": ["Comparación de histogramas con 10, 30 y 50 bins"],
            "Resultado": [fig_path]
        })
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)

    #19 Añade una línea vertical que indique la media de los datos en el histograma.

    def punto_19(self):
        np.random.seed(42)

        # Generamos datos con distribución normal
        data = np.random.normal(loc=0, scale=1, size=1000)
        media = np.mean(data)  # Calculamos la media

        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=30, color="blue", alpha=0.7, edgecolor="black")
        
        # Añadir línea vertical en la media
        plt.axvline(media, color="red", linestyle="dashed", linewidth=2, label=f"Media = {media:.2f}")
        
        plt.title("Histograma con Media")
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        plt.legend()  # Mostrar leyenda con la media

        # Guardar el gráfico
        fig_path = os.path.join(self.output_directory, "Mejora_punto_19.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Guardar resultados en el DataFrame
        nueva_fila = pd.DataFrame({
            "Ejercicio": [19],
            "Descripción": ["Histograma con línea vertical indicando la media"],
            "Resultado": [fig_path]
        })
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)

    #20 

    def punto_20(self):
        np.random.seed(42)

        # Generar dos conjuntos de datos con distribuciones normales diferentes
        data1 = np.random.normal(loc=0, scale=1, size=1000)   # Media = 0, Desviación = 1
        data2 = np.random.normal(loc=3, scale=1.5, size=1000) # Media = 3, Desviación = 1.5

        plt.figure(figsize=(8, 6))

        # Crear histogramas superpuestos
        plt.hist(data1, bins=30, color="blue", alpha=0.6, edgecolor="black", label="Distribución 1")
        plt.hist(data2, bins=30, color="red", alpha=0.6, edgecolor="black", label="Distribución 2")

        plt.title("Histogramas Superpuestos")
        plt.xlabel("Valor")
        plt.ylabel("Frecuencia")
        plt.legend()  # Mostrar leyenda para diferenciar las distribuciones

        # Guardar el gráfico
        fig_path = os.path.join(self.output_directory, "21_histograma_superpuesto.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Guardar resultados en el DataFrame
        nueva_fila = pd.DataFrame({
            "Ejercicio": [20],
            "Descripción": ["Histogramas superpuestos de dos distribuciones normales"],
            "Resultado": [fig_path]
        })
        self.df = pd.concat([self.df, nueva_fila], ignore_index=True)




    # Método para ejecutar todos los ejercicios
    def ejecutar(self):
        """Ejecuta todos los ejercicios y guarda los resultados en un archivo Excel."""
        print("Ejecutando ejercicios...")
        self.punto_1()
        self.punto_2()
        self.punto_3()
        self.punto_4()
        self.punto_5()
        self.punto_6()  
        self.punto_7()  
        self.punto_8()  
        self.punto_9()  
        self.punto_10()  
        self.punto_11()
        self.punto_12()
        self.punto_13()
        self.punto_14()
        self.punto_15()
        self.mejora_punto_12()
        self.punto_16()
        self.punto_17()
        self.punto_18()
        self.punto_19()
        self.punto_20()



        


        # Guardar resultados en Excel
        self.guardar_resultados()
        
        print("¡Ejercicios completados!")
        return self.df
    
    # Método para guardar los resultados
    def guardar_resultados(self, nombre_archivo="resultados_ejercicios.xlsx"):
        """Guarda los resultados en un archivo Excel."""
        try:
            # Asegurarse de que openpyxl esté instalado
            self.df.to_excel(nombre_archivo, index=False)
            print(f"Resultados guardados en {nombre_archivo}")
        except Exception as e:
            print(f"Error al guardar los resultados: {e}")
            # Alternativa: guardar como CSV si hay problemas con Excel
            nombre_csv = nombre_archivo.replace('.xlsx', '.csv')
            try:
                self.df.to_csv(nombre_csv, index=False)
                print(f"Los resultados se guardaron alternativamente en {nombre_csv}")
            except Exception as csv_error:
                print(f"También falló el guardado como CSV: {csv_error}")


# Crear instancia y ejecutar
if __name__ == "__main__":
    ejercicios = Ejercicios()
    ejercicios.ejecutar()
