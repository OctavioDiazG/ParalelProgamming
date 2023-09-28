# **Examen sobre Complejidad Temporal en Algoritmos, Programación Paralela y CUDA**

1. ¿Qué es la complejidad temporal de un algoritmo?
   - [ ] El número de instrucciones en un algoritmo.
   - [ ] La cantidad de memoria utilizada por un algoritmo.
   - [x] El tiempo que tarda un algoritmo en ejecutarse en función del tamaño de entrada.

2. ¿Cuál de las siguientes notaciones se utiliza comúnmente para describir la complejidad temporal de un algoritmo?
   - [ ] Números romanos. $I,II,III$ 
   - [ ] Notación big $\theta$     $\theta(N)$ 
   - [x] Notación big $O$      $O(n)$.

3. ¿Qué significa $O(log n)$  en la notación big O?
   - [x] El tiempo de ejecución del algoritmo crece de manera logarítmica con el tamaño de entrada.
   - [ ] El tiempo de ejecución del algoritmo es constante.
   - [ ] El tiempo de ejecución del algoritmo crece de manera lineal con el tamaño de entrada.

6. ¿Qué es la programación paralela en informática?
   - [ ] Un tipo de lenguaje de programación.
   - [ ] Un método para ocultar la complejidad de un algoritmo.
   - [x] Una técnica que utiliza múltiples hilos o procesadores para resolver un problema de manera simultánea.

7. ¿Cuál es uno de los beneficios clave de la programación paralela?
   - [ ] Mayor utilización de recursos de almacenamiento.
   - [x] Mayor capacidad de procesamiento y reducción de tiempos de ejecución.
   - [ ] Menor consumo de energía. (En teoria este tambien ya que termina mas rapido por lo tanto ya no consume tantos watts y reduce el tiempo de consumo maximo de energia por lo que hay menor consumo de energia)

8. ¿Qué es CUDA en el contexto de la programación paralela?
   - [ ] Un lenguaje de programación paralela.
   - [x] Una plataforma de computación paralela desarrollada por NVIDIA&trade; 
   - [ ] Una técnica de programación secuencial.

9. ¿Cuál es el objetivo principal de CUDA?
   - [ ] Ejecutar algoritmos de manera secuencial.
   - [x] Aprovechar el poder de procesamiento de las tarjetas gráficas (GPUs) para cálculos paralelos.
   - [ ] Optimizar el uso de la memoria RAM.

10. ¿Cuál de las siguientes declaraciones sobre CUDA es cierta?
   - [ ] CUDA solo funciona en sistemas operativos Windows.
   - [x] CUDA es una plataforma de programación paralela que permite el desarrollo de aplicaciones para GPUs NVIDIA&trade; . 
   - [ ] CUDA se utiliza exclusivamente en aplicaciones de inteligencia artificial.

11. ¿Qué es un "kernel" en el contexto de CUDA?
 - [ ] Un tipo de unidad de procesamiento central (CPU).
   - [ ] Un tipo de tarjeta de video.
   - [x] Un pequeño programa o función que se ejecuta en la GPU de manera paralela.
12. ¿Cuál de las siguientes afirmaciones describe mejor la ventaja de utilizar CUDA en aplicaciones de procesamiento intensivo?
    - [ ] CUDA solo se utiliza para aplicaciones de bajo rendimiento.
    - [ ] CUDA solo es adecuado para aplicaciones de oficina.
    - [x] CUDA permite un aumento significativo en el rendimiento al aprovechar la potencia de cálculo de las GPUs.
  
  
## Calculo de complejidad de un algoritmo

Analiza los siguientes algoritmos y calcula su complejidad en _Big O notation_ 

Algoritmo 1:
```cpp
#include <iostream>

int algoritmo1(int n) {
    int resultado = 0;
    
    for (int i = 0; i < n; i++) {
        resultado += i; // Operación simple O(1)
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            resultado += i * j; // Operación simple O(1)
        }
    }
    
    return resultado;
}

int main() {
    int n;
    std::cout << "Ingrese el valor de n: ";
    std::cin >> n;
    
    int resultado = 1(n);
    std::cout << "Resultado: " << resultado << std::endl;
    
    return 0;
}

```

The first 'for' loop iterates from 0 to n, performing a simple operation 'result += i' in each iteration. This is a constant-time operation O(1), and since the loop runs n times, its contribution to the overall complexity is O(n).

The second nested loop also iterates from 0 to n, and within this loop, there is another simple operation 'result += i * j,' which is also a constant-time O(1) operation. Since this loop is nested inside the first loop, it will run n times for each iteration of the first loop. Therefore, the second loop contributes to a complexity of O(n^2).

The code complexity is deceptive because the main function never calls the algoritmo1 function. As a result, the main function is merely performing a constant-time operation, denoted as O(1), as it consists of a single multiplication by 1(n). However, the complexity of the algoritmo1 function is indeed O(n^2).

Algoritmo 2:

```cpp
#include <iostream>
#include <algorithm>
#include <vector>

int algoritmo2(std::vector<int>& arr) {
    int n = arr.size();
    int resultado = 0;
    
    // Ordenamos el vector utilizando QuickSort con complejidad O(n log n)
    std::sort(arr.begin(), arr.end());
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                resultado += arr[i] * arr[j] * arr[k];
            }
        }
    }
    
    return resultado;
}

int main() {
    int n;
    std::cout << "Ingrese la cantidad de elementos: ";
    std::cin >> n;
    
    std::vector<int> arr(n);
    std::cout << "Ingrese los elementos del vector:" << std::endl;
    for (int i = 0; i < n; i++) {
        std::cin >> arr[i];
    }
    
    int resultado = algoritmo2(arr);
    std::cout << "Resultado: " << resultado << std::endl;
    
    return 0;
}

```

Sorting a vector of size 'n' using QuickSort: This has a complexity of O(n log n) because QuickSort is efficient in most cases.

Three nested loops: These loops share the same variable 'n' as their limit, which means each of them will execute 'n' times. Within these loops, there is an operation that involves multiplying elements of the vector. Since the loops are nested, this operation will run a total of n * n * n = n^3 times.

