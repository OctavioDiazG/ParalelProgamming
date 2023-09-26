# Create Naive Multiplication in C  matrix by matrix

To multiply two matrices, we need to follow set of rules. we have two matrices A and B, with dimensions m x n and n x p, respectively. The resulting matrix C will have dimensions m x p.

Here’s how to calculate the elements of C:

For each element in the resulting matrix C, take the dot product of the corresponding row in matrix A and the corresponding column in matrix B.
Add up the products from step 1 to get the value of the element in C.

- [ ] 1
- [x] 2
- [ ] 3

---
[CheatSheet](https://www.markdownguide.org/cheat-sheet/)

![marix](Matrix.png)

```c

#include <stdio.h>

//multiply a matrix with a matrix

int main() {
    int a[10][10] = {
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
    }; // determine the value of Matrix a
    int b[10][10] = { 
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
        {1,2,3,4,5,6,7,8,9,10},
    }; // determine the value of Matrix b
    int c[10][10] = {}; //initialize and determine the space of the Matrix c 
    int i, j, k; //initialize for variables
    
	
    for(i=0;i<10;i++){ // first for iterates through rows of Matrix a
        for(j=0;j<10;j++){ // second For iterates through the comumns of Matrix b
            for(k=0;k<10;k++){ // third For iterates through Columns of Matrix a and rows of Matrix b
                c[i][j]+=a[i][k]*b[k][j];  // Multiply corresponding elements from matrices 'a' and 'b',
                // then add the result to the corresponding element in matrix 'c'.
            }
        }
    }
    
    for(i = 0; i < 10; i++){ //use this nested Fors to print the matrix 
        for(j = 0; j < 10; j++){
            printf ("%i ", c[i][j]);
        }
        printf("\n");
    }
    
    return 0;
}

```


This code performs matrix multiplication using three nested loops. 
It multiplies two 10x10 matrices 'a' and 'b' and stores the result in matrix 'c'. 
Here's a breakdown of the code:

1. It uses three nested 'for' loops:
   - The outermost loop (i) iterates from 0 to 9.
   - The middle loop (j) iterates from 0 to 9.
   - The innermost loop (k) iterates from 0 to 9.

2. Inside the innermost loop, it calculates the product of elements 'a[i][k]' 
and 'b[k][j]' and adds it to the corresponding element 
in the result matrix 'c[i][j]'.

In summary, this code performs a naive matrix multiplication, 
where each element in the result matrix 'c' is the sum of the products of 
corresponding elements from matrices 'a' and 'b'. It does this by using three 
nested loops to iterate through the indices of the matrices.
