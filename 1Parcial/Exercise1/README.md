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
    };
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
    };
    int c[10][10] = {};
    int i, j, k;
    
	
    for(i=0;i<10;i++){
        for(j=0;j<10;j++){
            for(k=0;k<10;k++){
                c[i][j]+=a[i][k]*b[k][j];
            }
        }
    }
    
    for(i = 0; i < 10; i++){
        for(j = 0; j < 10; j++){
            printf ("%i ", c[i][j]);
        }
        printf("\n");
    }
    
    return 0;
}

```
