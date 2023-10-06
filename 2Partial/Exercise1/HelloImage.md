# Hello Image Exercise

With OpenCV make an image apear via Code and then apply filters to it via code.


### What is OpenCV?
OpenCV is an open-source computer vision library that provides a real-time optimized Computer Vision library, tools, and hardware. It also supports model execution for Machine Learning (ML) and Artificial Intelligence. OpenCV is released under the Apache 2 License and is free for commercial use.
Some of the most common use cases of OpenCV include face recognition, automated inspection and surveillance, counting the number of people in public places such as airports, vehicle counting on highways along with their speeds, and interactive art installations.

```c++
#include <iostream>
#include <Opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <filesystem>

int main() {
    std::string imagePath; //Initialize image path variable
    std::cout << "Enter the path to the image: ";
    std::getline(std::cin, imagePath); //Give a value to the image path

    //Check if file exist, if not, exit program
    if(!std::filesystem::exists(imagePath)) {
        std::cout << "File does not Exist at the given path" << std::endl;
        return -1;
    }

    //save image in a Matrix class, which cv can read as an image
    cv::Mat image = cv::imread(imagePath);

    //Check the file in the given path if it's not an image, exit program
    if (image.empty()) {
        std::cout << "Error, no image found in the given path" << std::endl;
        return -1;
    } else{
        std::cout << "Image was found successfully in the given path" << std::endl;
    }

    //show the image in the display
    cv::imshow("Image", image);


    //wait until a key is pressed
    cv::waitKey(0);

    //create a Matrix class with the name of rgb to separate the image in 3 channels
    cv::Mat rgb[3];

    cv::split(image, rgb);
    cv::Mat redChannel, greenChannel, blueChannel; //initialize each filter as a new matrix class
    cv::applyColorMap(rgb[0], redChannel, cv::COLORMAP_AUTUMN);
    cv::applyColorMap(rgb[1], greenChannel, cv::COLORMAP_SPRING);
    cv::applyColorMap(rgb[2], blueChannel, cv::COLORMAP_JET);

    //show the 3 images with the filters
    cv::imshow("blueChannel", blueChannel);
    cv::imshow("greenChannel", greenChannel);
    cv::imshow("redChannel", redChannel);

    //wait until a key is pressed
    cv::waitKey(0);


    return 0;
}
```

### Code Explanation
The code is written in C++ and uses the OpenCV library to perform several operations on an image. Here’s a breakdown of what it does:

It prompts the user to enter the path to an image file.

It checks if the file exists at the given path. If not, it prints an error message and exits the program.

It reads the image from the given path and stores it in a matrix, which is a data structure provided by OpenCV to store images.

It checks if the image is empty (i.e., if it’s not a valid image file). If it is, it prints an error message and exits the program.

It displays the original image in a window.

It splits the image into three channels: blue, green, and red. This is because images in OpenCV are typically stored in BGR format.

It applies different color maps to each of the channels to highlight different features of the image:

The blue channel is mapped with a ‘jet’ color map, which is a rainbow-like color map.

The green channel is mapped with a ‘spring’ color map, which varies from magenta to yellow.

The red channel is mapped with an ‘autumn’ color map, which varies from red to yellow.

It displays each of these color-mapped channels in separate windows.

Finally, it waits for a key press before closing all windows and ending the program.

The code could be used for various purposes such as image analysis or feature extraction, depending on what you want to do with the color-mapped channels.

### complexity 

The code doesn’t have a Big O notation complexity as it doesn’t contain any loops or recursive calls that would affect the time or space complexity. The operations made in the code (such as reading an image, applying color maps, and displaying images) are all constant time operations, meaning they take the same amount of time regardless of the size of the input. Therefore, we can say the time complexity of the code is O(1), which means it has constant time complexity.

However, please note that while the code itself has a time complexity of O(1), the underlying operations (like cv::imread, cv::imshow) may have different complexities based on their implementation in the OpenCV library. These complexities are abstracted away from the user and generally not considered when analyzing the time complexity of the code. but please take that in mind.


### References
[1] “OpenCV - overview,” GeeksforGeeks, https://www.geeksforgeeks.org/opencv-overview/ (accessed Oct. 6, 2023). 

[2] “Home,” OpenCV, https://opencv.org/ (accessed Oct. 6, 2023). 
