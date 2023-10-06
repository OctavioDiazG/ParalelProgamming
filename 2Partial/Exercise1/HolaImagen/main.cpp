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
