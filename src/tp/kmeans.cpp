#include "ocv_utils.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

using namespace cv;
using namespace std;


void printHelp(const string& progName)
{
    cout << "Usage:\n\t " << progName << " <image_file> <K_num_of_clusters> [<image_ground_truth>]" << endl;
}

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 4)
    {
        cout << " Incorrect number of arguments." << endl;
        printHelp(string(argv[0]));
        return EXIT_FAILURE;
    }

    const auto imageFilename = string(argv[1]);
    const string groundTruthFilename = (argc == 4) ? string(argv[3]) : string();
    const int k = stoi(argv[2]);

    // just for debugging
    {
        cout << " Program called with the following arguments:" << endl;
        cout << " \timage file: " << imageFilename << endl;
        cout << " \tk: " << k << endl;
        if(!groundTruthFilename.empty()) cout << " \tground truth segmentation: " << groundTruthFilename << endl;
    }

    // load the color image to process from file
    Mat m;
    m = imread(imageFilename, cv::IMREAD_COLOR);
    //namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    //imshow("Display window", m);
    //waitKey(0);
    // for debugging use the macro PRINT_MAT_INFO to print the info about the matrix, like size and type
    PRINT_MAT_INFO(m);

    // 1) in order to call kmeans we need to first convert the image into floats (CV_32F)
    // see the method Mat.convertTo()
    Mat newm;
    m.convertTo(newm,CV_32F);

    // 2) kmeans asks for a mono-dimensional list of "points". Our "points" are the pixels of the image that can be seen as 3D points
    // where each coordinate is one of the color channel (e.g. R, G, B). But they are organized as a 2D table, we need
    // to re-arrange them into a single vector.
    // see the method Mat.reshape(), it is similar to matlab's reshape
    vector<int> new_shape = {m.cols * m.rows, 1};
    newm = newm.reshape(3, new_shape);
    // now we can call kmeans(...)
    Mat bestLabels;
    Mat centers;
    kmeans(newm,k,bestLabels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
    Mat newnewm;
    normalize(bestLabels, newnewm, 0, 255, cv::NORM_MINMAX);
    new_shape = {m.rows, m.cols};
    newnewm = newnewm.reshape(1,new_shape);
    newnewm.convertTo(newnewm, CV_8U);
    namedWindow("Après kmeans", cv::WINDOW_AUTOSIZE);
    imshow("Après kmeans", newnewm);

    namedWindow("Avant kmeans", cv::WINDOW_AUTOSIZE);
    imshow("Avant kmeans", m);


    waitKey();

    return EXIT_SUCCESS;
}
