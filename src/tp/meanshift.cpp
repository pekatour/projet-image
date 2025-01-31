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
    cout << "Usage:\n\t " << progName << " <image_file> [<image_ground_truth>]" << endl;
}

int main(int argc, char** argv)
{
    if(argc != 2 && argc != 3)
    {
        cout << " Incorrect number of arguments." << endl;
        printHelp(string(argv[0]));
        return EXIT_FAILURE;
    }

    const auto imageFilename = string(argv[1]);
    const string groundTruthFilename = string(argv[2]);
}