#include "ocv_utils.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/** Print usage for this implementation of meanshift algorithme.
 */
void printHelp(const string& progName) {
    cout << "Usage:\n\t " << progName << " <image_file> [<image_ground_truth>]" << endl;
}

int main(int argc, char** argv) {

    // get passed arguments
    if(argc != 2 && argc != 3) {
        cout << " Incorrect number of arguments." << endl;
        printHelp(string(argv[0]));
        return EXIT_FAILURE;
    }
    const auto imageFilename = string(argv[1]);
    const string groundTruthFilename = (argc == 3) ? string(argv[2]) : string();

    // load image from file
    Mat m;
    m = imread(imageFilename, cv::IMREAD_COLOR);
    
    // show base image
    namedWindow("Base image", cv::WINDOW_AUTOSIZE);
    imshow("Base image", m);

    // resize down image for efficiency
    const int down_width = 200;
    const int down_height = 200;
    resize(m, m, Size(down_width, down_height), INTER_LINEAR);
    namedWindow("Resized image", cv::WINDOW_AUTOSIZE);
    imshow("Resized image", m);

    // convert image to floats CV_32F
    m.convertTo(m,CV_32F);

    // meanshift algorithm
    const int hs = 3; // spatial threshold
    const int hc = 50; // color threshold
    const float eps = TermCriteria::EPS; // stop parameter for distance pixel - mean
    const int kmax = 30; // stop parameter, max iterations
    int k = 0; // iteration parameter
    bool arret = false; // true when stop conditions are met
    while (!arret) {
        bool existe = false;

        // iterate over pixels in image
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {

                Vec3f s(0,0,0); // sum of neighboring pixels
                float nx = 0; // total of neighboring pixels

                // iterate over neighboring pixels
                for (int i_xi = max(i-hs,0); i_xi < min(hs + i - 1,m.rows - 1); i_xi++) {   
                    for (int j_xi = max(j-hs,0); j_xi < min(hs + j, m.cols); j_xi++) {   
                        if (cv::norm(m.at<Vec3f>(i_xi,j_xi) - m.at<Vec3f>(i,j)) <= hc) {
                            nx++;
                            s[0] += m.at<Vec3f>(i_xi,j_xi)[0];
                            s[1] += m.at<Vec3f>(i_xi,j_xi)[1];
                            s[2] += m.at<Vec3f>(i_xi,j_xi)[2];
                        }
                    }
                }
                
                // compute mean of neighboring pixels
                Vec3f mh;
                mh[0] = s[0] / nx;
                mh[1] = s[1] / nx;
                mh[2] = s[2] / nx;

                // update stop condition
                existe = (cv::norm(mh - m.at<Vec3f>(i,j)) > eps) || existe ;
                
                // replace current pixel by mean of neighboring pixels
                m.at<Vec3f>(i,j) = mh;
            }
        }

        // update parameters
        k++;
        arret = (k > kmax) && !existe;
    }

    // show resulting segmented image using meanshift algorithm
    namedWindow("Segmented image", cv::WINDOW_AUTOSIZE);
    imshow("Segmented image", m);

    // wait until esc pressed
    while(cv::waitKey(1) != 27);
}