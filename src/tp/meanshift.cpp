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
    cout << "Usage:\n\t " << progName << " <image_file> <hs> <hc> [<image_ground_truth>]" << endl;
}

void Criterias(Mat im, Mat ref, float res[3]) {
    // returns the precision, sensitivity and DSC of the segmentation in the res parameter
    // im and ref are the same size, of type CV_8UC1

    float TP = 0;
    float FP = 0;
    float TN = 0;
    float FN = 0;
    for (size_t i = 0; i < im.rows; i++)
        {
            for (size_t j = 0; j < im.cols; j++)
            {
                if (im.at<uchar>(i,j) == ref.at<uchar>(i,j)){
                    if(ref.at<uchar>(i,j) == 255){
                        TN++;
                    }
                    else {
                        TP++;
                    }
                }
                else {
                    if(ref.at<uchar>(i,j) == 255){
                        FP++;
                    }
                    else {
                        FN++;
                    }
                }
            }
        }
        
        float P = TP / ( TP + FP );
        float S = TP / ( TP + FN );
        float DSC = 2 * TP / ( 2 * TP + FP + FN );
        res[0] = P;
        res[1] = S;
        res[2] = DSC;
}

int main(int argc, char** argv) {

    // get passed arguments
    if(argc != 4 && argc != 5) {
        cout << " Incorrect number of arguments." << endl;
        printHelp(string(argv[0]));
        return EXIT_FAILURE;
    }
    const auto imageFilename = string(argv[1]);
    const string groundTruthFilename = (argc == 5) ? string(argv[4]) : string();
    const int hs = stoi(argv[2]); // spatial threshold
    const int hc = stoi(argv[3]); // color threshold

    // load image from file
    Mat m;
    m = imread(imageFilename, cv::IMREAD_COLOR);

    // resize down image for efficiency
    const int down_width = 200;
    const int down_height = 200;
    resize(m, m, Size(down_width, down_height), INTER_LINEAR);
    namedWindow("Resized image", cv::WINDOW_AUTOSIZE);
    imshow("Resized image", m);

    // convert image to floats CV_32F
    m.convertTo(m,CV_32F);

    // meanshift algorithm
    const float eps = TermCriteria::EPS; // stop parameter for distance pixel - mean
    const int kmax = 50; // stop parameter, max iterations
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
                for (int i_xi = max(i-hs,0); i_xi < min(hs + i,m.rows); i_xi++) {   
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
        arret = (k > kmax) || !existe;
    }

    /* Mat res;
    cv::cvtColor(m, res, cv::COLOR_BGR2GRAY);
    res.convertTo(res, CV_8U);
    int* c_modes = new int[50];
    int* hist_modes = new int[50];
    int nb_modes = 0;

    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            bool app_mode = false;
            for (int k = 0; k < nb_modes; k++) {l;
                if (static_cast<int>(res.at<uchar>(i,j)) == c_modes[k]) {
                    app_mode = true;
                    hist_modes[k] = hist_modes[k]+1;
                }
            }
            if (!app_mode && nb_modes<=50){
                
                hist_modes[nb_modes]=1;
                c_modes[nb_modes] = res.at<uchar>(i,j);
                nb_modes++;
            }
            else {
                if (nb_modes > 50){
                }
            }
        }
    }
    int maxi = 0;
    int i_max;
    int i_min;
    int mini = m.cols*m.rows;

    for (int i = 0; i < nb_modes; i++) {
        if (hist_modes[i]> maxi){
            maxi = hist_modes[i];
            i_max = i;
        }
        if (hist_modes[i]< mini){
            mini = hist_modes[i];
            i_min = i;
        }
    }
    int milieu;
    milieu = (c_modes[i_min] + c_modes[i_max])/2;

    cv::threshold(res, res, milieu, 255, cv::THRESH_BINARY); */

    Mat res;
    vector<int> new_shape = {m.cols * m.rows, 1};
    res = m.reshape(3, new_shape);
    // now we can call kmeans(...)
    Mat bestLabels;
    Mat centers;
    kmeans(res,2,bestLabels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
    normalize(bestLabels, res, 0, 255, cv::NORM_MINMAX);
    new_shape = {m.rows, m.cols};
    res = res.reshape(1,new_shape);
    res.convertTo(res, CV_8U);


    // show resulting segmented image using meanshift algorithm
    m.convertTo(m, CV_8U);
    namedWindow("Segmented image with MeanShift", cv::WINDOW_AUTOSIZE);
    imshow("Segmented image with MeanShift", m);

    // show resulting segmented image after using kmeans to binarize
    namedWindow("Segmented image after kmeans", cv::WINDOW_AUTOSIZE);
    imshow("Segmented image after kmeans", res);

    if(!groundTruthFilename.empty()) {
        Mat ref;
        ref = imread(groundTruthFilename, cv::IMREAD_GRAYSCALE); 

        // Arrays for the criterias of the final image, with its original colors and its inverted colors
        float* tab1 = new float[3];
        float* tab2 = new float[3];

        // Criterias for the segmented image
        resize(ref, ref, Size(down_width, down_height), INTER_LINEAR);
        Criterias(res, ref, tab1);
        res = Mat::ones(res.size(),res.type()) * 255 - res;
        Criterias(res, ref, tab2);
        cout << "MeanShift" << endl;
        if (tab1[2] > tab2[2]) {
            cout << "P : " << tab1[0] << endl;
            cout << "S : " << tab1[1] << endl;
            cout << "DSC : " << tab1[2] << endl;
        }
        else {
            cout << "P : " << tab2[0] << endl;
            cout << "S : " << tab2[1] << endl;
            cout << "DSC : " << tab2[2] << endl;
        }

    }

    // wait until esc pressed
    while(cv::waitKey(1) != 27);
}