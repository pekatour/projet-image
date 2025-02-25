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

void kmeans2(Mat data, Mat& bestLabels, int maxIter){
    // Segmentation fond/forme.
    // data : Data for clustering : Image with 3 channels.
    // bestLabels : Output integer array that stores the cluster indices for every sample.
    // maxIter : The maximum number of iterations.
    // centers : Output matrix of the cluster centers, one row per each cluster center.

    bestLabels.create(data.rows, data.cols, CV_8UC1);

    cv::RNG rng;

    // Center initialization
    Vec3f c0 = data.at<Vec3f>(rng.uniform(0,data.rows),rng.uniform(0,data.cols));
    Vec3f c1 = data.at<Vec3f>(rng.uniform(0,data.rows),rng.uniform(0,data.cols));

    int it = 0;
    Vec3f s0(0,0,0);
    Vec3f s1(0,0,0);
    double p0 = 0;
    double p1 = 0;
    while (it<maxIter) {
        it++;
        // cout << s0[0] << " ";
        s0[0] = 0;
        s0[1] = 0;
        s0[2] = 0;
        s1[0] = 0;
        s1[1] = 0;
        s1[2] = 0;
        // cout << s0[0] << "\n";
        for (size_t i = 0; i < data.rows; i++)
        {
            for (size_t j = 0; j < data.cols; j++)
            {
                auto d0 = cv::norm(c0 - data.at<Vec3f>(i,j));
                auto d1 = cv::norm(c1 - data.at<Vec3f>(i,j));
                // cout << d0 << " " << d1 << "\n";

                if (d0 < d1) {
                    bestLabels.at<uchar>(i,j) = 1; 
                    s0[0] += data.at<Vec3f>(i,j)[0];
                    s0[1] += data.at<Vec3f>(i,j)[1];
                    s0[2] += data.at<Vec3f>(i,j)[2];
                }
                else {
                    bestLabels.at<uchar>(i,j) = 0;
                    s1[0] += data.at<Vec3f>(i,j)[0];
                    s1[1] += data.at<Vec3f>(i,j)[1];
                    s1[2] += data.at<Vec3f>(i,j)[2];
                }
            }  
        }
        
        if (countNonZero(bestLabels) == 0){ p0 = 1; }
        else { p0 = 1./countNonZero(bestLabels); }
        c0[0] = s0[0] * p0;
        c0[1] = s0[1] * p0;
        c0[2] = s0[2] * p0;

        if (countNonZero(bestLabels) == bestLabels.cols * bestLabels.rows){ p1 = 1; }
        else { p1 = 1./(bestLabels.cols * bestLabels.rows - countNonZero(bestLabels)); }
        c1[0] = s1[0] * p1;
        c1[1] = s1[1] * p1;
        c1[2] = s1[2] * p1;        
    }
}

void Criterias(Mat im, Mat ref, float res[3]) {
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
                        TP++;
                    }
                    else {
                        TN++;
                    }
                }
                else {
                    if(ref.at<uchar>(i,j) == 255){
                        FN++;
                    }
                    else {
                        FP++;
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
    Mat converted;
    m.convertTo(converted,CV_32F);

    // 2) kmeans asks for a mono-dimensional list of "points". Our "points" are the pixels of the image that can be seen as 3D points
    // where each coordinate is one of the color channel (e.g. R, G, B). But they are organized as a 2D table, we need
    // to re-arrange them into a single vector.
    // see the method Mat.reshape(), it is similar to matlab's reshape
    vector<int> new_shape = {m.cols * m.rows, 1};
    converted = converted.reshape(3, new_shape);
    // now we can call kmeans(...)
    Mat bestLabels;
    Mat centers;
    kmeans(converted,k,bestLabels, TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
    Mat res;
    normalize(bestLabels, res, 0, 255, cv::NORM_MINMAX);
    new_shape = {m.rows, m.cols};
    res = res.reshape(1,new_shape);
    res.convertTo(res, CV_8U);

    kmeans2(converted,bestLabels,15);
    Mat res2;
    normalize(bestLabels, res2, 0, 255, cv::NORM_MINMAX);
    res2 = res2.reshape(1,new_shape);
    res2.convertTo(res2, CV_8U);

    namedWindow("Avant kmeans", cv::WINDOW_AUTOSIZE);
    imshow("Avant kmeans", m);

    namedWindow("Avec kmeans OpenCV", cv::WINDOW_AUTOSIZE);
    imshow("Avec kmeans OpenCV", res);    

    namedWindow("Avec kmeans2", cv::WINDOW_AUTOSIZE);
    imshow("Avec kmeans2", res2); 

    waitKey();

    if(!groundTruthFilename.empty()) {
        Mat ref;
        ref = imread(groundTruthFilename, cv::IMREAD_GRAYSCALE); 

        float* tab1 = new float[3];
        float* tab2 = new float[3];
        Criterias(res, ref, tab1);
        res = Mat::ones(res.size(),res.type()) * 255 - res;
        Criterias(res, ref, tab2);
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

        Criterias(res2, ref, tab1);
        res2 = Mat::ones(res2.size(),res2.type()) * 255 - res2;
        Criterias(res2, ref, tab2);
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
    return EXIT_SUCCESS;
}
