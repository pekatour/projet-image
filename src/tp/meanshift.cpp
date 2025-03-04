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
    const string groundTruthFilename = (argc == 3) ? string(argv[2]) : string();;

    const int kmax = 2;
    // load the color image to process from file
    Mat m;
    m = imread(imageFilename, cv::IMREAD_COLOR);
    int down_width = 200;
    int down_height = 200;
 
   // resize down
    resize(m, m, Size(down_width, down_height), INTER_LINEAR);
    namedWindow("Image", cv::WINDOW_AUTOSIZE);
    imshow("Image", m);
    m.convertTo(m,CV_32F);
    const int hs = 10;
    const int hc = 10; // Aucune idée de comment déterminer
    const float eps = TermCriteria::EPS;

    int k = 10;
    bool arret = false;

    while (!arret) {
        k++;
        bool existe = false;
        for (int i = 0; i < m.rows; i++)
        {
            for (int j = 0; j < m.cols; j++)
            {
                Vec3f s(0,0,0);
                float nx = 0;
                for (int i_xi = max(i-hs,0); i_xi < min(hs + i - 1,m.rows - 1); i_xi++)
                {   
                    for (int j_xi = max(i-hs,0); j_xi < min(hs + i, m.cols); j_xi++)
                    {   
                        if (cv::norm(m.at<Vec3f>(i_xi,j_xi) - m.at<Vec3f>(i,j)) <= hc)
                        {
                            nx++;
                            s[0] += m.at<Vec3f>(i_xi,j_xi)[0];
                            s[1] += m.at<Vec3f>(i_xi,j_xi)[1];
                            s[2] += m.at<Vec3f>(i_xi,j_xi)[2];
                        }
                    }
                }
                
                Vec3f mh;
                mh[0] = s[0] / nx;
                mh[1] = s[1] / nx;
                mh[2] = s[2] / nx;

                if (cv::norm(mh - m.at<Vec3f>(i,j)) > eps){
                    existe = true;
                }

                m.at<Vec3f>(i,j) = mh;
            }
        }

        arret = (k > kmax) && !existe;
    }
    namedWindow("Image mean-shift", cv::WINDOW_AUTOSIZE);
    imshow("Image mean-shift", m);
    waitKey(0);

}