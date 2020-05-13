/**
 * This file is part of FLaME.
 * Copyright (C) 2017 W. Nicholas Greene (wng@csail.mit.edu)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see <http://www.gnu.org/licenses/>.
 *
 * @file epipolar_geometry_test.cc
 * @author W. Nicholas Greene
 * @date 2017-08-18 19:18:41 (Fri)
 */
//Required for sfm to work
#define CERES_FOUND 1

#include <unistd.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <chrono>

#include <boost/filesystem.hpp>
#include <sophus/se3.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/sfm/reconstruct.hpp>
#include <opencv2/sfm/fundamental.hpp>

#include <opencv2/core/eigen.hpp>

#include "flame/stereo/epipolar_geometry.h"
#include "flame/pose/gms_matcher.h"
#include "flame/flame.h"

#include "gtest/gtest.h"

namespace fs = boost::filesystem;
using namespace std;
using namespace cv;
using namespace cv::sfm;

namespace flame {

    namespace stereo {

        Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type) {
            const int height = max(src1.rows, src2.rows);
            const int width = src1.cols + src2.cols;
            Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
            src1.copyTo(output(Rect(0, 0, src1.cols, src1.rows)));
            src2.copyTo(output(Rect(src1.cols, 0, src2.cols, src2.rows)));

            if (type == 1)
            {
                for (size_t i = 0; i < inlier.size(); i++)
                {
                    Point2f left = kpt1[inlier[i].queryIdx].pt;
                    Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
                    line(output, left, right, Scalar(0, 255, 255));
                }
            }
            else if (type == 2)
            {
                for (size_t i = 0; i < inlier.size(); i++)
                {
                    Point2f left = kpt1[inlier[i].queryIdx].pt;
                    Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
                    line(output, left, right, Scalar(255, 0, 0));
                }

                for (size_t i = 0; i < inlier.size(); i++)
                {
                    Point2f left = kpt1[inlier[i].queryIdx].pt;
                    Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
                    circle(output, left, 1, Scalar(0, 255, 255), 2);
                    circle(output, right, 1, Scalar(0, 255, 0), 2);
                }
            }

            return output;
        }

        bool GmsMatch(Mat &img1, Mat &img2) {
            vector<KeyPoint> kp1, kp2;
            Mat d1, d2;
            vector<DMatch> matches_all, matches_gms;

            Ptr<ORB> orb = ORB::create(10000);
            orb->setFastThreshold(0);

            orb->detectAndCompute(img1, Mat(), kp1, d1);
            orb->detectAndCompute(img2, Mat(), kp2, d2);

#ifdef USE_GPU
            GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
            BFMatcher matcher(NORM_HAMMING);
            matcher.match(d1, d2, matches_all);
#endif

            // GMS filter
            vector<bool> vbInliers;
            gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);
            int num_inliers = gms.GetInlierMask(vbInliers, false, false);
            cout << "Get total " << num_inliers << " matches." << endl;

            // collect matches
            for (size_t i = 0; i < vbInliers.size(); ++i)
            {
                if (vbInliers[i] == true)
                {
                    matches_gms.push_back(matches_all[i]);
                }
            }

            // draw matching
            Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 1);

            imwrite("./matches.png", show);

            //-- Localize the object
            vector<Point2f> obj;
            vector<Point2f> scene;
            for( size_t i = 0; i < matches_gms.size(); i++ )
            {
                //-- Get the keypoints from the good matches
                obj.push_back( kp1[ matches_gms[i].queryIdx ].pt );
                scene.push_back( kp2[ matches_gms[i].trainIdx ].pt );
            }
            Mat H = findHomography( obj, scene, RANSAC );
        }

        Mat ReadCameraC(string filename = "camera_c.txt")
        {
            int rows = 3, cols = 3;
            double m;
            Mat out = Mat::zeros(rows, cols, CV_64FC1);//Matrix to store values

            ifstream fileStream(filename);
            string trash;
            for(int i = 0; i<6; ++i) {
                fileStream >> trash;
            }
            int cnt = 0;//index starts from 0
            while (fileStream >> m)
            {
                int temprow = cnt / cols;
                int tempcol = cnt % cols;
                out.at<double>(temprow, tempcol) = m;
                cnt++;
            }
            return out;
        }

        Eigen::Quaternionf GetQuaternion(Mat R)
        {
            vector<float> result(4, 0);
            double trace = R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2);

            if (trace > 0.0)
            {
                double s = sqrt(trace + 1.0);
                result[3] = (s * 0.5);
                s = 0.5 / s;
                result[0] = ((R.at<double>(2,1) - R.at<double>(1,2)) * s);
                result[1] = ((R.at<double>(0,2) - R.at<double>(2,0)) * s);
                result[2] = ((R.at<double>(1,0) - R.at<double>(0,1)) * s);
            }

            else
            {
                int i = R.at<double>(0,0) < R.at<double>(1,1) ? (R.at<double>(1,1) < R.at<double>(2,2) ? 2 : 1) : (R.at<double>(0,0) < R.at<double>(2,2) ? 2 : 0);
                int j = (i + 1) % 3;
                int k = (i + 2) % 3;

                double s = sqrt(R.at<double>(i, i) - R.at<double>(j,j) - R.at<double>(k,k) + 1.0);
                result[i] = s * 0.5;
                s = 0.5 / s;

                result[3] = (R.at<double>(k,j) - R.at<double>(j,k)) * s;
                result[j] = (R.at<double>(j,i) + R.at<double>(i,j)) * s;
                result[k] = (R.at<double>(k,i) + R.at<double>(i,k)) * s;
            }
            Eigen::Quaternionf qresult(result[0], result[1], result[2], result[3]);
            return qresult;
        }

        struct FrameDescriptor {
            Mat1b frame;
#ifdef USE_GPU
            vector<GpuMat> descriptors;
#else
            vector<Mat> descriptors;
#endif
            vector<KeyPoint> points;


            vector<Point2f> & GetPointsOnFrame() {
                BakePointsOnFrame();
                return pointsOnFrame;

            }

            void SetPointsOnFrame(vector<Point2f> & otherPts) {
                pointsOnFrame = otherPts;
            }

        private:
            vector<Point2f> pointsOnFrame;
            void BakePointsOnFrame() {
                for(auto & pt : points) {
                    pointsOnFrame.push_back(pt.pt);
                }
            }
        };

        //We track features along all inOut frames and keep only ones that are persistent
        shared_ptr<FrameDescriptor> GetDescriptor(Mat1b & frame, shared_ptr<FrameDescriptor>  in = nullptr) {
            if (frame.empty()) {
                return nullptr;
            }
            auto out = make_shared<FrameDescriptor>();

            vector<DMatch> matches_all, matches_gms;

            Ptr<ORB> orb = ORB::create(500);
            //orb->setFastThreshold(0);
            Mat descriptors_tmp;
            orb->detectAndCompute(frame, Mat(), out->points, descriptors_tmp);

            vector<Mat> vec;
            for (int i=0; i< descriptors_tmp.rows; i++)
            {
                out->descriptors.push_back(descriptors_tmp.row(i));
            }
            auto firstFrame = (in == nullptr);
            out->frame = frame.clone();
            if(firstFrame) {
                return out;
            }

#ifdef USE_GPU

            static Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(in.descriptors, out.descriptors, matches_all);
#else
            static BFMatcher matcher(NORM_HAMMING);
            Mat a, b;
            vconcat(in->descriptors, a);
            vconcat(out->descriptors, b);
            matcher.match(a, b, matches_all);
#endif

            // GMS filter
            vector<bool> vbInliers;
            auto size = frame.size();
            gms_matcher gms(in->points, size, out->points, size, matches_all);
            int num_inliers = gms.GetInlierMask(vbInliers, false, false);
            cout << "Get total " << num_inliers << " matches." << endl;

            // collect matches
            for (size_t i = 0; i < vbInliers.size(); ++i)
            {
                if (vbInliers[i] == true)
                {
                    matches_gms.push_back(matches_all[i]);
                }
            }
            //-- Localize the object
            vector<KeyPoint> outGoodPoints;
            vector<Mat> outGoodDescriptors;

            vector<KeyPoint> inGoodPoints;
             vector<Mat>inGoodDescriptors;

            for( size_t i = 0; i < matches_gms.size(); i++ )
            {
                inGoodPoints.push_back(in->points[matches_gms[i].queryIdx]);
                inGoodDescriptors.push_back(in->descriptors[matches_gms[i].queryIdx]);

                outGoodPoints.push_back(out->points[ matches_gms[i].trainIdx ]);
                outGoodDescriptors.push_back(out->descriptors[ matches_gms[i].trainIdx ]);
            }
            out->points = outGoodPoints;
            out->descriptors = outGoodDescriptors;
            in->points = inGoodPoints;
            in->descriptors = inGoodDescriptors;

            return out;
        }

        void  RestrictPointsKnn(Size size,  shared_ptr<FrameDescriptor>  in, shared_ptr<FrameDescriptor> out,
                                                     int pointsCount = 9){
            // Get matches that are near to point
            auto & inPts = in->GetPointsOnFrame();
            auto & outPts = out->GetPointsOnFrame();
            flann::KDTreeIndexParams indexParams;
            flann::Index kdtree(Mat(outPts).reshape(1), indexParams);
            vector<float> query;
            query.push_back( size.width/4 + rand() % (size.width / 2) );
            query.push_back( size.height/4 + rand() % (size.height / 2 ));
            vector<int> indices;
            vector<float> dists;

            vector<Point2f> inGoodPoints;
            vector<Point2f> outGoodPoints;
            kdtree.knnSearch(query, indices, dists, pointsCount);


            for( size_t i = 0; i < pointsCount; i++ ) {
                inGoodPoints.push_back(inPts[i]);
                outGoodPoints.push_back(outPts[i]);
            }
            in->SetPointsOnFrame(inGoodPoints);
            out->SetPointsOnFrame(outGoodPoints);
        }

        struct MoveDescriptor {
            Eigen::Vector3f Transition;
            Eigen::Quaternionf Rotation;
            MoveDescriptor():
                    Transition(0,0,0),
                    Rotation(0,0,0,0)
            {}
            MoveDescriptor(shared_ptr<FrameDescriptor> from, shared_ptr<FrameDescriptor> to,  Mat &K){
                RestrictPointsKnn(from->frame.size(), from, to, 9);
                auto & inPts = from->GetPointsOnFrame();
                auto & outPts = to->GetPointsOnFrame();
                
                cout << "m1" << endl;
                auto F = findFundamentalMat(inPts, outPts, CV_FM_8POINT);

                cout << "m2" << endl;
                Mat E;
                essentialFromFundamental(F, K, K, E);

                cout << "m3" << endl;
                Mat R, T;
                recoverPose(E, inPts, outPts,K, R, T);

                cout << "m4" << endl;

                Transition = Eigen::Vector3f(T.at<float>(0),T.at<float>(1),T.at<float>(2));

                Rotation = GetQuaternion(R);
            }

            Sophus::SE3f GetPose() {
                return Sophus::SE3f(Rotation, Transition);
            }


        };


        TEST(EpipolarGeometryTest, minDepthProjectionXTranslate1) {
            char exe_str[200];
            readlink("/proc/self/exe", exe_str, 200);

            fs::path exe_path(exe_str);

            string base_dir = exe_path.parent_path().string();

            VideoCapture capture("/flame/v.avi");
            auto K = ReadCameraC("/flame/camera_c.txt");


            if( !capture.isOpened() )
            {
                throw runtime_error("Could not initialize capturing from file");
            }

            cv::Mat3b frame;
            capture >> frame;
            Mat Kinv = K.inv();
            Matrix3f Ke, Keinv;
            cv2eigen(K, Ke);
            cv2eigen(Kinv, Keinv);
            auto fsize = frame.size();
            flame::Flame f( fsize.width,
                            fsize.height,
                            Ke,
                            Keinv );

            VideoWriter videoOutWireframe("/flame/out_wires.avi",CV_FOURCC('F','M','P','4'),10, fsize);
            VideoWriter videoOutDepth("/flame/out_depth.avi",CV_FOURCC('F','M','P','4'),10, fsize);

            auto time = 0;

            // Detect and track Points
            // Reconstruct pos/rot https://docs.opencv.org/3.4/da/db5/group__reconstruction.html#gaadb8cc60069485cbb9273b1efebd757d

            cv::Mat1b bw_frame;
            cv::cvtColor(frame, bw_frame, CV_BGR2GRAY);
            auto lastFrame = make_shared<FrameDescriptor>();
            auto currentFrame = make_shared<FrameDescriptor>();
            lastFrame = GetDescriptor(bw_frame);
            for(;;) {
                cv::Mat1b bw_newframe;

                cout << "+" << endl;
                capture >> frame;
                if (frame.empty()) {
                    break;
                }
                auto t_start = chrono::high_resolution_clock::now();

                cv::cvtColor(frame, bw_newframe, CV_BGR2GRAY);
                currentFrame = GetDescriptor(bw_newframe, lastFrame);
                cout << "2" << endl;

                MoveDescriptor md(lastFrame, currentFrame, K);
                cout << "3" << endl;

                cv::Mat1f depth;
                cout << "pos: " << md.Transition << endl
                     << "rot: " << md.Rotation.vec() << endl;

                Sophus::SE3f T_new = md.GetPose();
                auto t_end = chrono::high_resolution_clock::now();
                cout << "getting pos/rot ready took: " <<  chrono::duration<double, milli>(t_end-t_start).count() << " milliseconds" << endl;
                // Feed to Flame
                f.update(time, time, T_new, bw_newframe, true, depth);

                auto wf = f.getDebugImageWireframe();
                auto idm = f.getDebugImageInverseDepthMap();
                videoOutWireframe.write(wf);
                videoOutDepth.write(idm);

                lastFrame = currentFrame;
                cout << "frame " << ++time << endl;
            }

            videoOutWireframe.release();
            videoOutDepth.release();

            EXPECT_TRUE(true);

        }

    }  // namespace stereo

}  // namespace flame
