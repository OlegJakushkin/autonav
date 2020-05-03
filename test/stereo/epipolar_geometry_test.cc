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

#include <unistd.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "gtest/gtest.h"

#include "flame/stereo/epipolar_geometry.h"
#include <boost/filesystem.hpp>
#include "flame/pose/gms_matcher.h"
#include "flame/flame.h"
#include <opencv2/core/eigen.hpp>
//TodO add sfm
//#include <opencv2/sfm.hpp>
//#include <opencv2/sfm/reconstruct.hpp>
#include <sophus/se3.hpp>
namespace fs = boost::filesystem;
using namespace cv;

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
            std::vector<bool> vbInliers;
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
            std::vector<Point2f> obj;
            std::vector<Point2f> scene;
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

        struct FrameDescriptor {
            Mat1b frame;
#ifdef USE_GPU
            vector<GpuMat> descriptors;
#else
            vector<Mat> descriptors;
#endif

            vector<KeyPoint> points;
        };


        FrameDescriptor GetDescriptor(Mat1b & frame, FrameDescriptor & inOut) {
            FrameDescriptor out;
            if (frame.empty()) {
                return out;
            }

            vector<DMatch> matches_all, matches_gms;

            Ptr<ORB> orb = ORB::create(10000);
            orb->setFastThreshold(0);
            orb->detectAndCompute(frame, Mat(), out.points, out.descriptors);

            auto firstFrame = inOut.frame.empty();
            out.frame = frame;
            if(firstFrame ) {
                return out;
            }

#ifdef USE_GPU

            static Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(in.descriptors, out.descriptors, matches_all);
#else
            static BFMatcher matcher(NORM_HAMMING);
            matcher.match(inOut.descriptors, out.descriptors, matches_all);
#endif

            // GMS filter
            std::vector<bool> vbInliers;
            auto size = frame.size();
            gms_matcher gms(inOut.points, size, out.points, size, matches_all);
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
            vector<Mat> inGoodDescriptors;


            for( size_t i = 0; i < matches_gms.size(); i++ )
            {
                inGoodPoints.push_back(inOut.points[ matches_gms[i].queryIdx]);
                inGoodDescriptors.push_back(inOut.descriptors[ matches_gms[i].queryIdx]);

                outGoodPoints.push_back(out.points[ matches_gms[i].trainIdx ]);
                outGoodDescriptors.push_back(out.descriptors[ matches_gms[i].trainIdx ]);
            }
            out.points = outGoodPoints;
            out.descriptors = outGoodDescriptors;

            inOut.points = inGoodPoints;
            inOut.descriptors = inGoodDescriptors;
            return out;
        }


        vector< vector<Point2f> >  RestriktPointsKnn(Size size, std::vector<std::vector<Point2f> > frames, int pointsCount=9){
            // Get matches that are near to point
            auto & obj = frames[0];
            flann::KDTreeIndexParams indexParams;
            flann::Index kdtree(Mat(obj).reshape(1), indexParams);
            vector<float> query;
            query.push_back( size.width/4 + rand() % (size.width / 2) );
            query.push_back( size.height/4 + rand() % (size.height / 2 ));
            vector<int> indices;
            vector<float> dists;

            kdtree.knnSearch(query, indices, dists, pointsCount);

            auto fsize = frames.size();
            vector< vector<Point2f> > result(fsize);

            for( size_t i = 0; i < pointsCount; i++ ) {
                for(size_t j = 0; j < fsize; j++) {
                    result[j].push_back(frames[j][i]);
                }
            }
            return result;
        }

        vector< vector<Point2f> >  GetPointTracks( vector<FrameDescriptor> & frames){
            auto fsize = frames.size();
            auto psize = frames[0].points.size();
            vector< vector<Point2f> > fromFrames(fsize);
            for( size_t i = 0; i < fsize; i++ ) {
                for(size_t j = 0; j < psize; j++) {
                    fromFrames[i].push_back(frames[i].points[j].pt);
                }
            }
            auto frameSize = frames[0].frame.size();
            auto result = RestriktPointsKnn(frameSize, fromFrames);
        }

        struct MoveDescriptor {
            Eigen::Vector3f Transition;
            Eigen::Quaternionf Rotation;
            MoveDescriptor():
                    Transition(0,0,0),
                    Rotation(0,0,0,0)
            {}

            MoveDescriptor(FrameDescriptor from, FrameDescriptor to){
                vector<FrameDescriptor > fd = {from, to};
                auto tracks = GetPointTracks(fd);
                //ToDo: cv::sfm::reconstruct();
                // Todo - fill t and q
                // Use https://gist.github.com/shubh-agrawal/76754b9bfb0f4143819dbd146d15d4c8

            }

            Sophus::SE3f GetPose() {
                return Sophus::SE3f(Rotation, Transition);
            }


        };


        TEST(EpipolarGeometryTest, minDepthProjectionXTranslate1) {
            char exe_str[200];
            readlink("/proc/self/exe", exe_str, 200);

            fs::path exe_path(exe_str);

            std::string base_dir = exe_path.parent_path().string();

            VideoCapture capture("v.avi");
            auto K = ReadCameraC("camera_c.txt");


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

            VideoWriter videoOutWireframe("out_wires.avi",CV_FOURCC('F','M','P','4'),10, fsize);
            VideoWriter videoOutDepth("out_depth.avi",CV_FOURCC('F','M','P','4'),10, fsize);

            auto time = 0;

            // Detect and track Points
            // Reconstruct pos/rot https://docs.opencv.org/3.4/da/db5/group__reconstruction.html#gaadb8cc60069485cbb9273b1efebd757d

            cv::Mat1b bw_frame;
            cv::cvtColor(frame, bw_frame, CV_BGR2GRAY);
            FrameDescriptor lastFrame;
            lastFrame = GetDescriptor(bw_frame, lastFrame);

            for(;;) {
                cv::Mat1b bw_newframe;
                cv::cvtColor(frame, bw_newframe, CV_BGR2GRAY);
                auto currentFrame = GetDescriptor(bw_newframe, lastFrame);
                capture >> frame;
                if(frame.empty()){
                    break;
                }

                MoveDescriptor md(lastFrame, currentFrame);
                cv::Mat1f depth;
                Sophus::SE3f T_new = md.GetPose();

                // Feed to Flame
                f.update(time, time, T_new, bw_newframe, true, depth);
                auto wf = f.getDebugImageWireframe();
                auto idm = f.getDebugImageInverseDepthMap();
                videoOutWireframe.write(wf);
                videoOutDepth.write(idm);
            }
            videoOutWireframe.release();
            videoOutDepth.release();

            EXPECT_TRUE(true);

        }

/**
 * \brief Test minDepthProjection with translations in X direction.
 */
        TEST(EpipolarGeometryTest, minDepthProjectionXTranslate2) {
            Eigen::Matrix3f K;
            K << 525, 0, 640.0/2,
                    0, 525, 480.0/2,
                    0, 0, 1;

            cv::Point2f u_ref(320, 0);
            EpipolarGeometry<float> epigeo(K, K.inverse());

            // Positive X translation
            epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f(2, 0, 0));
            cv::Point2f u_min1;
            epigeo.minDepthProjection(u_ref, &u_min1);

            EXPECT_TRUE(u_min1.x > K(0, 2)*2);
            EXPECT_NEAR(0, u_min1.y, 1e-3);

            // Negative X translation.
            epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f(-2, 0, 0));
            cv::Point2f u_min2;
            epigeo.minDepthProjection(u_ref, &u_min2);

            EXPECT_TRUE(u_min2.x < 0);
            EXPECT_NEAR(0, u_min2.y, 1e-3);
        }

/**
 * \brief Test minDepthProjection with translations in Y direction.
 */
        TEST(EpipolarGeometryTest, minDepthProjectionYTranslate1) {
            Eigen::Matrix3f K;
            K << 525, 0, 640.0/2,
                    0, 525, 480.0/2,
                    0, 0, 1;

            cv::Point2f u_ref(320, 240);
            EpipolarGeometry<float> epigeo(K, K.inverse());

            // Positive Y translation
            epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f(0, 2, 0));
            cv::Point2f u_min1;
            epigeo.minDepthProjection(u_ref, &u_min1);

            EXPECT_TRUE(u_min1.y > K(1, 2)*2);
            EXPECT_NEAR(320, u_min1.x, 1e-3);

            // Negative Y translation.
            epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f(0, -2, 0));
            cv::Point2f u_min2;
            epigeo.minDepthProjection(u_ref, &u_min2);

            EXPECT_TRUE(u_min2.y < 0);
            EXPECT_NEAR(320, u_min2.x, 1e-3);
        }

/**
 * \brief Test minDepthProjection with translations in Y direction.
 */
        TEST(EpipolarGeometryTest, minDepthProjectionYTranslate2) {
            Eigen::Matrix3f K;
            K << 525, 0, 640.0/2,
                    0, 525, 480.0/2,
                    0, 0, 1;

            cv::Point2f u_ref(0, 240);
            EpipolarGeometry<float> epigeo(K, K.inverse());

            // Positive Y translation
            epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f(0, 2, 0));
            cv::Point2f u_min1;
            epigeo.minDepthProjection(u_ref, &u_min1);

            EXPECT_TRUE(u_min1.y > K(1, 2)*2);
            EXPECT_NEAR(0, u_min1.x, 1e-3);

            // Negative Y translation.
            epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f(0, -2, 0));
            cv::Point2f u_min2;
            epigeo.minDepthProjection(u_ref, &u_min2);

            EXPECT_TRUE(u_min2.y < 0);
            EXPECT_NEAR(0, u_min2.x, 1e-3);
        }

/**
 * \brief Test minDepthProjection when yawed.
 */
        TEST(EpipolarGeometryTest, minDepthProjection60Yaw) {
            Eigen::Matrix3f K;
            K << 525, 0, 640.0/2,
                    0, 525, 480.0/2,
                    0, 0, 1;

            Eigen::AngleAxisf aa21(-M_PI/3, Eigen::Vector3f::UnitY());
            Eigen::Quaternionf q21(aa21);
            Eigen::Vector3f t21(2, 0, 0);

            Eigen::Quaternionf q12(q21.inverse());
            Eigen::Vector3f t12(-(q12 * t21));

            cv::Point2f u_ref(320, 240);
            EpipolarGeometry<float> epigeo(K, K.inverse());
            double tol = 1e-4;

            // ref = 1, cmp = 2
            epigeo.loadGeometry(q12, t12);

            cv::Point2f u_min1;
            epigeo.minDepthProjection(u_ref, &u_min1);

            EXPECT_NEAR(16.8910904, u_min1.x, tol);
            EXPECT_NEAR(240, u_min1.y, tol);

            // ref = 2, cmp = 1
            epigeo.loadGeometry(q21, t21);

            cv::Point2f u_min2;
            epigeo.minDepthProjection(u_ref, &u_min2);

            EXPECT_NEAR(1049999424, u_min2.x, tol);
            EXPECT_NEAR(240, u_min2.y, tol);
        }

/**
 * \brief Test using real data point where ref cam is in front of cmp cam.
 */
        TEST(EpipolarGeometryTest, minDepthProjectionRefFontCmp) {
            Eigen::Matrix3f K;
            K << 535.43310546875f, 0.0f, 320.106652814575f,
                    0.0f, 539.212524414062f, 247.632132204719f,
                    0.0f, 0.0f, 1.0f;

            Eigen::Quaternionf q_ref_to_cmp(0.999138, -0.000878, 0.041493, 0.000386);
            Eigen::Vector3f t_ref_to_cmp(-0.221092, -0.036134, 0.084099);

            EpipolarGeometry<float> epigeo(K, K.inverse());
            epigeo.loadGeometry(q_ref_to_cmp, t_ref_to_cmp);

            cv::Point2f u_zero;
            epigeo.minDepthProjection(cv::Point2f(320, 240), &u_zero);

            double tol = 1e-2;
            EXPECT_NEAR(-1087.525391, u_zero.x, tol);
            EXPECT_NEAR(15.954912, u_zero.y, tol);
        }

/**
 * \brief Test using real data point where ref cam is behind cmp cam.
 */
        TEST(EpipolarGeometryTest, minDepthProjectionRefBehindCmp) {
            Eigen::Matrix3f K;
            K << 535.43310546875f, 0.0f, 320.106652814575f,
                    0.0f, 539.212524414062f, 247.632132204719f,
                    0.0f, 0.0f, 1.0f;

            Eigen::Quaternionf q_ref_to_cmp(-0.999853, 0.014856, -0.005249, -0.006822);
            Eigen::Vector3f t_ref_to_cmp(-0.258187, 0.040849, -0.054990);

            EpipolarGeometry<float> epigeo(K, K.inverse());
            epigeo.loadGeometry(q_ref_to_cmp, t_ref_to_cmp);

            cv::Point2f u_zero;
            epigeo.minDepthProjection(cv::Point2f(320, 240), &u_zero);

            double tol = 1e-1;
            EXPECT_NEAR(187.65597534179688, u_zero.x, tol);
            EXPECT_NEAR(278.55392456054688, u_zero.y, tol);
        }

/**
 * \brief Test maxDepthProjection when ref and cmp camera have the same pose.
 */
        TEST(EpipolarGeometryTest, maxDepthProjectionIdentity) {
            Eigen::Matrix3f K;
            K << 525, 0, 640/2,
                    0, 525, 480/2,
                    0, 0, 1;
            Eigen::Matrix3f Kinv(K.inverse());

            EpipolarGeometry<float> epigeo(K, Kinv);
            epigeo.loadGeometry(Eigen::Quaternionf::Identity(), Eigen::Vector3f::Zero());

            cv::Point2f u_ref(320, 240);
            cv::Point2f u_cmp;
            epigeo.maxDepthProjection(u_ref, &u_cmp);

            float tol = 1e-3;
            EXPECT_NEAR(u_ref.x, u_cmp.x, tol);
            EXPECT_NEAR(u_ref.y, u_cmp.y, tol);
        }

/**
 * \brief Test maxDepthProjection when yawed.
 */
        TEST(EpipolarGeometryTest, maxDepthProjection30Yaw) {
            Eigen::Matrix3f K;
            K << 525, 0, 640/2,
                    0, 525, 480/2,
                    0, 0, 1;
            Eigen::AngleAxisf aa_right(-M_PI/6, Eigen::Vector3f::UnitY());
            Eigen::Quaternionf q_right(aa_right);

            EpipolarGeometry<float> epigeo(K, K.inverse());
            epigeo.loadGeometry(q_right, Eigen::Vector3f::Zero());

            cv::Point2f u_ref(320, 240);
            cv::Point2f u_cmp;
            epigeo.maxDepthProjection(u_ref, &u_cmp);

            double tol = 1e-4;
            EXPECT_NEAR(16.891090393066406, u_cmp.x, tol);
            EXPECT_NEAR(240, u_cmp.y, tol);
        }

/**
 * \brief Test maxDepthProjection when rolled.
 */
        TEST(EpipolarGeometryTest, maxDepthProjection30Roll) {
            Eigen::Matrix3f K;
            K << 525, 0, 640/2,
                    0, 525, 480/2,
                    0, 0, 1;
            Eigen::AngleAxisf aa_right(-M_PI/6, Eigen::Vector3f::UnitX());
            Eigen::Quaternionf q_right(aa_right);

            EpipolarGeometry<float> epigeo(K, K.inverse());
            epigeo.loadGeometry(q_right, Eigen::Vector3f::Zero());

            cv::Point2f u_ref(320, 240);
            cv::Point2f u_cmp;
            epigeo.maxDepthProjection(u_ref, &u_cmp);

            double tol = 1e-4;
            EXPECT_NEAR(320, u_cmp.x, tol);
            EXPECT_NEAR(543.10888671875, u_cmp.y, tol);
        }

        TEST(EpipolarGeometryTest, epiline60Yaw) {
            Eigen::Matrix3f K;
            K << 525, 0, 640/2,
                    0, 525, 480/2,
                    0, 0, 1;

            Eigen::AngleAxisf aa_right_to_left(-M_PI/3, Eigen::Vector3f::UnitY());
            Eigen::Quaternionf q_right_to_left(aa_right_to_left);
            Eigen::Vector3f t_right_to_left(2, 0, 0);

            Eigen::Quaternionf q_left_to_right(aa_right_to_left.inverse());
            Eigen::Vector3f t_left_to_right(-(q_right_to_left * t_right_to_left));

            EpipolarGeometry<float> epigeo(K, K.inverse());
            epigeo.loadGeometry(q_left_to_right, t_left_to_right);

            cv::Point2f u_ref(320, 240);
            cv::Point2f u_inf, epi;
            epigeo.epiline(u_ref, &u_inf, &epi);

            double tol = 1e-4;
            EXPECT_NEAR(1, epi.x, tol);
            EXPECT_NEAR(0, epi.y, tol);
        }

        TEST(EpipolarGeometryTest, epiline60Roll) {
            Eigen::Matrix3f K;
            K << 525, 0, 640/2,
                    0, 525, 480/2,
                    0, 0, 1;

            Eigen::AngleAxisf aa_right_to_left(M_PI/3, Eigen::Vector3f::UnitX());
            Eigen::Quaternionf q_right_to_left(aa_right_to_left);
            Eigen::Vector3f t_right_to_left(0, 2, 0);

            Eigen::Quaternionf q_left_to_right(aa_right_to_left.inverse());
            Eigen::Vector3f t_left_to_right(-(q_right_to_left * t_right_to_left));

            EpipolarGeometry<float> epigeo(K, K.inverse());
            epigeo.loadGeometry(q_left_to_right, t_left_to_right);

            cv::Point2f u_ref(320, 240);
            cv::Point2f u_inf, epi;
            epigeo.epiline(u_ref, &u_inf, &epi);

            double tol = 1e-4;
            EXPECT_NEAR(0, epi.x, tol);
            EXPECT_NEAR(1, epi.y, tol);
        }

/**
 * \brief Test point at (1, 0, 10) with cameras 1m apart and yawed -15 deg.
 *
 * T1 = -15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (1, 0, 10)
 */
        TEST(EpipolarGeometryTest, disparityToDepthTest1) {
            Eigen::Matrix3f K;
            K << 525.0f, 0.0f, 320.0f,
                    0.0f, 525.0f, 240.0f,
                    0.0f, 0.0f, 1.0f;
            Eigen::Matrix3f Kinv(K.inverse());

            // Geometry of cameras and landmarks.
            Eigen::AngleAxisf aa(-M_PI/12, Eigen::Vector3f::UnitY());
            Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
            Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                            Eigen::Vector3f(1.0f, 0.0f, 0.0f));
            Eigen::Vector3f p_world(1.0f, 0.0f, 10.0f);

            // Project into cameras.
            Sophus::SE3f T12(T2.inverse() * T1);
            Sophus::SE3f T21(T1.inverse() * T2);

            cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                              T1.translation(),
                                                              p_world);
            cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                              T2.translation(),
                                                              p_world);

            cv::Point2f u_inf, epi;
            float tol = 1e-4;

            // Depth from camera 1.
            EpipolarGeometry<float> epigeo1(K, Kinv);
            epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
            float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
            float depth1 = epigeo1.disparityToDepth(u1, u_inf, epi, disp1);
            EXPECT_NEAR((T1.inverse() * p_world)(2), depth1, tol);

            // Depth from camera 2.
            EpipolarGeometry<float> epigeo2(K, Kinv);
            epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
            float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
            float depth2 = epigeo2.disparityToDepth(u2, u_inf, epi, disp2);
            EXPECT_NEAR((T2.inverse() * p_world)(2), depth2, tol);
        }

/**
 * \brief Test point at (-1, 0, 10) with cameras 1m apart and yawed -15 deg.
 *
 * T1 = -15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (-1, 0, 10)
 */
        TEST(EpipolarGeometryTest, disparityToDepthTest2) {
            Eigen::Matrix3f K;
            K << 525.0f, 0.0f, 320.0f,
                    0.0f, 525.0f, 240.0f,
                    0.0f, 0.0f, 1.0f;
            Eigen::Matrix3f Kinv(K.inverse());

            // Geometry of cameras and landmarks.
            Eigen::AngleAxisf aa(-M_PI/12, Eigen::Vector3f::UnitY());
            Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
            Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                            Eigen::Vector3f(1.0f, 0.0f, 0.0f));
            Eigen::Vector3f p_world(-1.0f, 0.0f, 10.0f);

            // Project into cameras.
            Sophus::SE3f T12(T2.inverse() * T1);
            Sophus::SE3f T21(T1.inverse() * T2);

            cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                              T1.translation(),
                                                              p_world);
            cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                              T2.translation(),
                                                              p_world);

            cv::Point2f u_inf, epi;
            float tol = 1e-4;

            // Depth from camera 1.
            EpipolarGeometry<float> epigeo1(K, Kinv);
            epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
            float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
            float depth1 = epigeo1.disparityToDepth(u1, u_inf, epi, disp1);
            EXPECT_NEAR((T1.inverse() * p_world)(2), depth1, tol);

            // Depth from camera 2.
            EpipolarGeometry<float> epigeo2(K, Kinv);
            epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
            float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
            float depth2 = epigeo2.disparityToDepth(u2, u_inf, epi, disp2);
            EXPECT_NEAR((T2.inverse() * p_world)(2), depth2, tol);
        }

/**
 * \brief Test point at (0, 1, 10) with cameras 1m apart and yawed 15 deg.
 *
 * T1 = 15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (0, 1, 10)
 */
        TEST(EpipolarGeometryTest, disparityToDepthTest3) {
            Eigen::Matrix3f K;
            K << 525.0f, 0.0f, 320.0f,
                    0.0f, 525.0f, 240.0f,
                    0.0f, 0.0f, 1.0f;
            Eigen::Matrix3f Kinv(K.inverse());

            // Geometry of cameras and landmarks.
            Eigen::AngleAxisf aa(M_PI/12, Eigen::Vector3f::UnitY());
            Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
            Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                            Eigen::Vector3f(1.0f, 0.0f, 0.0f));
            Eigen::Vector3f p_world(0.0f, 1.0f, 10.0f);

            // Project into cameras.
            Sophus::SE3f T12(T2.inverse() * T1);
            Sophus::SE3f T21(T1.inverse() * T2);

            cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                              T1.translation(),
                                                              p_world);
            cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                              T2.translation(),
                                                              p_world);

            cv::Point2f u_inf, epi;
            float tol = 1e-4;

            // Depth from camera 1.
            EpipolarGeometry<float> epigeo1(K, Kinv);
            epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
            float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
            float depth1 = epigeo1.disparityToDepth(u1, u_inf, epi, disp1);
            EXPECT_NEAR((T1.inverse() * p_world)(2), depth1, tol);

            // Depth from camera 2.
            EpipolarGeometry<float> epigeo2(K, Kinv);
            epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
            float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
            float depth2 = epigeo2.disparityToDepth(u2, u_inf, epi, disp2);
            EXPECT_NEAR((T2.inverse() * p_world)(2), depth2, tol);
        }

/**
 * \brief Test point at (0, -1, 10) with cameras 1m apart and yawed 15 deg.
 *
 * T1 = 15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (0, -1, 10)
 */
        TEST(EpipolarGeometryTest, disparityToDepthTest4) {
            Eigen::Matrix3f K;
            K << 525.0f, 0.0f, 320.0f,
                    0.0f, 525.0f, 240.0f,
                    0.0f, 0.0f, 1.0f;
            Eigen::Matrix3f Kinv(K.inverse());

            // Geometry of cameras and landmarks.
            Eigen::AngleAxisf aa(M_PI/12, Eigen::Vector3f::UnitY());
            Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
            Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                            Eigen::Vector3f(1.0f, 0.0f, 0.0f));
            Eigen::Vector3f p_world(0.0f, -1.0f, 10.0f);

            // Project into cameras.
            Sophus::SE3f T12(T2.inverse() * T1);
            Sophus::SE3f T21(T1.inverse() * T2);

            cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                              T1.translation(),
                                                              p_world);
            cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                              T2.translation(),
                                                              p_world);

            cv::Point2f u_inf, epi;
            float tol = 1e-4;

            // Depth from camera 1.
            EpipolarGeometry<float> epigeo1(K, Kinv);
            epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
            float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
            float depth1 = epigeo1.disparityToDepth(u1, u_inf, epi, disp1);
            EXPECT_NEAR((T1.inverse() * p_world)(2), depth1, tol);

            // Depth from camera 2.
            EpipolarGeometry<float> epigeo2(K, Kinv);
            epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
            float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
            float depth2 = epigeo2.disparityToDepth(u2, u_inf, epi, disp2);
            EXPECT_NEAR((T2.inverse() * p_world)(2), depth2, tol);
        }

/**
 * \brief Test point at (1, 0, 10) with cameras 1m apart and yawed -15 deg.
 *
 * T1 = -15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (1, 0, 10)
 */
        TEST(EpipolarGeometryTest, disparityToInverseDepthTest1) {
            Eigen::Matrix3f K;
            K << 525.0f, 0.0f, 320.0f,
                    0.0f, 525.0f, 240.0f,
                    0.0f, 0.0f, 1.0f;
            Eigen::Matrix3f Kinv(K.inverse());

            // Geometry of cameras and landmarks.
            Eigen::AngleAxisf aa(-M_PI/12, Eigen::Vector3f::UnitY());
            Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
            Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                            Eigen::Vector3f(1.0f, 0.0f, 0.0f));
            Eigen::Vector3f p_world(1.0f, 0.0f, 10.0f);

            // Project into cameras.
            Sophus::SE3f T12(T2.inverse() * T1);
            Sophus::SE3f T21(T1.inverse() * T2);

            cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                              T1.translation(),
                                                              p_world);
            cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                              T2.translation(),
                                                              p_world);

            cv::Point2f u_inf, epi;
            float tol = 1e-4;

            // Depth from camera 1.
            EpipolarGeometry<float> epigeo1(K, Kinv);
            epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
            float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
            float idepth1 = epigeo1.disparityToInverseDepth(u1, u_inf, epi, disp1);
            EXPECT_NEAR(1.0f/(T1.inverse() * p_world)(2), idepth1, tol);

            // Depth from camera 2.
            EpipolarGeometry<float> epigeo2(K, Kinv);
            epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
            float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
            float idepth2 = epigeo2.disparityToInverseDepth(u2, u_inf, epi, disp2);
            EXPECT_NEAR(1.0f/(T2.inverse() * p_world)(2), idepth2, tol);
        }

/**
 * \brief Test point at (-1, 0, 10) with cameras 1m apart and yawed -15 deg.
 *
 * T1 = -15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (-1, 0, 10)
 */
        TEST(EpipolarGeometryTest, disparityToInverseDepthTest2) {
            Eigen::Matrix3f K;
            K << 525.0f, 0.0f, 320.0f,
                    0.0f, 525.0f, 240.0f,
                    0.0f, 0.0f, 1.0f;
            Eigen::Matrix3f Kinv(K.inverse());

            // Geometry of cameras and landmarks.
            Eigen::AngleAxisf aa(-M_PI/12, Eigen::Vector3f::UnitY());
            Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
            Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                            Eigen::Vector3f(1.0f, 0.0f, 0.0f));
            Eigen::Vector3f p_world(-1.0f, 0.0f, 10.0f);

            // Project into cameras.
            Sophus::SE3f T12(T2.inverse() * T1);
            Sophus::SE3f T21(T1.inverse() * T2);

            cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                              T1.translation(),
                                                              p_world);
            cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                              T2.translation(),
                                                              p_world);

            cv::Point2f u_inf, epi;
            float tol = 1e-4;

            // Depth from camera 1.
            EpipolarGeometry<float> epigeo1(K, Kinv);
            epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
            float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
            float idepth1 = epigeo1.disparityToInverseDepth(u1, u_inf, epi, disp1);
            EXPECT_NEAR(1.0f/(T1.inverse() * p_world)(2), idepth1, tol);

            // Depth from camera 2.
            EpipolarGeometry<float> epigeo2(K, Kinv);
            epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
            float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
            float idepth2 = epigeo2.disparityToInverseDepth(u2, u_inf, epi, disp2);
            EXPECT_NEAR(1.0f/(T2.inverse() * p_world)(2), idepth2, tol);
        }

/**
 * \brief Test point at (0, 1, 10) with cameras 1m apart and yawed 15 deg.
 *
 * T1 = 15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (0, 1, 10)
 */
        TEST(EpipolarGeometryTest, disparityToInverseDepthTest3) {
            Eigen::Matrix3f K;
            K << 525.0f, 0.0f, 320.0f,
                    0.0f, 525.0f, 240.0f,
                    0.0f, 0.0f, 1.0f;
            Eigen::Matrix3f Kinv(K.inverse());

            // Geometry of cameras and landmarks.
            Eigen::AngleAxisf aa(M_PI/12, Eigen::Vector3f::UnitY());
            Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
            Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                            Eigen::Vector3f(1.0f, 0.0f, 0.0f));
            Eigen::Vector3f p_world(0.0f, 1.0f, 10.0f);

            // Project into cameras.
            Sophus::SE3f T12(T2.inverse() * T1);
            Sophus::SE3f T21(T1.inverse() * T2);

            cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                              T1.translation(),
                                                              p_world);
            cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                              T2.translation(),
                                                              p_world);

            cv::Point2f u_inf, epi;
            float tol = 1e-4;

            // Depth from camera 1.
            EpipolarGeometry<float> epigeo1(K, Kinv);
            epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
            float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
            float idepth1 = epigeo1.disparityToInverseDepth(u1, u_inf, epi, disp1);
            EXPECT_NEAR(1.0f/(T1.inverse() * p_world)(2), idepth1, 1e-2);

            // Depth from camera 2.
            EpipolarGeometry<float> epigeo2(K, Kinv);
            epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
            float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
            float idepth2 = epigeo2.disparityToInverseDepth(u2, u_inf, epi, disp2);
            EXPECT_NEAR(1.0f/(T2.inverse() * p_world)(2), idepth2, tol);
        }

/**
 * \brief Test point at (0, -1, 10) with cameras 1m apart and yawed 15 deg.
 *
 * T1 = 15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (0, -1, 10)
 */
        TEST(EpipolarGeometryTest, disparityToInverseDepthTest4) {
            Eigen::Matrix3f K;
            K << 525.0f, 0.0f, 320.0f,
                    0.0f, 525.0f, 240.0f,
                    0.0f, 0.0f, 1.0f;
            Eigen::Matrix3f Kinv(K.inverse());

            // Geometry of cameras and landmarks.
            Eigen::AngleAxisf aa(M_PI/12, Eigen::Vector3f::UnitY());
            Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
            Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                            Eigen::Vector3f(1.0f, 0.0f, 0.0f));
            Eigen::Vector3f p_world(0.0f, -1.0f, 10.0f);

            // Project into cameras.
            Sophus::SE3f T12(T2.inverse() * T1);
            Sophus::SE3f T21(T1.inverse() * T2);

            cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                              T1.translation(),
                                                              p_world);
            cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                              T2.translation(),
                                                              p_world);

            cv::Point2f u_inf, epi;
            float tol = 1e-4;

            // Depth from camera 1.
            EpipolarGeometry<float> epigeo1(K, Kinv);
            epigeo1.loadGeometry(T12.unit_quaternion(), T12.translation());
            float disp1 = epigeo1.disparity(u1, u2, &u_inf, &epi);
            float idepth1 = epigeo1.disparityToInverseDepth(u1, u_inf, epi, disp1);
            EXPECT_NEAR(1.0f/(T1.inverse() * p_world)(2), idepth1, 1e-2);

            // Depth from camera 2.
            EpipolarGeometry<float> epigeo2(K, Kinv);
            epigeo2.loadGeometry(T21.unit_quaternion(), T21.translation());
            float disp2 = epigeo2.disparity(u2, u1, &u_inf, &epi);
            float idepth2 = epigeo2.disparityToInverseDepth(u2, u_inf, epi, disp2);
            EXPECT_NEAR(1.0f/(T2.inverse() * p_world)(2), idepth2, tol);
        }


/**
 * \brief Test point at (1, 0, 10) with cameras 1m apart and yawed -15 deg.
 *
 * Test projecting point between two cameras.
 *
 * T1 = -15 deg yaw
 * T2 = (1, 0, 0) trans
 * p_world = (1, 0, 10)
 */
        TEST(EpipolarGeometryTest, projectTest1) {
            Eigen::Matrix3f K;
            K << 525.0f, 0.0f, 320.0f,
                    0.0f, 525.0f, 240.0f,
                    0.0f, 0.0f, 1.0f;
            Eigen::Matrix3f Kinv(K.inverse());

            // Geometry of cameras and landmarks.
            Eigen::AngleAxisf aa(-M_PI/12, Eigen::Vector3f::UnitY());
            Sophus::SE3f T1(Eigen::Quaternionf(aa), Eigen::Vector3f::Zero());
            Sophus::SE3f T2(Eigen::Quaternionf::Identity(),
                            Eigen::Vector3f(1.0f, 0.0f, 0.0f));
            Eigen::Vector3f p_world(1.0f, 0.0f, 10.0f);

            // Project into cameras.
            Sophus::SE3f T12(T2.inverse() * T1);
            Sophus::SE3f T21(T1.inverse() * T2);

            cv::Point2f u1 = EpipolarGeometry<float>::project(K, T1.unit_quaternion(),
                                                              T1.translation(),
                                                              p_world);
            cv::Point2f u2 = EpipolarGeometry<float>::project(K, T2.unit_quaternion(),
                                                              T2.translation(),
                                                              p_world);

            EpipolarGeometry<float> epigeo(K, Kinv);

            epigeo.loadGeometry(T21.unit_quaternion(), T21.translation());

            cv::Point2f u_cmp = epigeo.project(u2, 1.0f/p_world(2));
            EXPECT_NEAR(u1.x, u_cmp.x, 1e-4);
            EXPECT_NEAR(u1.y, u_cmp.y, 1e-4);
        }

    }  // namespace stereo

}  // namespace flame
