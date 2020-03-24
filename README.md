# AutoNav

[![CircleCI](https://circleci.com/gh/OlegJakushkin/autonav/tree/master.svg?style=shield)](https://circleci.com/gh/OlegJakushkin/autonav/tree/master)

**GMS -> OpenCV -> FLaME -> Local NAV -> GPS/GSM Positioning -> Global Nav**

# Plan
We take a JetsonNano, stick a web camera into it
 1. Get features (Currently, We Are Here)
 2. Get camera pos difference
 3. FLaME to get depth
 4. First Project Priority - Local Obstacle avoidance
 5. Add GPS alike device positioning coordinates for global pathfinding
 6. Test different speed modes

A [youtube playlist](https://www.youtube.com/watch?v=TeQ7rxJ4UAQ&list=PLoDvqBmgo3AEqM35S4k0xp-3VEEE6wvGC) in russian covering our dev. process.

# References
We rely on these articles for a base understanding of concepts and capabilities:
 1. [Comparative evaluation of 2D feature correspondence selection algorithms](https://www.groundai.com/project/comparative-evaluation-of-2d-feature-correspondence-selection-algorithms/)
 2. [Features2D + Homography to find a known object](https://docs.opencv.org/3.4/d7/dff/tutorial_feature_homography.html)
 3. [How to compute camera pose from Homography matrix?](https://dsp.stackexchange.com/questions/1484/how-to-compute-camera-pose-from-homography-matrix)

We heavily modify code from projects:
 1. [FLaME: Fast Lightweight Mesh Estimation (GNU General Public License v3.0)](https://github.com/robustrobotics/flame_ros.git)
 2. [GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence (BSD 3-Clause "New" or "Revised" License)](https://github.com/JiawangBian/GMS-Feature-Matcher)
 
