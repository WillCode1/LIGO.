#pragma once
#include <pcl/registration/gicp.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pangolin/pangolin.h>
#include "../Header.h"

class ManuallyCorrectLoopClosure
{
public:
    double getFitnessScore(PointCloudType::Ptr submap, PointCloudType::Ptr scan, const Eigen::Affine3f &trans, double max_range)
    {
        double fitness_score = 0.0;

        // Transform the input dataset using the final transformation
        PointCloudType::Ptr scan_tmp(new PointCloudType);
        pcl::transformPointCloud(*scan, *scan_tmp, trans);

        pcl::search::KdTree<PointType> kdtree;
        kdtree.setInputCloud(submap);

        std::vector<int> nn_indices(1);
        std::vector<float> nn_dists(1);

        // For each point in the source dataset
        int nr = 0;
        for (size_t i = 0; i < scan_tmp->points.size(); ++i)
        {
            // Find its nearest neighbor in the target
            kdtree.nearestKSearch(scan_tmp->points[i], 1, nn_indices, nn_dists);

            // Deal with occlusions (incomplete targets)
            if (nn_dists[0] <= max_range)
            {
                // Add to the fitness score
                fitness_score += nn_dists[0];
                nr++;
            }
        }

        if (nr > 0)
            return (fitness_score / nr);
        else
            return (std::numeric_limits<double>::max());
    }

    // 键盘事件回调函数
    void keyboardEventCallback(const pcl::visualization::KeyboardEvent &event, void *viewer_void)
    {
        pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *>(viewer_void);

        if (event.getKeySym() == "q" && event.keyDown())
        {
            viewer->close();
        }
        else if (event.getKeySym() == "r" && event.keyDown())
        {
            trans_state = trans_state_backup;
        }
        else if (event.getKeySym() == "s" && event.keyDown())
        {
            viewer->saveScreenshot(string(ROOT_DIR) + "/screenshot.png");
        }
        else if (event.getKeySym() == "exclam" && event.keyDown())
        {
            offset_index = 0;
        }
        else if (event.getKeySym() == "at" && event.keyDown())
        {
            offset_index = 1;
        }
        else if (event.getKeySym() == "numbersign" && event.keyDown())
        {
            offset_index = 2;
        }
        else if (event.getKeySym() == "dollar" && event.keyDown())
        {
            offset_index = 3;
        }
        else if (event.getKeySym() == "percent" && event.keyDown())
        {
            offset_index = 4;
        }
        else if (event.getKeySym() == "asciicircum" && event.keyDown())
        {
            offset_index = 5;
        }
        else if (event.getKeySym() == "1")
        {
            trans_state_index = 0;
        }
        else if (event.getKeySym() == "2")
        {
            trans_state_index = 1;
        }
        else if (event.getKeySym() == "3")
        {
            trans_state_index = 2;
        }
        else if (event.getKeySym() == "4")
        {
            trans_state_index = 3;
        }
        else if (event.getKeySym() == "5")
        {
            trans_state_index = 4;
        }
        else if (event.getKeySym() == "6")
        {
            trans_state_index = 5;
        }
        else if (event.getKeySym() == "Up" && event.keyDown() && event.isShiftPressed())
        {
            offset[offset_index] += 0.01;
        }
        else if (event.getKeySym() == "Down" && event.keyDown() && event.isShiftPressed())
        {
            offset[offset_index] -= 0.01;
        }
        else if (event.getKeySym() == "Up" && event.keyDown())
        {
            trans_state[trans_state_index] += offset[trans_state_index];
        }
        else if (event.getKeySym() == "Down" && event.keyDown())
        {
            trans_state[trans_state_index] -= offset[trans_state_index];
        }
    }

    double manually_adjust_loop_closure(PointCloudType::Ptr submap, PointCloudType::Ptr scan, Eigen::Affine3f &transform)
    {
        // 创建Pangolin窗口
        pangolin::CreateWindowAndBind("Point Cloud Tuning", 1280, 720);
        glEnable(GL_DEPTH_TEST);

        // 设置相机
        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1280, 720, 420, 420, 640, 360, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -2, -2, 0, 0, 0, pangolin::AxisZ));

        // 创建3D显示区域
        pangolin::View &d_cam = pangolin::CreateDisplay()
                                    .SetBounds(0.0, 1.0, 0.0, 1.0)
                                    .SetHandler(new pangolin::Handler3D(s_cam));

        // 创建UI面板
        pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, 180.0f / 720.0f);

        // 添加滑块
        pangolin::Var<float> trans_x("ui.X Translation", 0.0f, -1.0f, 1.0f);
        pangolin::Var<float> trans_y("ui.Y Translation", 0.0f, -1.0f, 1.0f);
        pangolin::Var<float> trans_z("ui.Z Translation", 0.0f, -1.0f, 1.0f);
        pangolin::Var<float> rot_z("ui.Z Rotation (rad)", 0.0f, -3.14f, 3.14f);

        // 主循环
        while (!pangolin::ShouldQuit())
        {
            // 清除屏幕
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // 激活3D显示
            d_cam.Activate(s_cam);

            // 绘制子地图（红色）
            glColor3f(1.0f, 0.0f, 0.0f);
            glBegin(GL_POINTS);
            for (const auto &point : submap->points)
            {
                glVertex3f(point.x, point.y, point.z);
            }
            glEnd();

            // 计算变换
            Eigen::Affine3f transform = Eigen::Affine3f::Identity();
            transform.translation() = Eigen::Vector3f(trans_x, trans_y, trans_z);
            transform.rotate(Eigen::AngleAxisf(rot_z, Eigen::Vector3f::UnitZ()));

            // 变换扫描点云
            PointCloudType::Ptr transformed_scan(new PointCloudType);
            pcl::transformPointCloud(*scan, *transformed_scan, transform);

            // 绘制变换后的扫描点云（绿色）
            glColor3f(0.0f, 1.0f, 0.0f);
            glBegin(GL_POINTS);
            for (const auto &point : transformed_scan->points)
            {
                glVertex3f(point.x, point.y, point.z);
            }
            glEnd();

            // 计算并打印适配度评分
            if (pangolin::GuiVarHasChanged())
            {
                double fs = getFitnessScore(submap, scan, transform, 2);
                std::cout << "适配度评分: " << fs << std::endl;
                // pangolin::Text("Hello, Pangolin!", 10, 10);
                // pangolin::default_font().Text("你好，世界！").Draw(10, 10, 0);
            }

            // 完成帧渲染
            pangolin::FinishFrame();
        }

        pangolin::DestroyWindow("Point Cloud Tuning");
        return 0;
    }

public:
    std::vector<float> trans_state = std::vector<float>(6, 0);
    std::vector<float> offset = std::vector<float>(6, 0.1);
    std::vector<float> trans_state_backup;
    int trans_state_index = 0;
    int offset_index = 0;
};
