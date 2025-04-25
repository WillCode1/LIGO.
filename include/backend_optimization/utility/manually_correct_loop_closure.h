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

    double manually_adjust_loop_closure(PointCloudType::Ptr submap, PointCloudType::Ptr scan, Eigen::Affine3f &tuningtransform, bool &reject_this_loop)
    {
        // 计算点云边界
        float min_x = std::numeric_limits<float>::max(), max_x = -min_x;
        float min_y = min_x, max_y = -min_x, min_z = min_x, max_z = -min_x;
        for (const auto &point : scan->points)
        {
            min_x = std::min(min_x, point.x);
            max_x = std::max(max_x, point.x);
            min_y = std::min(min_y, point.y);
            max_y = std::max(max_y, point.y);
            min_z = std::min(min_z, point.z);
            max_z = std::max(max_z, point.z);
        }
        Eigen::Vector3f center((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2);
        float size = std::max({max_x - min_x, max_y - min_y, max_z - min_z});
        float distance = 40;

        // 创建Pangolin窗口
        const int window_width = 1920;
        const int window_height = 1080;
        pangolin::CreateWindowAndBind("Loop Closure Tuning", window_width, window_height);
        glEnable(GL_DEPTH_TEST);

        Eigen::Vector3f eye(center.x(), center.y(), center.z() + distance);
        Eigen::Vector3f look = (center - eye).normalized();
        Eigen::Vector3f up(0, 1, 0);

        // 计算3D视图中心
        const float ui_width = 180.0f; // UI面板固定宽度
        float view_width = window_width - ui_width;
        float u0 = ui_width + view_width / 2;
        float v0 = window_height / 2;

        // 设置相机
        pangolin::OpenGlRenderState s_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(window_width, window_height, 420, 420, u0, v0, 0.1, size * 10),
            pangolin::ModelViewLookAt(eye.x(), eye.y(), eye.z(),
                                      center.x(), center.y(), center.z(),
                                      up.x(), up.y(), up.z()));

        // 创建3D显示区域
        pangolin::View &d_cam = pangolin::CreateDisplay()
                                    .SetBounds(0.0, 1.0, ui_width / window_width, 1.0)
                                    .SetHandler(new pangolin::Handler3D(s_cam));

        // 创建UI面板
        pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, ui_width / window_width);

        // 添加滑块
        pangolin::Var<float> trans_x("ui.X Trans", 0.0f, -5.0f, 5.0f);
        pangolin::Var<float> trans_y("ui.Y Trans", 0.0f, -5.0f, 5.0f);
        pangolin::Var<float> trans_z("ui.Z Trans", 0.0f, -1.0f, 1.0f);
        pangolin::Var<float> rot_x("ui.Roll (rad)", 0.0f, RAD2DEG(-0.2f), RAD2DEG(0.2f));
        pangolin::Var<float> rot_y("ui.Pitch (rad)", 0.0f, RAD2DEG(-0.2f), RAD2DEG(0.2f));
        pangolin::Var<float> rot_z("ui.Yaw (rad)", 0.0f, RAD2DEG(-3.14f), RAD2DEG(3.14f));
        pangolin::Var<bool> reset("ui.Reset", false, false);
        pangolin::Var<float> fitness_score("ui.Score", 1.0f);
        pangolin::Var<bool> screenshot("ui.Screen Shot", false, false);
        pangolin::Var<bool> reject_loop("ui.Reject loop", false, false);

        bool first_run = true;
        while (!pangolin::ShouldQuit())
        {
            // 清除屏幕
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            if (first_run || reset.GuiChanged())
            {
                trans_x = 0.0f;
                trans_y = 0.0f;
                trans_z = 0.0f;
                rot_x = 0.0f;
                rot_y = 0.0f;
                rot_z = 0.0f;
                reset = false;
                first_run = false;
            }

#if 0
            // 检测截屏按钮
            if (screenshot.GuiChanged())
            {
                std::time_t now = std::time(nullptr);
                std::stringstream ss;
                ss << "./screenshot_" << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S") << ".png";
                d_cam.SaveRenderNow(ss.str());
                std::cout << "截屏已保存为 " << ss.str() << std::endl;
            }
#endif

            if (reject_loop.GuiChanged())
            {
                reject_this_loop = true;
            }

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
            tuningtransform.setIdentity();
            tuningtransform.translation() = Eigen::Vector3f(trans_x, trans_y, trans_z);
            tuningtransform.rotate(Eigen::AngleAxisf(DEG2RAD(rot_z), Eigen::Vector3f::UnitZ()) *
                                   Eigen::AngleAxisf(DEG2RAD(rot_y), Eigen::Vector3f::UnitY()) *
                                   Eigen::AngleAxisf(DEG2RAD(rot_x), Eigen::Vector3f::UnitX()));

            PointCloudType::Ptr transformed_scan(new PointCloudType);
            pcl::transformPointCloud(*scan, *transformed_scan, tuningtransform);

            // 绘制变换后的扫描点云（绿色）
            glColor3f(0.0f, 1.0f, 0.0f);
            glBegin(GL_POINTS);
            for (const auto &point : transformed_scan->points)
            {
                glVertex3f(point.x, point.y, point.z);
            }
            glEnd();

            fitness_score = getFitnessScore(submap, scan, tuningtransform, 2);

            // 完成帧渲染
            pangolin::FinishFrame();
        }

        pangolin::DestroyWindow("Loop Closure Tuning");
        pcl::getTransformation(trans_x, trans_y, trans_z, DEG2RAD(rot_x), DEG2RAD(rot_y), DEG2RAD(rot_z), tuningtransform);
        return fitness_score;
    }

public:
    std::vector<float> trans_state = std::vector<float>(6, 0);
    std::vector<float> offset = std::vector<float>(6, 0.1);
    std::vector<float> trans_state_backup;
    int trans_state_index = 0;
    int offset_index = 0;
};
