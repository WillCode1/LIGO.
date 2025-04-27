#pragma once
#include <unordered_map>
#include <condition_variable>
#include <atomic>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include "../Header.h"
#include "../global_localization/scancontext/Scancontext.h"
#include "backend_optimization/utility/manually_correct_loop_closure.h"

class LoopClosure
{
public:
    LoopClosure(const std::shared_ptr<ScanContext::SCManager> scManager, std::condition_variable &cv, std::atomic<bool> &isABlocked)
        : cv(cv), isABlocked(isABlocked)
    {
        copy_keyframe_pose6d.reset(new pcl::PointCloud<PointXYZIRPYT>());
        kdtree_history_keyframe_pose.reset(new pcl::KdTreeFLANN<PointXYZIRPYT>());
        kdtree_submap.reset(new pcl::KdTreeFLANN<PointType>());

        unused_result.reset(new PointCloudType());
        prevKeyframeCloud.reset(new PointCloudType());

        loop_vaild_period["odom"] = std::vector<double>();
        loop_vaild_period["scancontext"] = std::vector<double>();
        sc_manager = scManager;
    }

    /**
     * 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合，降采样
     */
    void loop_find_near_keyframes(PointCloudType::Ptr &near_keyframes, const int &key, const int &search_num,
                                  const deque<PointCloudType::Ptr> &keyframe_scan)
    {
        // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
        near_keyframes->clear();
        int cloudSize = copy_keyframe_pose6d->size();
        for (int i = -search_num; i <= search_num; ++i)
        {
            int key_near = key + i;
            if (key_near < 0 || key_near >= cloudSize)
                continue;

            *near_keyframes += *pointcloudKeyframeToWorld(keyframe_scan[key_near], copy_keyframe_pose6d->points[key_near]);
        }

        if (near_keyframes->empty())
            return;

        octreeDownsampling(near_keyframes, near_keyframes, icp_downsamp_size);
    }

    void perform_loop_closure(const deque<PointCloudType::Ptr> &keyframe_scan, int loop_key_cur, int loop_key_ref,
                              const std::string &type, bool use_guess = false, const Eigen::Matrix4f &init_guess = Eigen::Matrix4f::Identity())
    {
        // extract cloud
        PointCloudType::Ptr cur_keyframe_cloud(new PointCloudType());
        PointCloudType::Ptr ref_near_keyframe_cloud(new PointCloudType());
        {
            loop_find_near_keyframes(cur_keyframe_cloud, loop_key_cur, 0, keyframe_scan);
            loop_find_near_keyframes(ref_near_keyframe_cloud, loop_key_ref, keyframe_search_num, keyframe_scan);
            if (cur_keyframe_cloud->size() < 300 || ref_near_keyframe_cloud->size() < 1000)
            {
                return;
            }

            // publish loop submap
            *prevKeyframeCloud = *ref_near_keyframe_cloud;
        }

        // GICP match
        // pcl::GeneralizedIterativeClosestPoint<PointType, PointType> gicp;
        pcl::IterativeClosestPoint<PointType, PointType> gicp;
        gicp.setMaxCorrespondenceDistance(loop_closure_search_radius * 2);
        gicp.setMaximumIterations(1000);
        gicp.setTransformationEpsilon(1e-8);
        gicp.setEuclideanFitnessEpsilon(1e-8);
        gicp.setRANSACIterations(0);

        gicp.setInputSource(cur_keyframe_cloud);
        gicp.setInputTarget(ref_near_keyframe_cloud);
        if (use_guess)
            gicp.align(*unused_result, init_guess);
        else
            gicp.align(*unused_result);

        float loop_closure_fitness_score_thld = 0;
        if (loop_closure_fitness_use_adaptability)
        {
            if (dartion_time - last_loop_time > 40)
            {
                loop_closure_fitness_score_thld = loop_closure_fitness_score_thld_max;
            }
            else
            {
                loop_closure_fitness_score_thld = loop_closure_fitness_score_thld_min + (loop_closure_fitness_score_thld_max - loop_closure_fitness_score_thld_min) * 0.025 * (dartion_time - last_loop_time);
            }
        }
        else
            loop_closure_fitness_score_thld = loop_closure_fitness_score_thld_min;

        if (gicp.hasConverged() == false || gicp.getFitnessScore() > loop_closure_fitness_score_thld)
        {
            LOG_WARN("dartion_time = %.2f.loop closure failed by %s! %d, %.3f, %.3f", dartion_time, type.c_str(), gicp.hasConverged(), gicp.getFitnessScore(), loop_closure_fitness_score_thld);
            return;
        }

        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame, tuningLidarFrame;
        correctionLidarFrame = gicp.getFinalTransformation();
        tuningLidarFrame.setIdentity();
        float noiseScore = gicp.getFitnessScore();

#if 1
        bool reject_this_loop = false;
        if (is_vaild_loop_time_period(dartion_time, loop_vaild_period["manually"]))
        {
            isABlocked.store(true);
            noiseScore = mclc.manually_adjust_loop_closure(ref_near_keyframe_cloud, cur_keyframe_cloud, tuningLidarFrame, reject_this_loop);
            isABlocked.store(false);
            cv.notify_one();
        }
        if (reject_this_loop)
        {
            LOG_ERROR("dartion_time = %.2f. manually reject this loop closure! loop closure failed by %s!", dartion_time, type.c_str());
            return;
        }
#endif

        // Get current frame wrong pose
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_keyframe_pose6d->points[loop_key_cur]);
        // Get current frame corrected pose
        Eigen::Affine3f tCorrect = correctionLidarFrame * tuningLidarFrame * tWrong;
        pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
        // Get reference frame pose
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_keyframe_pose6d->points[loop_key_ref]);
        gtsam::Vector Vector6(6);
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);

        float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0, tmp5 = 0;
        float dis1 = 0, dis2 = 0, dis3 = 0;
        float loop_dis = correctionLidarFrame.translation().norm();

#if 1
        static int index = 0;
        auto cur_p0 = copy_keyframe_pose6d->points[loop_key_cur - 2];
        auto cur_p1 = copy_keyframe_pose6d->points[loop_key_cur - 1];
        auto cur_p2 = copy_keyframe_pose6d->points[loop_key_cur];
        V3F eigen_cp0(cur_p0.x, cur_p0.y, cur_p0.z);
        V3F eigen_cp1(cur_p1.x, cur_p1.y, cur_p1.z);
        V3F eigen_cp2(cur_p2.x, cur_p2.y, cur_p2.z);
        V3F eigen_cp3 = tCorrect.translation();

        auto ref_p1 = copy_keyframe_pose6d->points[loop_key_ref - 1];
        auto ref_p2 = copy_keyframe_pose6d->points[loop_key_ref];
        auto ref_p3 = copy_keyframe_pose6d->points[loop_key_ref + 1];
        V3F eigen_rp1(ref_p1.x, ref_p1.y, ref_p1.z);
        V3F eigen_rp2(ref_p2.x, ref_p2.y, ref_p2.z);
        V3F eigen_rp3(ref_p3.x, ref_p3.y, ref_p3.z);

        V3F direction_cp0 = eigen_cp1 - eigen_cp0;
        V3F direction_cp1 = eigen_cp2 - eigen_cp1;
        V3F direction_cp2 = eigen_cp3 - eigen_cp1;
        V3F direction_rp1 = eigen_rp1 - eigen_rp2;
        V3F direction_rp2 = eigen_rp2 - eigen_rp3;

        if (direction_cp1.norm() > 0.5 && direction_cp2.norm() > 0.5 && direction_rp1.norm() > 0.5 && direction_rp2.norm() > 0.5)
        {
            auto calculateAngle = [](const Eigen::Vector3f &v1, const Eigen::Vector3f &v2) -> float
            {
                float dot = v1.dot(v2);
                float norm_product = v1.norm() * v2.norm();
                float cos_theta = std::abs(dot / norm_product);
                return RAD2DEG(std::acos(cos_theta));
            };

            // 三点拟合直线，返回质心和方向向量
            auto fitLineToThreePoints = [](const Vector3f &p1, const Vector3f &p2, const Vector3f &p3, Vector3f &centroid, Vector3f &direction)
            {
                centroid = (p1 + p2 + p3) / 3.0f;

                Matrix3f centered;
                centered.col(0) = p1 - centroid;
                centered.col(1) = p2 - centroid;
                centered.col(2) = p3 - centroid;

                Matrix3f cov = (centered * centered.transpose()) / 2.0f;
                SelfAdjointEigenSolver<Matrix3f> solver(cov);
                direction = solver.eigenvectors().col(2);
            };

            // 计算点到直线的距离
            auto distanceToLine = [](const Vector3f &point, const Vector3f &centroid, const Vector3f &direction) -> float
            {
                Vector3f vec = point - centroid;
                Vector3f cross = vec.cross(direction);
                return cross.norm() / direction.norm();
            };

            tmp1 = calculateAngle(direction_rp1, direction_rp2);        // 小，ref按直线运动
            tmp2 = calculateAngle(direction_cp0, direction_cp1);        // 小，当前按直线运动
            tmp3 = calculateAngle(direction_cp2, direction_rp1);        // ref和当前运动方向修正后方向，判断修正后方向平行
            tmp4 = calculateAngle(direction_cp1, direction_rp1);        // ref和当前运动方向修正前方向，用于提取跑偏后的情况
            tmp5 = calculateAngle(direction_cp1, direction_cp2);        // 当前修正的delta方向

#if 1
            // if (tmp1 < 2 && tmp2 < 2 && tmp3 > 5 && tmp3 < 18 || tmp5 > 70 && loop_dis > 7 || loop_dis > 30)
            if (tmp5 > 70 && loop_dis > 7 || loop_dis > 30)
            {
                savePCDFile(PCD_FILE_DIR("src/" + to_string(index) + "src.pcd"), *unused_result);
                savePCDFile(PCD_FILE_DIR("tag/" + to_string(index) + "tag.pcd"), *ref_near_keyframe_cloud);
                ++index;
                LOG_ERROR("dartion_time = %.2f. loop_dis = %.2f, a1 = %.2f, a2 = %.2f, a3 = %.2f, a4 = %.2f, a5 = %.2f.",
                          dartion_time, loop_dis, tmp1, tmp2, tmp3, tmp4, tmp5);
                return;
            }
#endif

#if 1
            Vector3f centroid, direction;
            fitLineToThreePoints(eigen_rp1, eigen_rp2, eigen_rp3, centroid, direction);
            dis1 = distanceToLine(eigen_cp1, centroid, direction);
            dis2 = distanceToLine(eigen_cp2, centroid, direction);
            dis3 = distanceToLine(eigen_cp3, centroid, direction);

            if (tmp1 < 2 && tmp2 < 2)
            {
                if ((dis1 > 3 || dis2 > 3) && (tmp4 > 3 && tmp4 < 40) && dis3 < 1)
                {
                    savePCDFile(PCD_FILE_DIR("src/" + to_string(index) + "src.pcd"), *unused_result);
                    savePCDFile(PCD_FILE_DIR("tag/" + to_string(index) + "tag.pcd"), *ref_near_keyframe_cloud);
                    ++index;
                    LOG_ERROR("dartion_time = %.2f. loop_dis = %.2f, a1 = %.2f, a2 = %.2f, a3 = %.2f, a4 = %.2f, a5 = %.2f, dis1 = %.2f, dis2 = %.2f, dis3 = %.2f.",
                              dartion_time, loop_dis, tmp1, tmp2, tmp3, tmp4, tmp5, dis1, dis2, dis3);
                    return;
                }
            }
        }
#endif

#if 1
        int point_in_range_num = 0;
        std::vector<int> indices;
        std::vector<float> distances;
        PointXYZIRPYT pose_correct;
        pose_correct.x = x;
        pose_correct.y = y;
        pose_correct.z = z;
        pose_correct.roll = roll;
        pose_correct.pitch = pitch;
        pose_correct.yaw = yaw;

        kdtree_submap->setInputCloud(ref_near_keyframe_cloud);
        *cur_keyframe_cloud = *pointcloudKeyframeToWorld(keyframe_scan[loop_key_cur], pose_correct);

        for (auto i = 0; i < cur_keyframe_cloud->size(); ++i)
        {
            kdtree_submap->radiusSearch(cur_keyframe_cloud->points[i], 0.5, indices, distances, 1);
            if (distances.size() == 1)
            {
                ++point_in_range_num;
            }
        }
        int points_num_out_of_range = cur_keyframe_cloud->points.size() - point_in_range_num;
        if (points_num_out_of_range > cur_keyframe_cloud->points.size() * 0.2)
        {
            LOG_FATAL("points_num_out_of_range  = %d, total = %lu, outof_per = %f.",
                      points_num_out_of_range, cur_keyframe_cloud->points.size(),
                      points_num_out_of_range * 1.0 / cur_keyframe_cloud->points.size());
            return;
        }
        else
        {
            LOG_INFO("points_num_out_of_range  = %d, total = %lu, outof_per = %f.",
                      points_num_out_of_range, cur_keyframe_cloud->points.size(),
                      points_num_out_of_range * 1.0 / cur_keyframe_cloud->points.size());
        }
#endif

#endif

        last_loop_time = dartion_time;

        loop_mtx.lock();
        loop_constraint.loop_indexs.push_back(make_pair(loop_key_cur, loop_key_ref));
        loop_constraint.loop_pose_correct.push_back(poseFrom.between(poseTo));
        loop_constraint.loop_noise.push_back(constraintNoise);
        loop_mtx.unlock();

        savePCDFile(PCD_FILE_DIR("src/" + to_string(index) + "src.pcd"), *unused_result);
        savePCDFile(PCD_FILE_DIR("tag/" + to_string(index) + "tag.pcd"), *ref_near_keyframe_cloud);
        ++index;
        LOG_INFO("dartion_time = %.2f.Loop Factor Added by %s! keyframe id = %d, noise = %.3f, loop_dis = %.2f, a1 = %.2f, a2 = %.2f, a3 = %.2f, a4 = %.2f, a5 = %.2f, dis1 = %.2f, dis2 = %.2f, dis3 = %.2f.",
                 dartion_time, type.c_str(), loop_key_ref, noiseScore, loop_dis, tmp1, tmp2, tmp3, tmp4, tmp5, dis1, dis2, dis3);
        loop_constraint_records[loop_key_cur] = loop_key_ref;
    }

    void detect_loop_by_distance(const deque<PointCloudType::Ptr> &keyframe_scan)
    {
        int latest_id = copy_keyframe_pose6d->size() - 1; // 当前关键帧索引
        int closest_id = -1;                              // 最近关键帧索引

        // 当前帧已经添加过闭环对应关系，不再继续添加
        auto it = loop_constraint_records.find(latest_id);
        if (it != loop_constraint_records.end())
            return;

        // 在历史关键帧中查找与当前关键帧距离最近的关键帧
        std::vector<int> indices;
        std::vector<float> distances;
        kdtree_history_keyframe_pose->setInputCloud(copy_keyframe_pose6d);
        kdtree_history_keyframe_pose->radiusSearch(copy_keyframe_pose6d->back(), loop_closure_search_radius, indices, distances, 0);
        for (int i = 0; i < (int)indices.size(); ++i)
        {
            int id = indices[i];
            if (abs(id - latest_id) > loop_closure_keyframe_interval)
            {
                closest_id = id;
                break;
            }
        }
        if (closest_id == -1 || latest_id == closest_id)
            return;

        perform_loop_closure(keyframe_scan, latest_id, closest_id, "odom");
    }

    void detect_loop_by_scancontext(const deque<PointCloudType::Ptr> &keyframe_scan)
    {
        int loop_key_cur = copy_keyframe_pose6d->size() - 1;

        auto detectResult = sc_manager->detectLoopClosureID(50); // first: nn index, second: yaw diff
        int loop_key_ref = detectResult.first;
        float sc_yaw_rad = detectResult.second; // sc2右移 <=> lidar左转 <=> 左+sc_yaw_rad

        if (loop_key_ref == -1)
            return;

        const auto &pose_ref = copy_keyframe_pose6d->points[loop_key_ref];
        Eigen::Matrix4f pose_ref_mat = EigenMath::CreateAffineMatrix(V3D(pose_ref.x, pose_ref.y, pose_ref.z), V3D(pose_ref.roll, pose_ref.pitch, pose_ref.yaw + sc_yaw_rad)).cast<float>();
        const auto &pose_cur = copy_keyframe_pose6d->back();
        Eigen::Matrix4f pose_cur_mat = EigenMath::CreateAffineMatrix(V3D(pose_cur.x, pose_cur.y, pose_cur.z), V3D(pose_cur.roll, pose_cur.pitch, pose_cur.yaw)).cast<float>();

        perform_loop_closure(keyframe_scan, loop_key_cur, loop_key_ref, "scancontext", true, pose_cur_mat.inverse() * pose_ref_mat);
    }

    void run(const deque<PointCloudType::Ptr> &keyframe_scan)
    {
        if (copy_keyframe_pose6d->points.size() < loop_keyframe_num_thld)
        {
            return;
        }

        dartion_time = copy_keyframe_pose6d->back().time - copy_keyframe_pose6d->front().time;

        // 1.在历史关键帧中查找与当前关键帧距离最近的关键帧
        if (is_vaild_loop_time_period(dartion_time, loop_vaild_period["odom"]))
        {
            detect_loop_by_distance(keyframe_scan);
        }

        // 2.scan context
        if (is_vaild_loop_time_period(dartion_time, loop_vaild_period["scancontext"]))
        {
            detect_loop_by_scancontext(keyframe_scan);
        }
    }

    void get_loop_constraint(LoopConstraint &loop_constr)
    {
        loop_mtx.lock();
        loop_constr = loop_constraint;
        loop_constraint.clear();
        loop_mtx.unlock();
    }

    bool is_vaild_loop_time_period(const double &time, const std::vector<double> &vaild_period)
    {
        if (vaild_period.empty())
            return true;
        if (vaild_period.size() % 2 != 0)
        {
            LOG_ERROR("time_period size must be double!");
            return true;
        }

        for (auto i = 0; i < vaild_period.size(); i = i + 2)
        {
            if (vaild_period[i] > vaild_period[i + 1])
            {
                LOG_ERROR("time_period must before early than after!");
                continue;
            }
            if (time >= vaild_period[i] && time <= vaild_period[i + 1])
                return true;
        }

        return false;
    }

public:
    std::condition_variable &cv;
    std::atomic<bool> &isABlocked;

    std::unordered_map<std::string, std::vector<double>> loop_vaild_period;
    std::mutex loop_mtx;
    int loop_keyframe_num_thld = 50;
    float loop_closure_search_radius = 10;
    int loop_closure_keyframe_interval = 30;
    int keyframe_search_num = 20;
    bool loop_closure_fitness_use_adaptability = false;
    float loop_closure_fitness_score_thld_min = 0.05;
    float loop_closure_fitness_score_thld_max = 0.05;
    float icp_downsamp_size = 0.1;

    pcl::PointCloud<PointXYZIRPYT>::Ptr copy_keyframe_pose6d;
    pcl::KdTreeFLANN<PointXYZIRPYT>::Ptr kdtree_history_keyframe_pose;
    pcl::KdTreeFLANN<PointType>::Ptr kdtree_submap;

    unordered_map<int, int> loop_constraint_records; // <new, old>, keyframe index that has added loop constraint
    LoopConstraint loop_constraint;
    std::shared_ptr<ScanContext::SCManager> sc_manager; // scan context

    // for visualize
    double last_loop_time = 0;
    double dartion_time;
    PointCloudType::Ptr unused_result;
    PointCloudType::Ptr prevKeyframeCloud;
    ManuallyCorrectLoopClosure mclc;
};
