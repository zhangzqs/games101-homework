#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0],
        0, 1, 0, -eye_pos[1],
        0, 0, 1, -eye_pos[2],
        0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_rotation_matrix(Eigen::Vector3f axis, float angle)
{
    // 归一化轴向量
    axis.normalize();
    // 计算旋转角度的弧度值
    float rad = angle * MY_PI / 180.0f;

    // n的叉乘矩阵
    Eigen::Matrix3f n(3, 3);
    n << 0, -axis.z(), axis.y(),
        axis.z(), 0, -axis.x(),
        -axis.y(), axis.x(), 0;
    // n的外积矩阵
    Eigen::Matrix3f n_outer = axis * axis.transpose();

    // 计算三部分
    Eigen::Matrix3f component1, component2, component3;
    component1 = Eigen::Matrix3f::Identity() * cos(rad);
    component2 = n * sin(rad);
    component3 = n_outer * (1 - cos(rad));

    // 组合成旋转矩阵
    Eigen::Matrix3f rotation_matrix = component1 + component2 + component3;

    // 将3x3旋转矩阵转换为4x4矩阵
    Eigen::Matrix4f rotation = Eigen::Matrix4f::Identity();
    rotation.block<3, 3>(0, 0) = rotation_matrix;

    return rotation;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // 实现三维空间中绕z轴旋转的变换矩阵
    Eigen::Matrix4f rotate;
    float rad = rotation_angle * MY_PI / 180.0f;
    rotate << cos(rad), -sin(rad), 0, 0,
        sin(rad), cos(rad), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;
    model = rotate * model;
    return model;
}

// 参考：
// https://github.com/DrFlower/GAMES_101_202_Homework/blob/main/Homework_101/Assignment1/Code/main.cpp
Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    assert(zNear > 0 && zFar > 0); // 这里的zNear和zFar是距离，是正数

    // 这里转化成坐标
    zNear = -zNear;
    zFar = -zFar;

    float t = tan(eye_fov / 2 * MY_PI / 180.0f) * abs(zNear);
    float r = t * aspect_ratio;
    float b = -t;
    float l = -r;

    // 正交投影先归一化到一个[-1,1]^3的立方体
    Eigen::Matrix4f M_ortho_scale;
    M_ortho_scale << 2 / (r - l), 0, 0, 0,
        0, 2 / (t - b), 0, 0,
        0, 0, 2 / (zFar - zNear), 0,
        0, 0, 0, 1;

    // 再平移到原点
    Eigen::Matrix4f M_ortho_translate;
    M_ortho_translate << 1, 0, 0, -(r + l) / 2,
        0, 1, 0, -(t + b) / 2,
        0, 0, 1, -(zNear + zFar) / 2,
        0, 0, 0, 1;

    // 透视转正交投影矩阵
    Eigen::Matrix4f M_persp2ortho;
    M_persp2ortho << zNear, 0, 0, 0,
        0, zNear, 0, 0,
        0, 0, zNear + zFar, -zNear * zFar,
        0, 0, 1, 0;
    Eigen::Matrix4f projection = M_ortho_scale * M_ortho_translate * M_persp2ortho;
    return projection;
}

int main(int argc, const char **argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3)
    {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4)
        {
            filename = std::string(argv[3]);
        }
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    // 旋转轴向量
    Eigen::Vector3f rotate_axis = {1, 1, 0};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        // r.set_model(get_model_matrix(angle)); // 围绕z轴旋转的变换矩阵
        r.set_model(get_rotation_matrix(rotate_axis, angle)); // 绕任意轴旋转的变换矩阵
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a')
        {
            std::cout << "angle: " << angle << '\n';
            angle += 10;
        }
        else if (key == 'd')
        {
            std::cout << "angle: " << angle << '\n';
            angle -= 10;
        }
    }

    return 0;
}
