//
// Created by hrh on 2019-09-02.
//

#ifndef JI_SAMPLEDETECTOR_HPP
#define JI_SAMPLEDETECTOR_HPP

#include <string>
#include <opencv2/core/mat.hpp>
#include <map>

#define STATUS int

/**
 * @brief 检测类算法的抽象类，子类必须实现三个抽象接口
 */

class SampleDetector {

public:
    typedef struct {
        float prob;
        std::string name;
        cv::Rect rect;
    } Object;

    /**
     * @brief 初始化模型
     *
     * @return 如果初始化正常，返回INIT_OK，否则返回`ERROR_*`
     */
    virtual STATUS init() = 0;

    /**
     * 反初始化函数
     */
    virtual void unInit() = 0;

    /**
     * @brief 对cv::Mat格式的图片进行分类，并输出预测分数前top排名的目标名称到mProcessResult
     *
     * @param[in] image 输入图片
     * @param[out] detectResults 检测到的结果
     * @return 如果处理正常，则返回PROCESS_OK，否则返回`ERROR_*`
     */
    virtual STATUS processImage(const cv::Mat &image, std::vector<Object> &detectResults) = 0;

    /**
     * @brief 设置阈值
     *
     * @param thresh
     * @return
     */
    bool setThresh(double thresh) {
        mThresh = thresh;
    };


public:
    static const int ERROR_BASE = 0x0100;
    static const int ERROR_INVALID_INPUT = 0x0101;
    static const int ERROR_INIT = 0x0102;
    static const int ERROR_PROCESS = 0x0103;

    static const int SUCCESS_PROCESS = 0x1001;
    static const int SUCCESS_INIT = 0x1002;

protected:
    double mThresh = 0.5;
    std::map<int, std::string> mIDNameMap;  // id与label的映射表
};

#endif //JI_SAMPLEDETECTOR_HPP
