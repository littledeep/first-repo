//
// Created by hrh on 2020/4/27.
//

#ifndef JI_SAMPLEDETECTORIMPL_HPP
#define JI_SAMPLEDETECTORIMPL_HPP

#include <tensorflow/c/c_api.h>
#include "SampleDetector.hpp"

/**
 * @brief 模型基于ssd inception v2 coco训练得到行人检测模型
 *
 * 这个示例使用Tensorflow的C接口实现推理
 */
class SampleDetectorImpl: public SampleDetector {
public:
    STATUS init() override;

    void unInit() override;

    STATUS processImage(const cv::Mat &image, std::vector<Object> &detectResults) override;

private:
    TF_Graph *mNetwork{nullptr};
    TF_Session *mSession{nullptr};
    int mInputHeight = 300;
    int mInputWidth = 300;
};


#endif //JI_SAMPLEDETECTORIMPL_HPP
