//
// Created by hrh on 2020/4/27.
//

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <glog/logging.h>
#include <cJSON.h>
#include <sys/stat.h>

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <opencv2/imgproc.hpp>

#include "SampleDetectorImpl.hpp"

static void DeallocateBuffer(void *data, size_t) {
    std::free(data);
}

/**
 * Read TensorFlow pb file
 * @param file pb file path
 * @return Buffer
 */
static TF_Buffer *ReadBufferFromFile(const char *file) {
    std::ifstream f(file, std::ios::binary);
    if (f.fail() || !f.is_open()) {
        return nullptr;
    }

    if (f.seekg(0, std::ios::end).fail()) {
        f.close();
        return nullptr;
    }
    auto fsize = f.tellg();
    if (f.seekg(0, std::ios::beg).fail()) {
        f.close();
        return nullptr;
    }

    if (fsize <= 0) {
        return nullptr;
    }

    auto data = static_cast<char *>(std::malloc(fsize));
    if (f.read(data, fsize).fail()) {
        free(data);
        f.close();
        return nullptr;
    }

    auto buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = DeallocateBuffer;

    return buf;
}

STATUS SampleDetectorImpl::init() {
    // --------------------------- Replace to use your onw model ------------------------
    const char *modelPath = "/usr/local/ev_sdk/model/ssd_inception_v2.pb";

    // ------------------------------ Add object name -----------------------------------
    // Replace your target label here
    mIDNameMap.insert(std::make_pair<int, std::string>(1, "pedestrian"));

    struct stat st;
    if (stat(modelPath, &st) != 0) {
        LOG(ERROR) << modelPath << " not found!";
        return ERROR_INIT;
    }

    // -------------------------------- Initialize model --------------------------------
    LOG(INFO) << "Loading model...";
    auto buffer = ReadBufferFromFile(modelPath);
    if (buffer == nullptr) {
        LOG(ERROR) << "Can't read buffer from file " << modelPath;
        return ERROR_INIT;
    }

    // Init graph
    mNetwork = TF_NewGraph();
    auto status = TF_NewStatus();
    auto opt = TF_NewImportGraphDefOptions();

    TF_GraphImportGraphDef(mNetwork, buffer, opt, status);
    TF_DeleteImportGraphDefOptions(opt);
    TF_DeleteBuffer(buffer);
    if (TF_GetCode(status) != TF_OK) {
        LOG(ERROR) << "Can't import GraphDef!";
        return ERROR_INIT;
    }
    TF_DeleteStatus(status);
    LOG(INFO) << "Load graph success";

    // Create session
    auto sess_opt = TF_NewSessionOptions();
    status = TF_NewStatus();
    mSession = TF_NewSession(mNetwork, sess_opt, status);
    if (TF_GetCode(status) != TF_OK) {
        LOG(ERROR) << "Can't create session! status code:" << TF_GetCode(status) << ", msg:" << TF_Message(status);
        return ERROR_INIT;
    }
    TF_DeleteSessionOptions(sess_opt);
    TF_DeleteStatus(status);
    LOG(INFO) << "Create session success";

    return SUCCESS_INIT;
}

void SampleDetectorImpl::unInit() {
    auto status = TF_NewStatus();
    TF_CloseSession(mSession, status);
    if (TF_GetCode(status) != TF_OK) {
        LOG(ERROR) << "Can't close session! status code:" << TF_GetCode(status) << ", msg:" << TF_Message(status);
    }
    if (mNetwork != nullptr) {
        TF_DeleteGraph(mNetwork);
        mNetwork = nullptr;
    }
}

STATUS SampleDetectorImpl::processImage(const cv::Mat &cv_image, std::vector<Object> &result) {
    if (cv_image.empty()) {
        LOG(ERROR) << "Invalid input!";
        return ERROR_INVALID_INPUT;
    }

    // -------------------------------- Prepare input -----------------------------------
    int image_width = cv_image.cols;
    int image_height = cv_image.rows;
    cv::Mat input_image;
    cv::resize(cv_image, input_image, cv::Size(mInputWidth, mInputHeight));
    if (input_image.channels() == 1) {
        cv::cvtColor(input_image, input_image, cv::COLOR_GRAY2BGR);
    } else if (input_image.channels() == 4) {
        cv::cvtColor(input_image, input_image, cv::COLOR_RGBA2BGR);
    }

    std::int64_t input_dims[4] = {1, mInputWidth, mInputHeight, 3};
    auto input_op = TF_Output{TF_GraphOperationByName(mNetwork, "image_tensor"), 0};
    if (input_op.oper == nullptr) {
        LOG(ERROR) << "Can't find input_op `image_tensor`";
        return ERROR_PROCESS;
    }

    // Create tensor and copy image data to tensor
    std::size_t len = mInputWidth * mInputHeight * 3;
    auto input_tensor = TF_AllocateTensor(TF_UINT8, input_dims, 4, len);
    if (input_tensor == nullptr) {
        LOG(ERROR) << "Can't create input tensor!";
        return ERROR_PROCESS;
    }
    auto tensor_data = TF_TensorData(input_tensor);
    if (tensor_data == nullptr) {
        LOG(ERROR) << "Can't create tensor data!";
        return ERROR_PROCESS;
    }
    len = std::min(len, TF_TensorByteSize(input_tensor));
    if (len != 0) {
        std::memcpy(tensor_data, input_image.data, len);
    }

    // -------------------------------- Prepare output ----------------------------------
    TF_Output output_ops[3];
    output_ops[0] = TF_Output{TF_GraphOperationByName(mNetwork, "detection_classes"), 0};
    output_ops[1] = TF_Output{TF_GraphOperationByName(mNetwork, "detection_scores"), 0};
    output_ops[2] = TF_Output{TF_GraphOperationByName(mNetwork, "detection_boxes"), 0};
    for (auto &output : output_ops) {
        if (output.oper == nullptr) {
            LOG(ERROR) << "Invalid output! index=" << output.index;
        }
    }
    TF_Tensor *output_tensors[3];

    // -------------------------------- Inference ---------------------------------------
    auto status = TF_NewStatus();
    TF_SessionRun(mSession, nullptr, &input_op, &input_tensor, 1,
                  &output_ops[0], &output_tensors[0], 3,
                  nullptr, 0, nullptr, status);
    if (TF_GetCode(status) != TF_OK) {
        LOG(ERROR) << "Error run session, status:" << TF_GetCode(status) << ", msg:" << TF_Message(status);
        TF_CloseSession(mSession, status);
        TF_DeleteStatus(status);
        return ERROR_PROCESS;
    }
    TF_DeleteStatus(status);

    // -------------------------------- Parse output ------------------------------------
    result.clear();
    auto class_tensor_data = static_cast<float *>(TF_TensorData(output_tensors[0]));
    auto prob_tensor_data = static_cast<float *>(TF_TensorData(output_tensors[1]));
    auto box_tensor_data = static_cast<float *>(TF_TensorData(output_tensors[2]));
    auto dims = TF_NumDims(output_tensors[2]);
    auto num_of_predictions = TF_Dim(output_tensors[2], 1); // (batch size=1, num of predictions, 4)
    for (int i = 0; i < num_of_predictions; ++i) {
        int label = int(class_tensor_data[i]);
        float confidence = prob_tensor_data[i];
        int ymin = int(box_tensor_data[i * 4] * image_height);
        int xmin = int(box_tensor_data[i * 4 + 1] * image_width);
        int ymax = int(box_tensor_data[i * 4 + 2] * image_height);
        int xmax = int(box_tensor_data[i * 4 + 3] * image_width);
        if (confidence > mThresh) {
            LOG(INFO) << "Found object label=" << label << ", confidence=" << confidence
                << ", (" << xmin << ", " << ymin << ", " << xmax << ", " << ymax << ")";
            result.emplace_back(SampleDetector::Object({
                confidence,
                mIDNameMap[label],
                cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin)
            }));
        }
    }

    // -------------------------------- Release resource --------------------------------
    for (auto &t : output_tensors) {
        TF_DeleteTensor(t);
    }
    TF_DeleteTensor(input_tensor);


    return SUCCESS_PROCESS;
}