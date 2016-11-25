#pragma once

#include "caffe/blob.hpp"
#include "caffe/common.hpp"

namespace caffe {

enum ProcessBlobField {kData=0, kDiff};

template <typename Dtype>
void processBlobs(const vector<Blob<Dtype>*>& blobs, 
                  const ProcessBlobField field,
                  const bool flush, const bool fp16, const unsigned int mask);

template <typename Dtype>
void processBlobs(const vector<shared_ptr<Blob<Dtype>>>& blobs,
                  const ProcessBlobField field,
                  const bool flush, const bool fp16, const unsigned int mask);

} // namespace caffe