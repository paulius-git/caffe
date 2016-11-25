#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/maxpercentdiff_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class MaxPercentDiffLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  MaxPercentDiffLayerTest() {
    // Create the blobs for testing.
    std::vector<int> shape({2, 3, 4, 5});
    blob_bottom_0_.Reshape(shape),
    blob_bottom_1_.Reshape(shape);
    blob_bottom_vec_.push_back(&blob_bottom_0_);
    blob_bottom_vec_.push_back(&blob_bottom_1_);
    blob_top_vec_.push_back(&blob_top_);

    // Fill the values.
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_bottom_0_);
    filler.Fill(&blob_bottom_1_);
  }
  virtual ~MaxPercentDiffLayerTest() {
  }

  Blob<Dtype> blob_bottom_0_;
  Blob<Dtype> blob_bottom_1_;
  Blob<Dtype> blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MaxPercentDiffLayerTest, TestDtypes);

TYPED_TEST(MaxPercentDiffLayerTest, TestSetUp) {
  LayerParameter layer_param;
  MaxPercentDiffLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  CHECK_EQ(this->blob_top_vec_[0]->shape(0), 1);  
}

TYPED_TEST(MaxPercentDiffLayerTest, TestForward) {
  LayerParameter layer_param;
  shared_ptr<MaxPercentDiffLayer<TypeParam>> layer(
      new MaxPercentDiffLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const TypeParam computed_value = this->blob_top_vec_[0]->cpu_data()[0];

  // Compute the reference value.
  const TypeParam* data0 = this->blob_bottom_vec_[0]->cpu_data();
  const TypeParam* data1 = this->blob_bottom_vec_[1]->cpu_data();
  const int n = this->blob_bottom_vec_[0]->count();
  TypeParam* temp = new TypeParam[n];
  CHECK(temp) << "Failed to allocate a temporary memory buffer.";

  caffe_sub(n, data0, data1, temp);
  caffe_abs(n, temp, temp);
  TypeParam reference_value = TypeParam(0);
  for (int i = 0; i < n; ++i) {
    reference_value = std::max(reference_value, temp[i] / data1[i]);
  }
  EXPECT_FLOAT_EQ(reference_value, computed_value);
}

}  // namespace caffe
