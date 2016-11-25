#include <vector>

#include "caffe/layers/maxpercentdiff_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MaxPercentDiffLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void MaxPercentDiffLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2) << "AMaxDiff layer requires 2 bottom blobs.";
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "The 2 input blobs must have the same number of elements.";
  top[0]->Reshape({1});
}

template <typename Dtype>
void MaxPercentDiffLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* computed = bottom[0]->cpu_data();
  const Dtype* reference = bottom[1]->cpu_data();
  const int n = bottom[0]->count();

  Dtype max_percent_diff = Dtype(0);
  for (int i = 0; i < n; ++i) {
    max_percent_diff = 
        std::max(max_percent_diff, 
                 std::abs(computed[i] - reference[i]) / reference[i]);
  }

  top[0]->mutable_cpu_data()[0] = max_percent_diff;
}

INSTANTIATE_CLASS(MaxPercentDiffLayer);
REGISTER_LAYER_CLASS(MaxPercentDiff);

}  // namespace caffe
