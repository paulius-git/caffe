#include "caffe/util/stats.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace caffe {

template <typename Dtype>
bool computeBlobStats(const shared_ptr<Blob<Dtype>>& blob,
					            Dtype& min_data,     Dtype& max_data,
                      Dtype& min_diff,     Dtype& max_diff,
                      Dtype& min_abs_data, Dtype& min_abs_diff,
                      double& mean_data,   double& mean_diff,
                      double& sum_sq_data, double& sum_sq_diff) {
  const Dtype* data = blob->cpu_data();
  const Dtype* diff = blob->cpu_diff();

  min_data = min_diff = std::numeric_limits<Dtype>::infinity();
  max_data = max_diff = -std::numeric_limits<Dtype>::infinity();

  double sum_data = 0;
  double sum_grad = 0;

  for(int i = 0; i < blob->count(); i++) {
    const Dtype value = data[i];
    const Dtype grad  = diff[i];

    min_data = std::min(min_data, value);
    max_data = std::max(max_data, value);
    min_diff = std::min(min_diff, grad);
    max_diff = std::max(max_diff, grad);

    min_abs_data = std::min(min_abs_data, std::abs(value));
    min_abs_diff = std::min(min_abs_diff, std::abs(grad));

    sum_data += value;
    sum_sq_data += value * value;
    sum_grad += grad;
    sum_sq_diff += grad * grad;
  }

  mean_data = sum_data / double(blob->count());
  mean_diff = sum_grad / double(blob->count());

	return true;
}

template bool computeBlobStats<float>(
    const shared_ptr<Blob<float>>&, 
    float&, float&, float&, float&, float&, float&,
    double&, double&, double&, double&);
template bool computeBlobStats<double>(
    const shared_ptr<Blob<double>>&,
    double&, double&, double&, double&, double&, double&,
    double&, double&, double&, double&);

} // namespace caffe