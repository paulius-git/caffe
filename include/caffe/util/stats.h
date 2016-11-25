#pragma once

#include <boost/shared_ptr.hpp>

#include "caffe/blob.hpp"

namespace caffe {

template <typename Dtype>
bool computeBlobStats(const shared_ptr<Blob<Dtype>>& blob,
                      Dtype& min_data,     Dtype& max_data,
                      Dtype& min_diff,     Dtype& max_diff,
                      Dtype& min_abs_data, Dtype& min_abs_diff,
                      double& mean_data,   double& mean_diff,
                      double& sum_sq_data, double& sum_sq_diff);

// template <typename Dtype>
// class Stats {
// public:
// 	Stats();
// 	void add(const Dtype& value);
// 	Dtype getMin();
// 	Dtype getMax();
// 	Dtype getMean();
// protected:
// 	Dtype min_val;
// };

// template <typename Type>
// class Histogram {
// public:
// 	explicit Histogram(const std::vector<Type>&);
// 	void add(const Type& value);
// 	const std::vector<int>& getBins() const;
// protected:
// 	std::vector<Type> markers_;
// 	std::vector<int> bins_;
// };

} // namespace caffe