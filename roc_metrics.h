    
 /*
 * Copyright 2017 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef CLIF_EXAMPLES_ROC_METRICS_ROC_METRICS_H_
#define CLIF_EXAMPLES_ROC_METRICS_ROC_METRICS_H_

#include <functional>
#include <boost/sort/sort.hpp> // block_indirect_sort/block_indirect_sort.hpp>
#include "/usr/local/lib/python3.7/dist-packages/numpy/core/include/numpy/ndarrayobject.h"

namespace clif_example {
namespace roc_metrics {

template <typename T, typename Compare>
void ParallelSort(absl::Span<T> elements, const Compare& comp) {
  // As of 2020-May, boost's block_indirect_sort library is not allowed in
  // Google3.  If using this library in a non-third_party project, this will
  // need to be replaced.
   boost::sort::block_indirect_sort(elements.begin(), elements.end(), comp);
}

template <typename T, typename Compare>
void ParallelSortSlow(absl::Span<T> elements, const Compare& comp) {
  // Used for testing in environments where boost is not available.
  std::sort(elements.begin(), elements.end(), comp);
}

struct PredictElem {
  float score;
  int target;
};

struct RocData {
  std::vector<int> tps;
  std::vector<int> fps;
};

class RocMetrics {
 public:
  // Create an RocMetrics object with predictions py_scores and targets
  // py_objects are numpy array objects.
  explicit RocMetrics(PyObject* py_scores, PyObject* py_targets) {
    PyArrayObject* p_scores = reinterpret_cast<PyArrayObject*>(py_scores);
    PyArrayObject* p_targets = reinterpret_cast<PyArrayObject*>(py_targets);
    int p_data_len = PyArray_SHAPE(p_scores)[0];
    int t_data_len = PyArray_SHAPE(p_targets)[0];
    CHECK(p_data_len > 1) << "Scores array must be of length greater than 1.";
    CHECK(t_data_len > 1) << "Targets array must be of length greater than 1.";
    CHECK(p_data_len == t_data_len)
      << "Scores and targets must be of same length.";
    LOG(INFO) << "PKKK::: in rocmetrics";
    full_data_.reserve(p_data_len);
    auto scores_ptr = static_cast<const float*>(PyArray_GETPTR1(p_scores, 0));
    auto tgts_ptr = static_cast<const int*>(PyArray_GETPTR1(p_targets, 0));
    for (int i = 0; i < p_data_len; ++i) {
      if (tgts_ptr[i] >= 0) {
        full_data_.push_back({scores_ptr[i], tgts_ptr[i]});
      }
    }
  }
  // RocMetrics is designed to be used in a roughly singleton fashion.
  RocMetrics(const RocMetrics& other) = delete;
  RocMetrics& operator=(const RocMetrics& other) = delete;

  // Integrates the y vector along the x vector using the trapezoidal rule.
  // Performance of this function can be improved via parallelization.
  double Trapz(const std::vector<float>& y, const std::vector<float>& x) {
    // Performance can be improved via parallelization.
    double ret = 0.0;
    float x_prev = x[0];
    float y_prev = y[0];
    auto trap_area = [](float x0, float y0, float x1, float y1) {
      float retval = 0.5 * (x1 - x0) * (y0 + y1);
      return double{retval};
    };

    for (int i = 1; i < y.size(); ++i) {
      if (x_prev == 1.0) {
        // Early stop criteria.
        break;
      }
      if (x[i] != x_prev) {
         ret += trap_area(x_prev, y_prev, x[i], y[i]);
      }
          }
      x_prev = x[i];
      y_prev = y[i];
    }
    return ret;
  }

  //float ComputeRocAuc() {
  //  return 0.0;
  //}
  float ComputeRocAuc() {
     ParallelSort(absl::MakeSpan(full_data_),
               [](const PredictElem& t1, const PredictElem& t2) {
                 return t1.score > t2.score;
               });

     // Generate TPR and FPR.
     const auto [tps, fps] = BinaryRoc();
     std::vector<float> tpr(tps.size());
     std::vector<float> fpr(fps.size());
     const float tp_count = static_cast<float>(tps.back());
     const float fp_count = static_cast<float>(fps.back());
     for (int i = 0; i < tps.size(); ++i) {
        tpr[i] = static_cast<float>(tps[i]) / tp_count;
        fpr[i] = static_cast<float>(fps[i]) / fp_count;
      }
      // Trapezoidal integration to compute the AUC.
      double auc = Trapz(tpr, fpr);

      return static_cast<float>(auc);
  }
  // Computes the raw ROC vectors: TPS and FPS. These can be normalized to TPR
  // and FPR via element-wise division with the final value, vec.back().
  // The algorithm is a combination of a cumulative sum on the targets, a
  // filtering operation, and an averaging of duplicated score contributions.
  RocData BinaryRoc() const {
           // TPR and FPR should begin at point (0, 0).
    std::vector<int> tps = {0};
    std::vector<int> fps = {0};
   float prev_score = full_data_[0].score;
    int accum = 0, thresh_idx = 0;

    for (const PredictElem& d : full_data_) {
      float cur_score = d.score;
        if (cur_score != prev_score) {
          tps.push_back(accum);
          fps.push_back(thresh_idx - accum);
        }
        prev_score = cur_score;
        accum += d.target;
        thresh_idx++;
    }
    // Include full sum, for normalization to 1.0.
    tps.push_back(accum);
    fps.push_back(thresh_idx - accum);
    return {std::move(tps), std::move(fps)};
 }

 private:
  // Container for the full set of predictions and targets.
  std::vector<PredictElem> full_data_;
};

}  // namespace roc_metrics 
}  // namespace clif_example

#endif  // CLIF_EXAMPLES_ROC_METRICS_ROC_METRICS_H



    
