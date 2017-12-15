#ifndef CAFFE_ROI_CONV_LAYER_HPP_
#define CAFFE_ROI_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class ROIConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  explicit ROIConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ROIConvolution"; }

  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 2; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

  //void rois_convert_coordinate(Dtype* rois, int num_rois, Dtype spatial_scale);
  // Convert a vector of integer to string
  inline string vec_to_string(const vector<int>& my_vector){
    stringstream result;
    copy(my_vector.begin(), my_vector.end(), std::ostream_iterator<int>(result, " "));
    return result.str();
  }

  Dtype spatial_scale_; // scale
  Dtype* rois; // pointer to the ROIs
  int num_rois; // number of ROIs
  Blob<Dtype> output_buffer_; // for storing the diff of top blob, which needs to be masked by ROI during backprop
  Blob<Dtype> mask_buffer_; // for storing intermediate mask of ROI
};

}  // namespace caffe

#endif  // CAFFE_ROI_CONV_LAYER_HPP_
