#ifndef _CAFFE_UTIL_IM2COL_ROI_HPP_
#define _CAFFE_UTIL_IM2COL_ROI_HPP_

namespace caffe {

template <typename Dtype>
void roi_im2col_gpu(const Dtype* data_im, const Dtype* rois, const int
    num_rois, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col);

template <typename Dtype>
void roi_col2im_gpu(const Dtype* data_col, const Dtype* rois, const int num_rois,
    const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im);

template <typename Dtype>
void roi_im2col_cpu(const Dtype* data_im, const Dtype* rois, const int
    num_rois, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col);

template <typename Dtype>
void mask_output_gpu(Dtype* output, Dtype* mask, const Dtype* rois, const int num_rois, 
        const int channel, const int height, const int width); 

}  // namespace caffe
#endif  // CAFFE_UTIL_IM2COL_ROI_HPP_
