#include <vector>

#include "caffe/util/im2col_roi.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
bool is_pt_inside_roi(const int x, const int y, const Dtype* roi, 
        const int kernel_h, const int kernel_w){
	// check if the point (x,y) is inside the ROI, x on the axis of the column, y on axis of row
	return (roi[1]-kernel_w/2 <= x) && (x <= roi[3]+kernel_w/2) && 
        (roi[2]-kernel_h/2 <= y) && (y <= roi[4]+kernel_h/2);
}

template <typename Dtype>
bool any_rois_contain_pt(const int x, const int y, const Dtype* rois, const int num_rois, 
        const int kernel_h, const int kernel_w){
	for (int i = 0; i< num_rois; i++){
		if ( is_pt_inside_roi<Dtype>(x,y,&rois[5*i],kernel_h,kernel_w) )
            return true;
	}
    return false;
}

template <typename Dtype>
void roi_im2col_cpu(const Dtype* data_im, const Dtype* rois, const int num_rois, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  int num_rois_contain = 0;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
	            // copy the pixel val into the output if it's inside the boundary and inside a roi
                // (w_im, h_im) is the (x,y) of the point
                num_rois_contain = any_rois_contain_pt(input_col, input_row, rois, num_rois, kernel_h, kernel_w) ? 1:0;
                *(data_col++) = num_rois_contain * data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}



// Explicit instantiation
template void roi_im2col_cpu<float>(const float* data_im, const float* rois, const int num_rois,
    const int channels, const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_col);
template void roi_im2col_cpu<double>(const double* data_im, const double* rois, const int num_rois,
    const int channels, const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_col);

}  // namespace caffe

