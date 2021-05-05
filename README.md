# STRRN
Source codes for the paper "Dual-stream Multi-path Recursive Residual Network for JPEG Image Compression Artifacts Reduction"

# Implementation
This work is implemented in Caffe.

GLOG_logtostderr=0 GLOG_log_dir='/home/caffe-sl-master/examples/STRRN/train/model' ./build/tools/caffe train --solver=/home/caffe-sl-master/examples/STRRN/train/code/STRRN_L20_solver.prototxt --gpu 0

# Citation
If you find our work useful in your research, please cite:

@article{Jin2021STRRN,
  title={Dual-stream Multi-path Recursive Residual Network for JPEG Image Compression Artifacts Reduction},
  author={Zhi Jin, Muhammad Zafar Iqbal, Wenbin Zou, Xia Li, Eckehard Steinbach},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  year={2021},
  volume={31},
  number={2},
  pages={467-479},
  doi={10.1109/TCSVT.2020.2982174}}
}
