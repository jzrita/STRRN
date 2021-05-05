clear;clc;
caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 1;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

model_dir = 'fill your folder path here';
folder_test_data = 'fill your test dataset folder path here';
test_dataset=1;


JPEG_Quality=10;
lamda_lq=0.04;
lamda_gt=0.02;

if test_dataset==1
    folder='LIVE1_gray/';
    filepaths = dir(fullfile(folder_test_data,folder,'*.bmp'));
% elseif test_dataset==2
%     folder='Set14/';
%     filepaths = dir(fullfile(folder_test_data,folder,'*.bmp'));
% elseif test_dataset==3
%     folder='Set5/';
%     filepaths = dir(fullfile(folder_test_data,folder,'*.bmp'));
% elseif test_dataset==4
%     folder='BSDS500_100/';
%     filepaths = dir(fullfile(folder_test_data,folder,'*.jpg'));
% elseif test_dataset==5
%     folder='classic5_gray/';
%     filepaths = dir(fullfile(folder_test_data,folder,'*.bmp'));
% elseif test_dataset==9
%     folder='Urban100_x2_HR/';
%     filepaths = dir(fullfile(folder_test_data,folder,'*.png'));
end

count11=1;
fid1=fopen('test_result_psnr.txt','wt');
fid2=fopen('test_result_ssim.txt','wt');


for idx=30800:30800:1232000
    net_model = [model_dir,'STRRN_L20_3B3U_net.prototxt'];
    net_weights = sprintf('%s/model/STRRN_q10.caffemodel',model_dir,idx);
    net = caffe.Net(net_model, net_weights, 'test');
    for k= 1:length(filepaths)
        
        im_gt = imread(fullfile(folder_test_data,folder,filepaths(k).name));
        
        if size(im_gt,3)>1
            im_gt_ycbcr = rgb2ycbcr(im_gt);
            imwrite(im_gt_ycbcr(:,:,1),'test1.jpg','jpg','Quality',JPEG_Quality);
            im_gt_y=im2double(im_gt_ycbcr(:,:,1));
            im_lq_y=im2double(imread('test1.jpg'));
        else
            imwrite(im_gt,'test1.jpg','jpg','Quality',JPEG_Quality);
            im_gt_y=im2double(im_gt);
            im_lq_y=im2double(imread('test1.jpg'));
        end
        delete('test1.jpg');
        sz = size(im_gt_y);
        if sz(1)*sz(2)> 512*384
            image_parts=4;%cut the input image into image_parts^2 parts
            im_gt_y = modcrop(im_gt_y, image_parts);
            im_lq_y = modcrop(im_lq_y, image_parts);
            sz = size(im_gt_y);
        else
            image_parts=4;
            im_gt_y = modcrop(im_gt_y, image_parts);
            im_lq_y = modcrop(im_lq_y, image_parts);
            sz = size(im_gt_y);
        end
        
        
        %% structure_texture separation
        [im_lq_text, im_lq_struct] =  TV_L2_Decomp(im_lq_y, lamda_lq) ; %ROF_decomp(y , 0.05, 100, 1) ;
        [im_gt_text, ~] =  TV_L2_Decomp(im_gt_y, lamda_gt) ;
        
        count=1;
        stepsize1=sz(1)/image_parts;
        stepsize2=sz(2)/image_parts;
        img_lq_s=cell(1,image_parts^2);
        img_lq_t=cell(1,image_parts^2);
        last=cell(1,image_parts^2);
        final_output=zeros(sz(1),sz(2));
        final_output_s=zeros(sz(1),sz(2));
        final_output_t=zeros(sz(1),sz(2));
        for i=1:(stepsize1):sz(1)-stepsize1+1
            for j=1:(stepsize2):sz(2)-stepsize2+1
                
                img_lq_s{count}=im_lq_struct(i:i+stepsize1-1,j:j+stepsize2-1);
                net.blobs('data_s').reshape([size(img_lq_s{count}) 1 1]);
                net.blobs('data_s').set_data(img_lq_s{count});
                img_lq_t{count}=im_lq_text(i:i+stepsize1-1,j:j+stepsize2-1);
                net.blobs('data_t').reshape([size(img_lq_t{count}) 1 1]);
                net.blobs('data_t').set_data(img_lq_t{count});
                
                %img_lq_s{count}=im_lq_y(i:i+stepsize1-1,j:j+stepsize2-1);
                %net.blobs('data').reshape([size(img_lq_s{count}) 1 1]);
                %net.blobs('data').set_data(img_lq_s{count});
                
                net.forward_prefilled();
                last_s{count} = net.blobs('HR_recovery_s').get_data();
                last_t{count} = net.blobs('HR_recovery_t').get_data();
                last{count} = net.blobs('HR_recovery').get_data();
                
                final_output(i:i+stepsize1-1,j:j+stepsize2-1)=last{count};
                final_output_s(i:i+stepsize1-1,j:j+stepsize2-1)=last_s{count};
                final_output_t(i:i+stepsize1-1,j:j+stepsize2-1)=last_t{count};
                final_output_combine=double(final_output_s+final_output_t);
                count=count+1;
                
            end
        end
        

        output_psnr_final(k)=psnr(im_gt_y,final_output);
        output_ssim_final(k)=ssim(im_gt_y,final_output);
        
    end
    caffe.reset_all();
    fprintf(fid1,'%0.4f\n',mean(output_psnr_final));
    fprintf(fid2,'%0.4f\n',mean(output_ssim_final));
    count11=count11+1;
end
fclose(fid1);
fclose(fid2);
