clear
%folder0='';
datapath='F:\4.Related to work\1.Postdoct\Year1\Project_Compression_Artifacts_Reduction\STCNN\test\Urban100\Urban100_x4_HR_with_data_aug\';
filepaths=dir(fullfile(datapath,'*.png'));
Nimg=length(filepaths);
%% Rotate images
for i=1:Nimg
    filename=filepaths(i).name;
    [imaddress,imname,type]=fileparts(filepaths(i).name);
    image=imread(fullfile(datapath,filename));
    im1=rot90(image,1);
    im2=rot90(image,2);
    im3=rot90(image,3);
    imwrite(im1,[datapath imname, '_rot90' '.png']);
    imwrite(im2,[datapath imname, '_rot180' '.png']);
    imwrite(im3,[datapath imname, '_rot270' '.png']);
end
% mirror images
filepaths=dir(fullfile(datapath,'*.png'));
Nimg=length(filepaths);

for i=1:Nimg
    filename=filepaths(i).name;
    [imaddress,imname,type]=fileparts(filepaths(i).name);
    image=imread(fullfile(datapath,filename));
    im1=fliplr(image);%horizontal flip
    imwrite(im1,[datapath imname, '_h' '.png']);
end