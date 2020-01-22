%visualizes slices%

close all;
patient = 1;

PET = images_segmented_PET(:,:);
CT = images_segmented_CT(:,:);

%CT_nan = turn_to_NaN(CT);

vol_PET = PET{patient};
vol_CT = CT{patient};
vol_CT = imhistmatchn(mat2gray(CT{patient}), mat2gray(PET{patient}));

slice_nr = 1;

figure;
current_slice_PET = vol_PET(:,:,slice_nr);
imagesc(current_slice_PET);
colorbar;
view([90 -90]);
set(gca,'XDir','reverse')
%title('PET');

figure; 
current_slice_CT = vol_CT(:,:,slice_nr);
%current_slice_CT_imhist = imhistmatch(mat2gray(current_slice_CT), mat2gray(current_slice_PET));
imagesc(current_slice_CT);%_imhist);
colorbar;
view([90 -90]);
set(gca,'XDir','reverse');
%title('CT');

figure;
current_slice_CT = images_segmented_CT{patient}(:,:,slice_nr);
imagesc(current_slice_CT);
colorbar;
view([90 -90]);
set(gca,'XDir','reverse');
