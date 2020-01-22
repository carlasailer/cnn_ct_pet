data_CT = h5read('patches_training.h5', '/CT');
data_PET = h5read('patches_training.h5','/PET');

for patch_nr = 363:284495
    if sum(data_CT(:,:,:,patch_nr)) ~= 0
        patch_CT = data_CT(:,:,:,patch_nr);
        patch_PET = data_PET(:,:,:,patch_nr);
        for slice_nr = 1
            f1 = figure;
            slice_CT = patch_CT(:,:,slice_nr);
            for x=1:13
                for y=1:13
                    if slice_CT(x,y) == 0
                        slice_CT(x,y) = 2000;
                    end
                end
            end
            slice_PET = patch_PET(:,:,slice_nr);
            imagesc(slice_CT)
            colorbar;
            f2 = figure;
            imagesc(slice_PET)
            colorbar;
        end
    end
    close all;
end
