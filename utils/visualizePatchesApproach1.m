data = h5read('dataset4.h5', '/X_train');

for patch_nr = 1:284495
    if sum(data(:,:,patch_nr)) ~= 0
        patch1 = double(data(:,:,patch_nr));
        mini = min(min(patch1));
        maxi = max(max(patch1));
        f1 = figure;
        patch1_nan = double(data(:,:,patch_nr));
        for x=1:8
            for y=1:8
                if patch1(x,y) == 0
                    patch1_nan(x,y) = 2000;
                end
            end
        end
        imagesc(patch1_nan)
        colorbar;
        figure;
        imagesc(patch1);
        colorbar;
        break;
    end
end
