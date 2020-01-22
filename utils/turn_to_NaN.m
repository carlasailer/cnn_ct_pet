function array_nan = turn_to_NaN(array)
array_nan = array(:,:);
    for patient=1:202
        if (patient == 71) || (patient == 162) || (patient == 180)
            current_slice = array{1,patient};
            current_slice_nan = double(array_nan{1,patient});
            current_slice_nan(current_slice(:,:) == 0) = nan;
            array_nan{1,patient} = current_slice_nan;
        else 
            n_slices = size(array{1,patient},1);
            for slice_nr=1:n_slices
                current_slice = squeeze(array{1,patient}(slice_nr,:,:));
                current_slice_nan = current_slice(:,:);
                current_slice_nan(current_slice(:,:) == 0) = NaN;
                array_nan{1,patient}(slice_nr,:,:) = current_slice_nan;
            end
        end 
        
    end

end 