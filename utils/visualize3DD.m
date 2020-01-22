close all;
patient = 2;
vol_PET = PET{patient};
vol_CT = CT{patient};

sizeDim_PET = size(vol_PET);
sizeDim_CT = size(vol_CT);

figure;
slice(double(vol_PET),sizeDim_PET(2)/2,sizeDim_PET(1)/2,4);
%shading interp;
%shading flat;
shading faceted;
title('PET');
axis on;
colorbar;

figure;
slice(double(vol_CT),sizeDim_CT(2)/2,sizeDim_CT(1)/2,4);
%shading interp;
%shading flat;
shading faceted;
title('Original');
axis on;
title('CT');
colorbar

