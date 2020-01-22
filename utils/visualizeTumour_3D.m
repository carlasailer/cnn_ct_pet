%close all;
clear Cs;
CT = images_segmented_CT(:,:);
PET = images_segmented_PET(:,:);
CT_nan = turn_to_NaN(CT);

patient = 1;
sizeDim_CT = size(CT_nan{patient});
x = uint8(linspace(1, sizeDim_CT(2), sizeDim_CT(2)));
y = uint8(linspace(1, sizeDim_CT(1), sizeDim_CT(1)));
z = uint8(linspace(1, sizeDim_CT(3), sizeDim_CT(3)));
vol_CT = imhistmatchn(mat2gray(CT{patient}), mat2gray(PET{patient}));
vol_PET = PET{patient};

vol = vol_CT;
%vol = vol_PET;

[Y,X,Z] = meshgrid(x,y,z);
figure;
colormap 'parula';
%C = CT_nan{patient};
C = uint8(255* mat2gray(vol));
Cs = C;
C = uint8(255*mat2gray(vol));
hiso = patch(isosurface(Cs,0,C));
hiso.FaceColor = 'interp';%[0,0.8,1];%[1,0.75,0.65];
hcap = patch(isocaps(C,0),...
   'FaceColor','interp',...
   'EdgeColor','none');
colormap default;
isonormals(Cs,hiso);
%view([0,-45]); 
% if size(vol) == size(vol_CT)
%     title('CT')
% else
%     title('PET')    
% end 
angle1 = 35;
angle2 = 35;
view(angle1,angle2);
grid on;
%axis off; 
grid on;
%daspect([1,1,2])
lightangle(angle1,angle2);
lighting gouraud
hcap.AmbientStrength = 0.4;
hcap.SpecularColorReflectance = 0;
hcap.SpecularExponent = 50;

