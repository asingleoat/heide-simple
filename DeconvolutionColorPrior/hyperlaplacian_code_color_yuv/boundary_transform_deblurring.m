function I_res = boundary_transform_deblurring(I, trans_mode, boundary_size, taper)
% boundary_transform Changes the boundary of an image for deblurring.
%
%	Author:		Felix Heide

%Check for sanity
if nargin < 3
    error('Too few arguments specified')
end

%Care for optional parameters
if nargin == 3
    taper = false;
end

%Parse parameters
if strncmpi(trans_mode ,'add',3)
    trans_mode = 'add';    
elseif strncmpi(trans_mode ,'rem',3)
    trans_mode = 'rem';
else  
    trans_mode = 'add';
end

% Now invoke selected method
if strcmp(trans_mode, 'add')
    I_res = addpad_boundary(I, boundary_size, taper);
elseif strcmp(trans_mode, 'rem')
    I_res = cutpad(I, boundary_size); 
end

return;

function f = addpad_boundary(f,p,taper)

%Czero boundary
[r,c,~] = size(f);
f = f([ones(1,p),1:r,r*ones(1,p)],[ones(1,p),1:c,c*ones(1,p)],:);

%Optonally taper
if taper
    for ch = 1:size(f,3)
        %Compute max edgetaper size
        half_min_dim = round( min(size(f(:,:,ch))) / 2 );
        if p*2 > half_min_dim
            p = floor(half_min_dim/2);
        end
        f(:,:,ch) = edgetaper(f(:,:,ch), fspecial('gaussian',p*2, p/3));
    end
end
return;

function f = cutpad(f_pad,p)
[r,c,~] = size(f_pad);
r = r - 2*p;
c = c - 2*p;
f = f_pad(p + (1:r),p + (1:c),:);
return;