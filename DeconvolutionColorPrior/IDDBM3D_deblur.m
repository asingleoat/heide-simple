function  [y_hat] = IDDBM3D_deblur(I, k, sigma, I_orig)

% ------------------------------------------------------------------------------------------
%
%     Demo software for BM3D-frame based image deblurring
%               Public release ver. 0.8 (beta) (June 03, 2011)
%
% ------------------------------------------------------------------------------------------
%
%  This function implements the IDDBM3D image deblurring algorithm proposed in:
%
%  [1] A.Danielyan, V. Katkovnik, and K. Egiazarian, "BM3D frames and 
%   variational image deblurring," submitted to IEEE TIP, May 2011 
%
% ------------------------------------------------------------------------------------------
%
% authors:               Aram Danielyan
%                        Vladimir Katkovnik
%
% web page:              http://www.cs.tut.fi/~foi/GCF-BM3D/
%
% contact:               firstname.lastname@tut.fi
%
% ------------------------------------------------------------------------------------------
% Copyright (c) 2011 Tampere University of Technology.
% All rights reserved.
% This work should be used for nonprofit purposes only.
% ------------------------------------------------------------------------------------------
%
% Disclaimer
% ----------
%
% Any unauthorized use of these routines for industrial or profit-oriented activities is
% expressively prohibited. By downloading and/or using any of these files, you implicitly
% agree to all the terms of the TUT limited license (included in the file Legal_Notice.txt).
% ------------------------------------------------------------------------------------------


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  FUNCTION INTERFACE:
%
%  [psnr, y_hat] = Demo_IDDBM3D(experiment_number, test_image_name)
%  
%  INPUT:
%   1) experiment_number: 1 -> PSF 1, sigma^2 = 2
%                         2 -> PSF 1, sigma^2 = 8
%                         3 -> PSF 2, sigma^2 = 0.308
%                         4 -> PSF 3, sigma^2 = 49
%                         5 -> PSF 4, sigma^2 = 4
%                         6 -> PSF 5, sigma^2 = 64
%                         7-13 -> experiments 7-13 are not described in [1].
%                         see this file for the blur and noise parameters.
%   2) test_image_name:   a valid filename of a grayscale test image
%
%  OUTPUT:
%   1) isnr           the output improvement in SNR, dB
%   2) y_hat:         the restored image
%
%  ! The function can work without any of the input arguments, 
%   in which case, the internal default ones are used !
%   
%   To run this demo functions within the BM3D package should be accessible to Matlab 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Add the according pathes
addpath('./BM3D');
addpath('./BM3D/IDDBM3D');

if 1 % 
    initType = 'bm3ddeb'; %use output of the BM3DDEB to initialize the algorithm
else
	initType = 'zeros'; %use zero image to initialize the algorithm
end

matchType = 'bm3ddeb'; %build groups using output of the BM3DDEB algorithm
numIt = 100;

%% ------- Generating bservation ---------------------------------------------
disp('--- Generating observation ----');
y=I;
[yN,xN]=size(y);
z = y;

bsnr=10*log10(norm(z(:)-mean(z(:)),2)^2 /sigma^2/yN/xN);
psnr_z =PSNR(I_orig,z,1,0);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Observation BSNR: %4.2f, PSNR: %4.2f\n', bsnr, psnr_z);

%% ----- Computing initial estimate ---------------------
disp('--- Computing initial estimate  ----');

y_hat_init=zeros(size(z));
match_im = z;
psnr_init = PSNR(I_orig, y_hat_init,1,0);

fprintf('Initialization method: %s\n', initType);
fprintf('Initial estimate ISNR: %4.2f, PSNR: %4.2f\n', psnr_init-psnr_z, psnr_init);

%% ------- Core algorithm ---------------------
%------ Description of the parameters of the IDDBM3D function ----------
%y - true image (use [] if true image is unavaliable)
%z - observed
%h - blurring PSF
%y_hat_init - initial estimate y_0
%match_im - image used to constuct groups and calculate weights g_r
%sigma - standard deviation of the noise
%threshType = 'h'; %use 's' for soft thresholding
%numIt - number of iterations
%gamma - regularization parameter see [1]
%tau - regularization parameter see [1] (thresholding level)
%xi - regularization parameter see [1], it is always set to 1 in this implementation
%showFigure - set to True to display figure with current estimate
%--------------------------------------------------------------------

threshType = 'h';
showFigure = true;

gamma = 0.0005;
tau   = 3/255*2.7;
xi    = 1;

disp('-------- Start ----------');
fprintf('Number of iterations to perform: %d\n', numIt);
fprintf('Thresholding type: %s\n', threshType);

y_hat = IDDBM3D(I_orig, k, z, y_hat_init, match_im, sigma, threshType, numIt, gamma, tau, xi, showFigure);
psnr = PSNR(I_orig,y_hat,1,0);
isnr = psnr-psnr_z;

disp('-------- Results --------');
fprintf('Final estimate ISNR: %4.2f, PSNR: %4.2f\n', isnr, psnr);
return;

end

function PSNRdb = PSNR(x, y, maxval, borders)
    if ~exist('borders', 'var'), borders = 0; end
    if ~exist('maxval', 'var'), maxval = 255; end
    
    xx=borders+1:size(x,1)-borders;
    yy=borders+1:size(x,2)-borders;
            
    PSNRdb = zeros(1,size(x,3));
    for fr=1:size(x,3) 
        err = x(xx,yy,fr) - y(xx,yy,fr);
        PSNRdb(fr) = 10 * log10((maxval^2)/mean2(err.^2));    
    end
end