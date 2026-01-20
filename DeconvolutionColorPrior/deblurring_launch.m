%Synthetic data test for "Primal dual cross-channel deconvolution"

%	Author copyright:		Felix Heide (fheide@cs.ubc.ca)

% Code including comparisons for our paper:
%
% F. Heide, M. Rouf, M. Hullin, B. Labitzke, W. Heidrich, A. Kolb.
% High-Quality Computational Imaging Through Simple Lenses.
% ACM ToG 2013 

%Clear workspace
clear all
close all hidden

%Pathes
addpath('./hyperlaplacian_code')
addpath('./hyperlaplacian_code_color_yuv')

%Load image and convert to grayscale
image_filename = 'images/houses_big.jpg'; 
I = imread(image_filename);
I = imresize(I, 0.15); %Sample down for faster deconvolution
I = double(I);
I = I ./ max(I(:));

%Write to disk
%imwrite( I, 'original_gamma_2_2.png', 'Bitdepth', 16 );

%Apply inverse gamma
I = I .^ (2.0);

%Write to disk
%imwrite( I, 'original_linear.png', 'Bitdepth', 16 );

%Display image information and noisy image
fprintf('Processing %s with size %d x %d \n', image_filename, size(I, 2), size(I ,1))

%Define sharp image
I_sharp = I;

%Define square sized blur kernel
blur_size = 15; %has to be an odd number !
K_blur = cell(0); %Blur for all channels

%Loaded kernel
inc_blur_exp = 1.7;
max_blur_size = round(blur_size * (inc_blur_exp)^(size(I,3) - 1));
max_blur_size = max_blur_size + (1 - mod(max_blur_size,2));
max_radius = floor(max_blur_size / 2);
K_blur_disp = zeros(max_blur_size, max_blur_size, size(I,3));
for ch = 1:size(I,3)
    %K_blur{ch} = imread('kernels/planoconvex_lens_kernel.png');
    K_blur{ch} = imread('kernels/fading.png');
    %K_blur{ch} = imread('kernels/snake.png');
    %Curr blur size
    curr_blur_size = round(blur_size* (inc_blur_exp)^(ch - 1));
    curr_blur_size = curr_blur_size + (1 - mod(curr_blur_size,2));
    
    K_blur{ch} = imresize( K_blur{ch}, [curr_blur_size, curr_blur_size] , 'bicubic');
    K_blur{ch} = img_to_norm_grayscale(K_blur{ch});
    K_blur{ch} = K_blur{ch} ./ sum(K_blur{ch}(:)); %Finally normalize kernel to sum of 1
    
    %Store kernel for display
    curr_blur_radius = floor(curr_blur_size / 2);
    K_blur_disp( 1 + max_radius + (-curr_blur_radius:curr_blur_radius),1 + max_radius + (-curr_blur_radius:curr_blur_radius), ch) = ...
        (K_blur{ch} ./ max(K_blur{ch}(:)));
end

%Set first channel to sharp image
K_blur{1} = imread('kernels/fading.png');
curr_blur_size = blur_size + (1 - mod(blur_size,2));
curr_blur_radius = floor(curr_blur_size / 2);
K_blur{1} = imresize( K_blur{1}, [curr_blur_size, curr_blur_size] , 'bicubic');
K_blur{1}(:) = 0;
K_blur{1} = img_to_norm_grayscale( K_blur{1} );
K_blur{1}(curr_blur_radius + 1, curr_blur_radius + 1) = 1.0;
K_blur{1} = K_blur{1} ./ sum(K_blur{1}(:));
   
K_blur_disp( 1 + max_radius + (-curr_blur_radius:curr_blur_radius), ...
            1 + max_radius + (-curr_blur_radius:curr_blur_radius), 1) = ...
             (K_blur{1} ./ max(K_blur{1}(:)));
    
%Compute blurred images
I_blurred = zeros(size(I));
K_blur_orig = K_blur;
for ch = 1:size(I,3)
    I_blurred(:,:,ch) = imfilter(I_sharp(:,:,ch), K_blur{ch}, 'conv', 'symmetric');

    %Add noise to blurred image
    noise_sd = 0.005;
    I_blurred(:,:,ch) = imnoise(I_blurred(:,:,ch), 'gaussian', 0, noise_sd.^2);

    %And kernel image
    kernel_var = 0.0000001;
    K_blur{ch} = imnoise(K_blur{ch}, 'gaussian', 0, kernel_var / ch^2);
    K_blur{ch} = K_blur{ch} ./ sum(K_blur{ch}(:)); %Normalize
end

%Display input
overview_figure = figure();
I_sharp_disp = I_sharp.^ (1.0/2.0) ;
I_sharp_disp(1:size(K_blur_disp,1), 1:size(K_blur_disp,1),:) = K_blur_disp;
subplot(6,size(I,3)+1,1), imshow(I_sharp_disp ), title(sprintf('Original image (%d x %d)',size(I, 2), size(I ,1)));
for ch = 1:size(I,3)
    K_curr_disp = [K_blur{ch}, K_blur_orig{ch}];
    K_curr_disp = (K_curr_disp ./ max(K_curr_disp(:))) .^ (1.0/2.0);
    subplot(6,size(I,3)+1,1 + ch), imshow( K_curr_disp .^ (1/2.0) ), axis image, title(sprintf('Kernel Ch%d (%d x %d)', ch, size(K_curr_disp, 2)/2, size(K_curr_disp ,1)));
end
I_blurred_disp = I_blurred.^ (1.0/2.0);
subplot(6,size(I,3)+1, size(I,3) + 5 ), imshow( I_blurred_disp ), title(sprintf('Blurred image %d'));

%Display blurred channels
for ch = 1:size(I,3)
    subplot(6,size(I,3)+1, size(I,3) + 1 + ch), imshow(I_blurred_disp(:,:,ch) ), title(sprintf('Blurred Ch%d',ch));
end

%Prepare data
%Extract current patches and blur kernels
channel_patch = cell(0);
for ch = 1:size(I,3)
    channel_patch(ch).Image = I_blurred(:,:,ch);
    channel_patch(ch).K = K_blur{ch};
end

%Proposed cross-channel method:
fprintf('\nComputing cross-channel deconvolution ... \n\n')

%Lambda
%format: #ch, l_residual, l_tv, l_black, l_chross_ch, #detail_layers
lambda_startup = [ [1, 300, 1.0, 0.0, [  0.0, 0.0, 0.0], 1]
                  [2, 750, 0.5, 0.0, [  1.0, 0.0, 0.0], 0];...
                  [3, 750, 0.5, 0.0, [  1.0, 0.0, 0.0], 0]];

tic;
verbose = 'brief';
I_deconv_channels = channel_patch;
I_deconv_channels = pd_joint_deconv(channel_patch, [], ...
                                lambda_startup, ...
                                200, 1e-4, verbose);

%Deblur using a hyperlaplacian prior independently on each channel
% D. Krishnan, R. Fergus: "Fast Image Deconvolution using Hyper-Laplacian
% Priors", Proceedings of NIPS 2009.
fprintf('\nComputing hyperlaplacian naive deconvolution ... \n\n')

I_deconv_channels_hyp = channel_patch;
lambda = 2000;
for ch = 1:size(I,3)
    I_deconv_channels_hyp(ch).Image = fast_deconv(edgetaper(channel_patch(ch).Image, channel_patch(ch).K), channel_patch(ch).K, lambda, 2/3);
end    

%Deblur using bm3d prior
%  [1] A.Danielyan, V. Katkovnik, and K. Egiazarian, "BM3D frames and 
%   variational image deblurring," submitted to IEEE TIP, May 2011 
fprintf('\nComputing bm3d naive deconvolution ... \n\n')

I_deconv_channels_bm3d = channel_patch;
sd_bm3d =0.01;
for ch = 2:size(I,3)
    I_deconv_channels_bm3d(ch).Image = IDDBM3D_deblur(edgetaper(channel_patch(ch).Image, channel_patch(ch).K),...
                                                channel_patch(ch).K, sd_bm3d, I_sharp(:,:,ch) );  
end       

%Deblur using YUV hyperlaplacian prior
% C. J. Schuler, M. Hirsch, S. Harmeling and B. Scholkopf: 
% "Non-stationary Correction of Optical Aberrations", Proceedings of ICCV 2011.
fprintf('\nComputing hyperlaplacian YUV deconvolution ... \n\n')

%Prepare image for hyperlaplacian yuv deconvolution
y_hyp_yuv = zeros(size(channel_patch(1).Image, 1), size(channel_patch(1).Image, 2), length(channel_patch));
k_hyp_yuv = cell(0);
for ch = 1:size(y_hyp_yuv,3)
    y_hyp_yuv(:,:,ch) = channel_patch(ch).Image;
    k_hyp_yuv{ch} = channel_patch(ch).K;
end

%Algorithm parameters from:
% C. J. Schuler, M. Hirsch, S. Harmeling and B. Scholkopf: 
% "Non-stationary Correction of Optical Aberrations", Proceedings of ICCV 2011.
lambda_hyp_yuv = 2e3;
alpha_hyp_yuv = 0.65;
rho_yuv = [0.1, 1.0, 1.0];
w_rgb = [1/4, 1/2, 1/4];
theta = max_blur_size^2 * kernel_var * 1e3;

%Do the minimization
verbose = 'all';
x = y_hyp_yuv;
[x] = fast_deconv_yuv(y_hyp_yuv, k_hyp_yuv, lambda_hyp_yuv, rho_yuv, w_rgb, theta, alpha_hyp_yuv);
 
%Store in cell
I_deconv_channels_yuv = cell(0);
for ch = 1:size(I,3)
    I_deconv_channels_yuv(ch).Image = x(:,:,ch);
    I_deconv_channels_yuv(ch).K = k_hyp_yuv{ch};
end
                            
%Gather result
I_deconv = zeros(size(I));
for ch = 1:length(I_deconv_channels)
    I_deconv(:,:,ch) = I_deconv_channels(ch).Image;
end

I_deconv_hyp = zeros(size(I));
for ch = 1:length(I_deconv_channels)
    I_deconv_hyp(:,:,ch) = I_deconv_channels_hyp(ch).Image;
end

I_deconv_bm3d = zeros(size(I));
for ch = 1:length(I_deconv_channels)
    I_deconv_bm3d(:,:,ch) = I_deconv_channels_bm3d(ch).Image;
end

I_deconv_yuv = zeros(size(I));
for ch = 1:length(I_deconv_channels_yuv)
    I_deconv_yuv(:,:,ch) = I_deconv_channels_yuv(ch).Image;
end

%Compute PSNR:
psnr_pad = round(size(K_blur,1) * 1.5);
I_diff = I_sharp(psnr_pad + 1:end - psnr_pad, psnr_pad + 1:end - psnr_pad,:) - I_deconv(psnr_pad + 1:end - psnr_pad, psnr_pad + 1:end - psnr_pad,:);
MSE = 1/size(I_diff(:),1)*(norm(I_diff(:), 2)^2);
if MSE > eps
    PSNR = 10*log10(1/MSE);
else
    PSNR = Inf;
end

I_diff = I_sharp(psnr_pad + 1:end - psnr_pad, psnr_pad + 1:end - psnr_pad,:) - I_deconv_hyp(psnr_pad + 1:end - psnr_pad, psnr_pad + 1:end - psnr_pad,:);
MSE = 1/size(I_diff(:),1)*(norm(I_diff(:), 2)^2);
if MSE > eps
    PSNR_hyp = 10*log10(1/MSE);
else
    PSNR_hyp = Inf;
end

I_diff = I_sharp(psnr_pad + 1:end - psnr_pad, psnr_pad + 1:end - psnr_pad,:) - I_deconv_bm3d(psnr_pad + 1:end - psnr_pad, psnr_pad + 1:end - psnr_pad,:);
MSE = 1/size(I_diff(:),1)*(norm(I_diff(:), 2)^2);
if MSE > eps
    PSNR_bm3d = 10*log10(1/MSE);
else
    PSNR_bm3d = Inf;
end

I_diff = I_sharp(psnr_pad + 1:end - psnr_pad, psnr_pad + 1:end - psnr_pad,:) - I_deconv_yuv(psnr_pad + 1:end - psnr_pad, psnr_pad + 1:end - psnr_pad,:);
MSE = 1/size(I_diff(:),1)*(norm(I_diff(:), 2)^2);
if MSE > eps
    PSNR_yuv = 10*log10(1/MSE);
else
    PSNR_yuv = Inf;
end

%Apply gamma
I_deconv(I_deconv < 0 ) = 0;
I_deconv = I_deconv .^ (1.0/2.0);

I_deconv_hyp(I_deconv_hyp < 0 ) = 0;
I_deconv_hyp = I_deconv_hyp .^ (1.0/2.0);

I_deconv_bm3d(I_deconv_bm3d < 0 ) = 0;
I_deconv_bm3d = I_deconv_bm3d .^ (1.0/2.0);

I_deconv_yuv(I_deconv_yuv < 0 ) = 0;
I_deconv_yuv = I_deconv_yuv .^ (1.0/2.0);

%Display result
figure(overview_figure);
for ch = 1:size(I,3)
    subplot(6,size(I,3) + 1, 2 *(size(I,3) + 1) + ch), imshow(I_deconv(:,:,ch)), axis image, title(sprintf('Proposed method Ch%d', ch));
end
subplot(6,size(I,3)+1,3 * size(I,3) + size(I,3)), imshow(I_deconv), title(sprintf('%s (PSNR %5.5g dB)', 'Proposed method', PSNR));

for ch = 1:size(I,3)
    subplot(6,size(I,3) + 1, 3 *(size(I,3) + 1) + ch), imshow(I_deconv_hyp(:,:,ch)), axis image, title(sprintf('Hyperlaplacian deconv Ch%d', ch));
end
subplot(6,size(I,3)+1,4 *(size(I,3) + 1) ), imshow(I_deconv_hyp), title(sprintf('%s (PSNR %5.5g dB)', 'Hyperlaplacian deconv', PSNR_hyp));

for ch = 1:size(I,3)
    subplot(6,size(I,3) + 1, 4 *(size(I,3) + 1) + ch), imshow(I_deconv_bm3d(:,:,ch)), axis image, title(sprintf('BM3D deconv Ch%d', ch));
end
subplot(6,size(I,3)+1, 5 *(size(I,3) + 1) ), imshow(I_deconv_bm3d), title(sprintf('%s (PSNR %5.5g dB)', 'BM3D deconv', PSNR_bm3d));

for ch = 1:size(I,3)
    subplot(6,size(I,3) + 1, 5 *(size(I,3) + 1) + ch), imshow(I_deconv_yuv(:,:,ch)), axis image, title(sprintf('YUV deconv Ch%d', ch));
end
subplot(6,size(I,3)+1, 6 *(size(I,3) + 1) ), imshow(I_deconv_yuv), title(sprintf('%s (PSNR %5.5g dB)', 'YUV deconv', PSNR_yuv));


%Comparison between ours and YUV (best of all tested algorithms)
%{
teaser_figure = figure();
subplot(1,3, 1), imshow(I_sharp_disp ), title(sprintf('Original image (%d x %d)',size(I, 2), size(I ,1)));
subplot(1,3, 2), imshow( I_blurred_disp ), title(sprintf('Blurred image'));
subplot(1,3,3), imshow(cat(2, I_deconv,I_deconv_yuv)), title('Reconstructed image');
%}

%Write out results
imwrite( I_sharp.^ (1.0/2.0), 'I_sharp.png'  );
imwrite( K_blur_disp, 'PSF.png'  );
imwrite( I_blurred_disp, 'I_blurred.png'  );

imwrite( I_deconv, 'I_deconv_ours.png'  );
imwrite( I_deconv_hyp, 'I_deconv_hyp.png'  );
imwrite( I_deconv_bm3d, 'I_deconv_bm3d.png'  );
imwrite( I_deconv_yuv, 'I_deconv_yuv.png'  );
imwrite( I_deconv_yuv, 'I_deconv_yuv.png'  );








