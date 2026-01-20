%Synthetic data test for "Primal dual deconvolution"

%Clear workspace
clear
close all hidden

%Load image and convert to grayscale
image_filename = 'images/houses_big.jpg'; 
I = imread(image_filename);
I = imresize(I, 0.15); %Sample down for faster deconvolution
I = double(I);
I = I ./ max(I(:));

%Write to disk
%imwrite( I, 'original_gamma_2_2.png', 'Bitdepth', 16 );

%Apply inverse gamma
I = I .^ (2.2);

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

%Write to disk
%imwrite( K_blur_disp, 'kernel_linear_normalize_per_channel.png', 'Bitdepth', 16 );
    
%Compute blurred images
I_blurred = zeros(size(I));
K_blur_orig = K_blur;
for ch = 1:size(I,3)
    I_blurred(:,:,ch) = imfilter(I_sharp(:,:,ch), K_blur{ch}, 'conv', 'symmetric');

    %Add noise to blurred image
    I_blurred(:,:,ch) = imnoise(I_blurred(:,:,ch), 'gaussian', 0, 0.00001);

    %And kernel image
    K_blur{ch} = imnoise(K_blur{ch}, 'gaussian', 0, 0.0000001 / ch^2);
    K_blur{ch} = K_blur{ch} ./ sum(K_blur{ch}(:)); %Normalize
end

%Write to disk
%imwrite( I_blurred, 'blurred.png', 'Bitdepth', 16 );

%Display input
overview_figure = figure();
I_sharp_disp = I_sharp.^ (1.0/2.2) ;
I_sharp_disp(1:size(K_blur_disp,1), 1:size(K_blur_disp,1),:) = K_blur_disp;
subplot(5,size(I,3)+1,1), imshow(I_sharp_disp ), title(sprintf('Original image (%d x %d)',size(I, 2), size(I ,1)));
for ch = 1:size(I,3)
    K_curr_disp = [K_blur{ch}, K_blur_orig{ch}];
    K_curr_disp = (K_curr_disp ./ max(K_curr_disp(:))) .^ (1.0/2.2);
    subplot(5,size(I,3)+1,1 + ch), imshow( K_curr_disp .^ (1/2.2) ), axis image, title(sprintf('Kernel Ch%d (%d x %d)', ch, size(K_curr_disp, 2)/2, size(K_curr_disp ,1)));
end
I_blurred_disp = I_blurred.^ (1.0/2.2);
subplot(5,size(I,3)+1, size(I,3) + 5 ), imshow( I_blurred_disp ), title(sprintf('Blurred image %d'));

%Display blurred channels
for ch = 1:size(I,3)
    subplot(5,size(I,3)+1, size(I,3) + 1 + ch), imshow(I_blurred_disp(:,:,ch) ), title(sprintf('Blurred Ch%d',ch));
end

%Prepare data
%Extract current patches and blur kernels
channel_patch = cell(0);
for ch = 1:size(I,3)
    channel_patch(ch).Image = I_blurred(:,:,ch);
    channel_patch(ch).K = K_blur{ch};
end

%Image reconstruction
fprintf('\n\nImage reconstruction with method %s started.', 'Fergus hyperlaplacian'); 

%Prepare image for hyperlaplacian deconvolution
y = zeros(size(channel_patch(1).Image, 1), size(channel_patch(1).Image, 2), length(channel_patch));
k = cell(0);
for ch = 1:size(y,3)
    y(:,:,ch) = channel_patch(ch).Image;
    k{ch} = channel_patch(ch).K;
end

%Algorithm parameters
lambda = 2e3
alpha = 0.65
rho_yuv = [0.1, 1.0, 1.0]
w_rgb = [1/4, 1/2, 1/4]
theta = max_blur_size^2 * (1e-3)^2 * 1e3

%Do the minimization
tic;
verbose = 'all';
[x] = fast_deconv(y, k, lambda, rho_yuv, w_rgb, theta, alpha);
tElapsed=toc;
fprintf('Time %5.5g seconds for Fergus hyperlaplacian deblurring. \n\n', tElapsed);
 
%Store in cell
I_deconv_channels = cell(0);
for ch = 1:size(I,3)
    I_deconv_channels(ch).Image = x(:,:,ch);
    I_deconv_channels(ch).K = k{ch};
end

%Deblur using wiener filter
I_deconv_channels_wnr = channel_patch;
for ch = 1:size(I,3)
    PSF_taper = fspecial('gaussian',10,3);
    I_deconv_channels_wnr(ch).Image = deconvwnr(edgetaper(channel_patch(ch).Image,PSF_taper), channel_patch(ch).K, 0.001);
end    

%Deblur using lucy filter
I_deconv_channels_lucy = channel_patch;
for ch = 1:size(I,3)
    PSF_taper = fspecial('gaussian',10,3);
    I_deconv_channels_lucy(ch).Image = deconvlucy(edgetaper(channel_patch(ch).Image,PSF_taper), channel_patch(ch).K, 30);
end       
                            
%Gather result
I_deconv = zeros(size(I));
for ch = 1:length(I_deconv_channels)
    I_deconv(:,:,ch) = I_deconv_channels(ch).Image;
end

I_deconv_wnr = zeros(size(I));
for ch = 1:length(I_deconv_channels)
    I_deconv_wnr(:,:,ch) = I_deconv_channels_wnr(ch).Image;
end

I_deconv_lucy = zeros(size(I));
for ch = 1:length(I_deconv_channels)
    I_deconv_lucy(:,:,ch) = I_deconv_channels_lucy(ch).Image;
end

%{
I_deconv_nli = zeros(size(I));
for ch = 1:length(I_deconv_nli_channels)
    I_deconv_nli(:,:,ch) = I_deconv_nli_channels(ch).Image;
end
%}
    
%Compute PSNR:
psnr_pad = round(size(K_blur,1) * 1.5);
I_diff = I_sharp(psnr_pad + 1:end - psnr_pad,:) - I_deconv(psnr_pad + 1:end - psnr_pad,:);
MSE = 1/size(I_diff(:),1)*(norm(I_diff(:), 2)^2);
if MSE > eps
    PSNR = 10*log10(1/MSE);
else
    PSNR = Inf;
end

I_diff = I_sharp(psnr_pad + 1:end - psnr_pad,:) - I_deconv_wnr(psnr_pad + 1:end - psnr_pad,:);
MSE = 1/size(I_diff(:),1)*(norm(I_diff(:), 2)^2);
if MSE > eps
    PSNR_wnr = 10*log10(1/MSE);
else
    PSNR_wnr = Inf;
end

I_diff = I_sharp(psnr_pad + 1:end - psnr_pad,:) - I_deconv_lucy(psnr_pad + 1:end - psnr_pad,:);
MSE = 1/size(I_diff(:),1)*(norm(I_diff(:), 2)^2);
if MSE > eps
    PSNR_lucy = 10*log10(1/MSE);
else
    PSNR_lucy = Inf;
end

%Apply gamma
I_deconv(I_deconv < 0 ) = 0;
I_deconv = I_deconv .^ (1.0/2.2);

I_deconv_wnr(I_deconv_wnr < 0 ) = 0;
I_deconv_wnr = I_deconv_wnr .^ (1.0/2.2);

I_deconv_lucy(I_deconv_lucy < 0 ) = 0;
I_deconv_lucy = I_deconv_lucy .^ (1.0/2.2);

%{
I_deconv_nli(I_deconv_nli < 0 ) = 0;
I_deconv_nli = I_deconv_nli .^ (1.0/2.2);
%}

%Display result
figure(overview_figure);
for ch = 1:size(I,3)
    subplot(5,size(I,3) + 1, 3 *size(I,3) + ch - 1), imshow(I_deconv(:,:,ch)), axis image, title(sprintf('Proposed method Ch%d', ch));
end
subplot(5,size(I,3)+1,3 * size(I,3) + size(I,3)), imshow(I_deconv), title(sprintf('%s (PSNR %5.5g dB)', 'Proposed method', PSNR));

for ch = 1:size(I,3)
    subplot(5,size(I,3) + 1, 3 *(size(I,3) + 1) + ch), imshow(I_deconv_wnr(:,:,ch)), axis image, title(sprintf('Wiener filter Ch%d', ch));
end
subplot(5,size(I,3)+1,4 *(size(I,3) + 1) ), imshow(I_deconv_wnr), title(sprintf('%s (PSNR %5.5g dB)', 'Wiener filter', PSNR_wnr));

for ch = 1:size(I,3)
    subplot(5,size(I,3) + 1, 4 *(size(I,3) + 1) + ch), imshow(I_deconv_lucy(:,:,ch)), axis image, title(sprintf('Richardson-Lucy Ch%d', ch));
end
subplot(5,size(I,3)+1, 5 *(size(I,3) + 1) ), imshow(I_deconv_lucy), title(sprintf('%s (PSNR %5.5g dB)', 'Richardson-Lucy', PSNR_lucy));

%Teaser figure
teaser_figure = figure();
subplot(1,3, 1), imshow(I_sharp_disp ), title(sprintf('Original image (%d x %d)',size(I, 2), size(I ,1)));
subplot(1,3, 2), imshow( I_blurred_disp ), title(sprintf('Blurred image'));
subplot(1,3,3), imshow(I_deconv), title('Reconstructed image');

%Write to disk
%imwrite(I_sharp, 'original.png')
%imwrite(I_deconv, 'reconstruction.png')
%imwrite(I_deconv_nli, 'reconstruction_nli.png')
