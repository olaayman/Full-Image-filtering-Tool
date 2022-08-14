classdef FullImageFilteringTool < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                        matlab.ui.Figure
        TabGroup                        matlab.ui.container.TabGroup
        ImageDisplayerTab               matlab.ui.container.Tab
        BrowseButton                    matlab.ui.control.Button
        Label                           matlab.ui.control.Label
        ConvertToGrayscaleButton        matlab.ui.control.Button
        UIAxes                          matlab.ui.control.UIAxes
        PixelscoloringTab               matlab.ui.container.Tab
        CreateimageButton               matlab.ui.control.Button
        colorpixelsButton               matlab.ui.control.Button
        Image                           matlab.ui.control.Image
        UIAxes_2                        matlab.ui.control.UIAxes
        ZoomTab                         matlab.ui.container.Tab
        NearestNeighborButton           matlab.ui.control.Button
        BilinearButton                  matlab.ui.control.Button
        InputZoomingfactorEditField_2Label  matlab.ui.control.Label
        InputZoomingfactorEditField_2   matlab.ui.control.NumericEditField
        NearestNeighborPanel            matlab.ui.container.Panel
        Nearest_UIAxes                  matlab.ui.control.UIAxes
        BilinearPanel                   matlab.ui.container.Panel
        Bilinear_UIAxes                 matlab.ui.control.UIAxes
        BeforeZoomingPanel              matlab.ui.container.Panel
        NoZoom_UIAxes                   matlab.ui.control.UIAxes
        HistogramTab                    matlab.ui.container.Tab
        equalizedimagePanel             matlab.ui.container.Panel
        equalized_image_axes            matlab.ui.control.UIAxes
        originalimagePanel              matlab.ui.container.Panel
        original_image_axes             matlab.ui.control.UIAxes
        originalhistogramPanel          matlab.ui.container.Panel
        original_histogram_axes         matlab.ui.control.UIAxes
        equalizedhistogramPanel         matlab.ui.container.Panel
        equalized_histogram_axes        matlab.ui.control.UIAxes
        ShowHistogramButton             matlab.ui.control.Button
        EqualizeButton                  matlab.ui.control.Button
        SpatialFilteringTab             matlab.ui.container.Tab
        BeforFilterPanel                matlab.ui.container.Panel
        BeforeFilter_axes               matlab.ui.control.UIAxes
        AfterFilteringPanel             matlab.ui.container.Panel
        AfterFiltering_axes             matlab.ui.control.UIAxes
        ApplyFilterButton               matlab.ui.control.Button
        KernelsizeEditField_2Label      matlab.ui.control.Label
        KernelsizeEditField_2           matlab.ui.control.NumericEditField
        factorKEditFieldLabel           matlab.ui.control.Label
        factorKEditField                matlab.ui.control.NumericEditField
        BoxFilterPanel                  matlab.ui.container.Panel
        BoxFilter_axes                  matlab.ui.control.UIAxes
        edgesPanel                      matlab.ui.container.Panel
        edges_axes                      matlab.ui.control.UIAxes
        ScalingmethodButtonGroup        matlab.ui.container.ButtonGroup
        thresholdmethodButton           matlab.ui.control.RadioButton
        scalethenewrangeButton          matlab.ui.control.RadioButton
        Fourier1Tab                     matlab.ui.container.Tab
        TimeDomainPanel                 matlab.ui.container.Panel
        TimeDomain_axes                 matlab.ui.control.UIAxes
        PhasePanel                      matlab.ui.container.Panel
        Phase_axes                      matlab.ui.control.UIAxes
        MagnitudePanel                  matlab.ui.container.Panel
        Magnitude_axes                  matlab.ui.control.UIAxes
        DiplayFouriertransformButton    matlab.ui.control.Button
        removefrequenciesPanel          matlab.ui.container.Panel
        selectfreqinXstartSpinnerLabel  matlab.ui.control.Label
        selectfreqinXstartSpinner       matlab.ui.control.Spinner
        removefrequenciesButton         matlab.ui.control.Button
        selectfreqinYstartSpinnerLabel  matlab.ui.control.Label
        selectfreqinYstartSpinner       matlab.ui.control.Spinner
        selectfreqinXendSpinnerLabel    matlab.ui.control.Label
        selectfreqinXendSpinner         matlab.ui.control.Spinner
        selectfreqinYendSpinnerLabel    matlab.ui.control.Label
        selectfreqinYendSpinner         matlab.ui.control.Spinner
        Phase_axes_2                    matlab.ui.control.UIAxes
        Fourier2Tab                     matlab.ui.container.Tab
        TimeDomainBeforePanel           matlab.ui.container.Panel
        TimeDomainBefore_axes           matlab.ui.control.UIAxes
        AfterFilterFreqPanel            matlab.ui.container.Panel
        AfterFilterFreq_axes            matlab.ui.control.UIAxes
        DifferencePanel                 matlab.ui.container.Panel
        Difference_axes                 matlab.ui.control.UIAxes
        KernelsizeEditField_3Label      matlab.ui.control.Label
        Kernelsize2                     matlab.ui.control.NumericEditField
        ApplyfilterinFrequencyDomainButton  matlab.ui.control.Button
        commentLabel                    matlab.ui.control.Label
        NoiseTab                        matlab.ui.container.Tab
        PhantomPanel                    matlab.ui.container.Panel
        DisplayPhantomButton            matlab.ui.control.Button
        phantom_axes                    matlab.ui.control.UIAxes
        NoisyImgPanel                   matlab.ui.container.Panel
        NoisyImg_axes                   matlab.ui.control.UIAxes
        HistogramPanel                  matlab.ui.container.Panel
        histogram_axes                  matlab.ui.control.UIAxes
        SelectROIButton                 matlab.ui.control.Button
        ROIPanel                        matlab.ui.container.Panel
        ROI_axes                        matlab.ui.control.UIAxes
        SelectNoiseButtonGroup          matlab.ui.container.ButtonGroup
        GaussiannoiseButton_2           matlab.ui.control.RadioButton
        UniformnoiseButton_2            matlab.ui.control.RadioButton
        NoneButton_2                    matlab.ui.control.RadioButton
        SaltpepperButton                matlab.ui.control.RadioButton
        SaltLabel                       matlab.ui.control.Label
        saltPercent                     matlab.ui.control.NumericEditField
        pepperLabel                     matlab.ui.control.Label
        pepperPercent                   matlab.ui.control.NumericEditField
        BackProjectionTab               matlab.ui.container.Tab
        lamino5Panel                    matlab.ui.container.Panel
        Lamino5_axes                    matlab.ui.control.UIAxes
        phantom2panel                   matlab.ui.container.Panel
        SheppPhantom_axes               matlab.ui.control.UIAxes
        lamino180Panel                  matlab.ui.container.Panel
        lamino180_axes                  matlab.ui.control.UIAxes
        ramLackPanel                    matlab.ui.container.Panel
        ramLack_axes                    matlab.ui.control.UIAxes
        hammingPanel                    matlab.ui.container.Panel
        hamming_axes                    matlab.ui.control.UIAxes
        GetBackProjectoinButton         matlab.ui.control.Button
        sinogramPanel                   matlab.ui.container.Panel
        sinogram_axes                   matlab.ui.control.UIAxes
        ColormapTab                     matlab.ui.container.Tab
        CTImgPanel                      matlab.ui.container.Panel
        CTImg_axes                      matlab.ui.control.UIAxes
        displaytheimageButton           matlab.ui.control.Button
    end

    
    properties (Access = private)
        whiteImage 
        rgbImage
        rgbImageSize
        grayImage
        grayImageSize
        generate
        normalizedHisto
        enhancedImg
        imgMagnitude
        imgPhase
        convImg
        spatialFilter
        phantom1
        noisyImg
        coordinates
        rec
        rec1
    end
    
    methods (Access = private)
        
        function grayMatrix = convertToGrayImage(app,rgbMatrix)
            red = rgbMatrix(:,:,1);
            green = rgbMatrix(:,:,2);
            blue = rgbMatrix(:,:,3);
            grayMatrix=(red*0.2989)+(green*0.5870)+(blue*0.114);
        end
        
        
        function DICOMDataText = getDICOMData(app,info)
            DicomInfo=[];
            if class(info.PatientName)== "struct"
                DicomInfo=[DicomInfo ,"anonymous"];
                %warndlg("patient name is anonymous","warning");
            else
                DicomInfo=[DicomInfo ,info.PatientName];
            end
        
            try
                DicomInfo=[DicomInfo ,info.PatientAge];
            catch 
                DicomInfo=[DicomInfo ,"none"];
                %warndlg("Dicom file does not have a patient age","warning");

            end
            try
                DicomInfo=[DicomInfo ,info.BodyPartExamined];
            catch
                DicomInfo=[DicomInfo ,"none"];
                %warndlg("Dicom file does not have the examined parts","warning");

            end
            try
                DicomInfo=[DicomInfo ,info.Modality];
            catch 
                DicomInfo=[DicomInfo ,"none"];
                %warndlg("Dicom file does not have a Modality","warning");

            end
            DICOMDataText = ["Patient name is " + DicomInfo(1) ,"Patient Age is " + DicomInfo(2) ,"Body Part Examined is "+ DicomInfo(3) ,  "Modality used is " + DicomInfo(4) ];

        end
        
        function ImageBasicDataText = getBasicData(app , info)
            size = info.Height*info.Width *info.BitDepth ;
            ImageBasicDataText = ["width = " + info.Width , "Height = " + info.Height , "BitDepth = "+ info.BitDepth ,"Image total size in bits  = "+ size, "Color Type is " + info.ColorType ];
        end
        
        function DispImages = DisplayImages(app)
            imshow(app.grayImage,'Parent',app.NoZoom_UIAxes);
            app.NoZoom_UIAxes.Position = [552 473 size(app.grayImage,1) size(app.grayImage,2)];
            imshow(app.grayImage,'Parent',app.original_image_axes);
            cla(app.original_histogram_axes);
            cla(app.equalized_image_axes);
            cla(app.equalized_histogram_axes);
            imshow(app.grayImage,'Parent',app.BeforeFilter_axes);
            cla(app.AfterFiltering_axes);
            imshow(app.grayImage,'Parent',app.TimeDomain_axes);
            cla(app.Magnitude_axes);
            cla(app.Phase_axes);
            imshow(app.grayImage,'Parent',app.TimeDomainBefore_axes);
            cla(app.AfterFilterFreq_axes);
            cla(app.Difference_axes);
            
        end
        
        function [mag ,phase] = GetFourierTrans(app , img)
            % Perform 2D FFTs
            Imgfft = fft2(double(img));
            shiftedImgFFT = fftshift(Imgfft);
            
            % Display magnitude and phase of 2D FFTs
            ImgMag=abs(shiftedImgFFT);                           
            ImgPhase=angle(shiftedImgFFT);
            mag = log(ImgMag);
            phase = log(ImgPhase);
            
        end
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: BrowseButton
        function BrowseButtonPushed(app, event)
            %global image1;
            try
                [filename, pathname] = uigetfile({'*.*' ; '*.jpeg';'*.dcm';'*.bmp'},"open file","D:\collage\senior1 fall\image processing\task1ImageDisplay");
                fullPathName = strcat(pathname,filename); 
                [filepath,name,ext] = fileparts(fullPathName);
                display(ext);
                if ext == ".dcm" || ext == ".DCM"
                    info = dicominfo(fullPathName);
                    disp(info);
                    app.rgbImage = dicomread(info);
                    disp(app.rgbImage);
                    imshow(app.rgbImage,[],'Parent',app.UIAxes);
                    app.Label.Text = [getBasicData(app , info) ,getDICOMData(app , info)];
                    app.grayImage = app.rgbImage;
                    
                else
                    app.rgbImage = imread(fullPathName);
                    imageInfo = imfinfo(fullPathName);
                    [~ ,~ ,numberOfColorChannels] = size(app.rgbImage);
                    disp(numberOfColorChannels);  
                    imshow(app.rgbImage,'Parent',app.UIAxes); 
                    app.Label.Text = getBasicData(app , imageInfo);
                    if numberOfColorChannels == 1
                        app.grayImage = app.rgbImage;
                        DisplayImages(app);
%                         imshow(app.grayImage,'Parent',app.original_image_axes);
%                         %app.original_image_axes.Position = [552 473 size(app.grayImage,1) size(app.grayImage,2)];
%                         cla(app.original_histogram_axes);
%                         cla(app.equalized_image_axes);
%                         cla(app.equalized_histogram_axes);
%                         imshow(app.grayImage,'Parent',app.BeforeFilter_axes);
%                         cla(app.AfterFiltering_axes);
            
                    end
                    %app.Image_2.ImageSource = app.rgbImage;
                    %app.Image_2.ScaleMethod = 'none';
                    %[rows columns numberOfColorChannels] = size(image1);
                    %disp(numberOfColorChannels);
                end
                            
           
            catch
                app.Label.Text = "";
                errordlg('file selected is not supported or corrupted','Error message');
            end    
           
        end

        % Button pushed function: colorpixelsButton
        function colorPixelsPushed(app, event)
            if app.generate
                app.whiteImage(2,47:50,2:3)=0;
                app.whiteImage(47:50,2,1:2)=0;
                app.Image.ImageSource = app.whiteImage;
                app.Image.ScaleMethod = 'none';
                imshow(app.whiteImage,'Parent',app.UIAxes_2);
                
                %truesize(app.UIFigure);
            else
                errordlg('generate image first','Error message');
            end
        end

        % Callback function: CreateimageButton, UIAxes_2
        function createImagePushed(app, event)
            app.whiteImage = zeros(50,50,3);
            app.whiteImage(:,:,:)=255;
            
            app.Image.ImageSource = app.whiteImage;
            app.Image.ScaleMethod = 'none';
            app.UIAxes_2.Clipping = 'off';
            imshow(app.whiteImage,'Parent',app.UIAxes_2);
            %imshow(app.whiteImage,app.Image);
            app.generate = true;
        end

        % Button pushed function: ConvertToGrayscaleButton
        function ConvertToGrayscaleButtonPushed(app, event)
            [~ ,~ ,numberOfColorChannels] = size(app.rgbImage);
            app.rgbImageSize = size(app.rgbImage);
            if numberOfColorChannels == 3
                app.grayImage = convertToGrayImage(app,app.rgbImage);
                imshow(app.grayImage,'Parent',app.UIAxes);
                [~,~,numberOfColorChannels] = size(app.grayImage);
                app.grayImageSize = size(app.grayImage);
                app.Label.Text = "Number of color Channels is now " + numberOfColorChannels;
                
            else 
                
                errordlg('image selected is alleady gray','Error message');
                    
            end
%             imshow(app.grayImage,'Parent',app.NoZoom_UIAxes);
%             app.NoZoom_UIAxes.Position = [552 473 size(app.grayImage,1) size(app.grayImage,2)];
            DisplayImages(app);
%             imshow(app.grayImage,'Parent',app.original_image_axes);
%             cla(app.original_histogram_axes);
%             cla(app.equalized_image_axes);
%             cla(app.equalized_histogram_axes);
%             imshow(app.grayImage,'Parent',app.BeforeFilter_axes);
%             cla(app.AfterFiltering_axes);
%             
        end

        % Button pushed function: NearestNeighborButton
        function NearestNeighborButtonPushed(app, event)
            try
                if app.InputZoomingfactorEditField_2.Value >0 
                    zoomFactor = app.InputZoomingfactorEditField_2.Value;
                else 
                    zoomFactor = 1;
                end
                
                rowPositions = [ones(1,ceil(zoomFactor)),round([zoomFactor:size(app.grayImage,2)*zoomFactor]./zoomFactor)];
                colPositions = [ ones(1,ceil(zoomFactor)),round([zoomFactor:size(app.grayImage,1)*zoomFactor]./zoomFactor)];
                zoomed = app.grayImage(:,rowPositions);
                zoomed = zoomed(colPositions,:);
                imshow(zoomed, 'parent', app.Nearest_UIAxes);
                app.Nearest_UIAxes.Position = [0 0 size(zoomed,1) size(zoomed,2)];
            catch
                errordlg('there is no gray scale image','Error message');
            end
            % sources
            % https://www.imageeprocessing.com/2017/11/nearest-neighbor-interpolation.html?m=1
        end

        % Button pushed function: BilinearButton
        function BilinearButtonPushed(app, event)
            try
                if app.InputZoomingfactorEditField_2.Value >0 
                    ratio = app.InputZoomingfactorEditField_2.Value;
                else 
                    ratio = 1;
                end
                [h, w,~] = size(app.grayImage);
                for i=1:h*ratio
                    %calculate the y coordinate
                      y = i/ratio;
                      if y < 1
                          y = 1;
                      elseif y >= h
                        y = h-1;
                      end
                      y1 = floor(y);
                      y2 = y1 + 1;
                  
                    for j=1:w*ratio
                        %calculate the x coordinate
                           x =  j/ratio ;
                          if x < 1
                              x = 1;
                          elseif x >= w
                              x = w-1;
                          end
                          x1 = floor(x);
                          x2 = x1 + 1;
                          
                  %calculate the Neighbors
                      N1 = app.grayImage(y1,x1);
                      N2 = app.grayImage(y1,x2);
                      N3 = app.grayImage(y2,x1); 
                      N4 = app.grayImage(y2,x2);
            
                  %calculate the square area
           
                      PW1 = (y2-y)*(x2-x);
                      PW2 = (y2-y)*(x-x1);
                      PW3 = (x2-x)*(y-y1);
                      PW4 = (y-y1)*(x-x1);
            
                      Zoomed(i,j) = PW1 * N1 + PW2 * N2 + PW3 * N3 + PW4 * N4;
            
                    end
                end
                imshow(Zoomed, 'parent', app.Bilinear_UIAxes);
                app.Bilinear_UIAxes.Position = [0 0 size(Zoomed,1) size(Zoomed,2)];
            catch
                errordlg('there is no gray scale image','Error message');
            end
        end

        % Button pushed function: ShowHistogramButton
        function ShowHistogramButtonPushed(app, event)
            Histo = zeros(1,256);
            [h,w,~] = size(app.grayImage);
            % we use the intensity as an index to the histo to find the
            % frequency of each intensity
            % we add 1 as the intensities start with 0 but the indexies
            % start with 1
            for i=1:h
                for j=1:w
                    Histo(app.grayImage(i,j)+1) = Histo(app.grayImage(i,j)+1) + 1;
                end
            end
            % to normalize we divide on the no of pixels
            app.normalizedHisto = Histo./(h*w);
            bar( app.original_histogram_axes , app.normalizedHisto );
        end

        % Button pushed function: EqualizeButton
        function EqualizeButtonPushed(app, event)
            try 
                equalizedHisto = zeros(1,256);
                % insialize a matrix for the equalized image 
                equalizedImg=uint8(zeros(size(app.grayImage,1),size(app.grayImage,2)));
                cdf = 0;
                for i=1:256
                    cdf = cdf + app.normalizedHisto(i);
                    CDF(i)= cdf;
                    sk(i) = round(CDF(i)*255);
                    equalizedHisto(round(CDF(i)*255)+1) = equalizedHisto(round(CDF(i)*255)+1) + app.normalizedHisto(i);
                end
                bar( app.equalized_histogram_axes , equalizedHisto );
                [h,w,~] = size(app.grayImage);
                % value Rk (app.grayImage(i,j)) has become Sk so we will use
                % the rk as an index for the sk to get the new value
                for i=1:h
                    for j=1:w
                        equalizedImg(i,j) = sk(app.grayImage(i,j)+1);
                    end
                end
                imshow(equalizedImg , 'Parent',app.equalized_image_axes);
            catch
                errordlg('get the histogram of the image first','Error message');
            end
        end

        % Button pushed function: ApplyFilterButton
        function ApplyFilterButtonPushed(app, event)
            try
                kernelSize= app.KernelsizeEditField_2.Value;
                if rem(kernelSize,2) == 0
                    kernelSize = kernelSize + 1;
                end
                if kernelSize < 0
                    kernelSize = -kernelSize;
                end
                K = app.factorKEditField.Value;
                if K <0 
                    K = -K;
                end
                % make the box filter kernel
                kernel = ones(kernelSize,kernelSize) ./ (kernelSize^2);
                % zero padding 
                [h,w,~]=size(app.grayImage);
                newH= h+kernelSize-1;
                newW= w+kernelSize-1;
                paddedImg = zeros(newH,newW);
                paddedImg(floor(kernelSize/2)+1:end-ceil(kernelSize/2)+1,floor(kernelSize/2)+1:end-ceil(kernelSize/2)+1)= app.grayImage;
                % convolution the box filter with the image to get the blurr image
                app.convImg = zeros(h,w);
                for i = 1:newH-kernelSize+1
                    for j = 1:newW-kernelSize+1
                        Temp = paddedImg(i:i+kernelSize-1,j:j+kernelSize-1) .* kernel;
                        app.convImg(i,j) = round(sum(Temp(:)));
                    end
                end
                % show the blurred image
                app.spatialFilter = uint8(app.convImg);
                imshow(app.spatialFilter,'parent' , app.BoxFilter_axes);
                Img = zeros(h,w);
                Img(:,:) = app.grayImage;
                % subtract the blurred img from the original image to get the edges
                edges = (Img - app.convImg) .* K;
                app.enhancedImg = Img + edges;
                edges = uint8(edges);
                imshow(edges , 'parent', app.edges_axes);
                % apply the choosen scaling method from the GUI
                if app.thresholdmethodButton.Value
                    afterFilter = uint8(app.enhancedImg);
                    imshow(afterFilter ,'parent' , app.AfterFiltering_axes);
                else
                    % scalling
                    positiveImg = app.enhancedImg - min(app.enhancedImg(:));
                    scaledImg = (positiveImg ./ max(positiveImg(:))) .* 255;
                    scaledImg = uint8(scaledImg);
                    imshow(scaledImg , 'parent' , app.AfterFiltering_axes);
                end
            catch
                errordlg("please enter a positive integer for the kernel size");
            end
        end

        % Button pushed function: DiplayFouriertransformButton
        function DiplayFouriertransformButtonPushed(app, event)
            [app.imgMagnitude , app.imgPhase] = GetFourierTrans(app , app.grayImage);
%             % Perform 2D FFTs
%             Imgfft = fft2(double(app.grayImage));
%             shiftedImgFFT = fftshift(Imgfft);
%             
%             % Display magnitude and phase of 2D FFTs
%             ImgMag=abs(shiftedImgFFT);                           
%             ImgPhase=angle(shiftedImgFFT);
%             ImgMagLog = log(ImgMag);
%             ImgPhaseLog = log(ImgPhase);
            imshow(app.imgMagnitude ,[], 'parent' , app.Magnitude_axes);
            imshow(app.imgPhase , [],'parent' , app.Phase_axes);

        end

        % Button pushed function: ApplyfilterinFrequencyDomainButton
        function ApplyfilterinFrequencyDomainButtonPushed(app, event)
%             try
                [h,w,~]=size(app.grayImage);
                kernelSize= app.Kernelsize2.Value;
                if rem(kernelSize,2) == 0
                    kernelSize = kernelSize + 1;
                end
                if kernelSize < 0
                    kernelSize = -kernelSize;
                end
                if kernelSize >= h || kernelSize >=w 
                    errordlg("enter a kernel size small than the image height and width");
                    kernelSize = 0;
                end
                % make the box filter kernel
                kernel = ones(kernelSize,kernelSize) ./ (kernelSize^2);
                
                %image padding if even
                %disp(size(app.grayImage));
                originalGrayImage = app.grayImage;
                if(rem(h,2) == 0)
                    app.grayImage(h+1,1)=0;
                end
                if(rem(w,2) == 0)
                    app.grayImage(1,w+1)=0;
                end
                % kernel padding 
                %disp(size(app.grayImage));
                [h,w,~]=size(app.grayImage);
                paddedKernel = zeros(h,w);
                paddedKernel(floor((h-kernelSize)/2)+1: end- ceil((h-kernelSize)/2),floor((w-kernelSize)/2)+1: end- ceil((w-kernelSize)/2)) = kernel;
                
                % apply fourier transform for image and kernel
                fftImg = fft2(app.grayImage);
                fftKernel = fft2(paddedKernel);
                
                % apply the filter
                FilteredImg = fftKernel .* fftImg;
                % Get inverse fourier and Display it
                FilteredImgTimeDomain = ifftshift(ifft2(FilteredImg));
                FilteredImgTimeDomain = uint8(FilteredImgTimeDomain);
                imshow( FilteredImgTimeDomain, 'parent' , app.AfterFilterFreq_axes);
                
                % Call the spatial filter with the same kernel size
                app.KernelsizeEditField_2.Value = kernelSize;
                app.factorKEditField.Value = 1 ; 
                ApplyFilterButtonPushed(app, matlab.ui.eventdata.ButtonPushedData);
                
                % Get the diff between freq and spatial
                DiffTimeAndFreq = FilteredImgTimeDomain - app.spatialFilter ; 
                imshow( DiffTimeAndFreq,[], 'parent' , app.Difference_axes);
                app.commentLabel.Text= "the final image shows the difference between the results of the box filter in both the spatial and frequency domains which are almost the same there for the image is black ";
                app.grayImage = originalGrayImage;
%             catch 
%                 errordlg("enter a kernel size small than the image height and width");
%             end
        end

        % Button pushed function: removefrequenciesButton
        function removefrequenciesButtonPushed(app, event)
            
            try
                xStart = app.selectfreqinXstartSpinner.Value;
                xEnd = app.selectfreqinXendSpinner.Value;
                yStrat = app.selectfreqinYstartSpinner.Value;
                yEnd = app.selectfreqinYendSpinner.Value;
                Imgfft = fft2(double(app.grayImage));
                %imshow(uint8(Imgfft), "Parent", app.Phase_axes_2);
                Imgfft(xStart:xEnd,yStrat:yEnd)= 0;
                shiftedImgFFT = fftshift(Imgfft);
                ImgMag=abs(shiftedImgFFT);                           
                mag = log(ImgMag);
                imshow(mag,[] , "Parent", app.Phase_axes_2);
            catch 
                errordlg("please enter a positive interger inrange or select a image")
            end
        end

        % Button pushed function: DisplayPhantomButton
        function DisplayPhantomButtonPushed(app, event)
            % generate the phantom
            % outer frame 
            app.phantom1 = ones(256,256) .* 50;
            % middle frame
            app.phantom1(40: end-40 , 40: end-40) = 150;
            % circle
            % get the cols and rows
            [columnsInImage ,rowsInImage] = meshgrid(1:256, 1:256);
            % determine the center h/2 and w/2
            centerX = 128;
            centerY = 128;
            % determine radius
            radius = 60;
            % check for every pixel if its distance from center <= radius
            % returns a boolen 
            circlePixels = (rowsInImage - centerX).^2 + (columnsInImage - centerY).^2 <= radius.^2;
            % for all pixels with bool 1 make the value 250
            app.phantom1(circlePixels) = 250;
            imshow(uint8(app.phantom1) , 'parent', app.phantom_axes);
        end

        % Button pushed function: SelectROIButton
        function SelectROIButtonPushed(app, event)
         try
      
            if app.noisyImg
                % open a figure to select the ROI from it
                imshow(uint8(app.noisyImg));
                % takes the row and col position for the 2 points
                [col , row] = ginput(2);
                hold(app.NoisyImg_axes,'on');
                % determine the start and end of the ROI in row and col
                startr = min(row(1),row(2));
                endr = max(row(1),row(2));
                startc = min(col(1),col(2));
                endc = max(col(1),col(2));
                h= abs(row(2)-row(1));
                w= abs(col(2)-col(1));
                disp(col);
                disp(row);
                ROI = zeros(h+1,w+1);
                % set the ROI values from the noisy image and display it
                ROI(:,:)= app.noisyImg(startr:endr,startc:endc);
                ROI =uint8(ROI);
                imshow(ROI,'Parent',app.ROI_axes);
                hold(app.NoisyImg_axes,'off');
                %intializa the local histogram and sum the pixels with same
                %intensity then diaplay it
                Histo = zeros(1,256);
                for i=1:h
                    for j=1:w
                        Histo(round(ROI(i,j))+1) = Histo(round(ROI(i,j))+1) + 1;
                    end
                end
                Histo = Histo ./ (h*w);
                bar( app.histogram_axes ,Histo);
                intensities = [0:255];
                mean = sum(intensities .* Histo);
                disp(mean);
                xline(mean, 'Color', 'g', 'LineWidth', 1 ,'parent' ,app.histogram_axes);
                std = sqrt(sum(((intensities - mean) .^ 2) .* Histo));
                xline(mean - std, 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--','parent' ,app.histogram_axes);
                xline(mean + std,'Color', 'r', 'LineWidth', 1, 'LineStyle', '--','parent' ,app.histogram_axes);
                TextBox = sprintf('  Mean = %.3f\n  SD = %.3f', mean, std);
                % Position the text 90% of the way from bottom to top.
                text(mean +2*std,0.02, TextBox ,"Parent",app.histogram_axes);
                
                disp(std);
                
         else 
             errordlg("display the phantom and select the noise first");
         end
        catch
            errordlg("You didn't select ROI");
        end

        end

        % Callback function: NoisyImg_axes, SelectNoiseButtonGroup
        function SelectNoiseButtonGroupSelectionChanged(app, event)
            selectedButton = app.SelectNoiseButtonGroup.SelectedObject;
            roiPos = [];
            
            if app.phantom1
                x=1;
            else
                DisplayPhantomButtonPushed(app, matlab.ui.eventdata.ButtonPushedData);
            end
            Noise = zeros(size(app.phantom1));
            if selectedButton.Text == "Uniform noise"
                % 
                Noise = unifrnd(-10,10,size(app.phantom1));
            elseif selectedButton.Text == "Gaussian noise"
                % perform the equation (1/ std *sqrt(2pi)) *exp((-(x-mean)^2/2std^2)
                Noise = normrnd(0,5,size(app.phantom1));
            elseif  selectedButton.Text == "None"
                Noise = zeros(size(app.phantom1));
            end
            app.noisyImg = app.phantom1 + Noise;
            
            if selectedButton.Text == "Salt & pepper"
                saltPer = app.saltPercent.Value;
                pepperPer = app.pepperPercent.Value;
                if saltPer +pepperPer <= 100
                    salt_pepper = randi([0,100], size(app.phantom1,1),size(app.phantom1,2));
                    salt = salt_pepper < saltPer;
                    pepper = salt_pepper > 100-pepperPer;
                    app.noisyImg(salt) = 255;
                    app.noisyImg(pepper) = 0;
                else
                    errordlg("select reasonable percentage (sum < 100)");
                end
            end
            imageHandle = imshow(uint8(app.noisyImg) , 'parent', app.NoisyImg_axes );
            set(imageHandle,'ButtonDownFcn',@noisyImgClicked);
            function noisyImgClicked ( ~ , eventData )
                 coords = eventData.IntersectionPoint;
                 disp(coords);
                   if size(roiPos,2) < 4
                        roiPos(size(roiPos,2)+1:size(roiPos,2)+2) = [round(coords(2)),round(coords(1))];
                    else
                        roiPos =[];
                        roiPos(size(roiPos,2)+1:size(roiPos,2)+2) = [round(coords(2)),round(coords(1))];
                   end
                   disp(roiPos);
                   if size(roiPos,2) == 4
                       startr = min(roiPos(1),roiPos(3));
                        endr = max(roiPos(1),roiPos(3));
                        startc = min(roiPos(2),roiPos(4));
                        endc = max(roiPos(2),roiPos(4));
                        h= endr-startr;
                        w= endc-startc;
                        ROI = zeros(h+1,w+1);
                        % set the ROI values from the noisy image and display it
                        ROI(:,:)= app.noisyImg(startr:endr,startc:endc);
                        ROI =uint8(ROI);
                        imshow(ROI,'Parent',app.ROI_axes);
                        recPos = [startc ,startr, w  , h];
                        delete(app.rec); 
                        app.rec = rectangle(app.NoisyImg_axes, 'Position', recPos, "EdgeColor",'#0072BD',"LineWidth",1,"FaceColor",[0.3010 ,0.7450 ,0.9330,0.2]);
                        
                        %intializa the local histogram and sum the pixels with same
                        %intensity then diaplay it
                        Histo = zeros(1,256);
                        for i=1:h
                            for j=1:w
                                Histo(round(ROI(i,j))+1) = Histo(round(ROI(i,j))+1) + 1;
                            end
                        end
                        Histo = Histo ./ (h*w);
                        bar( app.histogram_axes ,Histo);
                        intensities = [0:255];
                        mean = sum(intensities .* Histo); 
                        xline(mean, 'Color', 'r', 'LineWidth', 1 ,'parent' ,app.histogram_axes);
                        std = sqrt(sum(((intensities - mean) .^ 2) .* Histo));
                        xline(mean - std, 'Color', 'y', 'LineWidth', 1, 'LineStyle', '--','parent' ,app.histogram_axes);
                        xline(mean + std, 'Color', 'y', 'LineWidth', 1, 'LineStyle', '--','parent' ,app.histogram_axes);

                        TextBox = sprintf('  Mean = %.3f\n  SD = %.3f', mean, std);
                        % Position the text
                        text(mean +std +5,0.02, TextBox ,"Parent",app.histogram_axes);
                        
                   end     
                  
            end
        end

        % Callback function
        function SelectNoiseButtonGroupSelectionChanged2(app, event)
            
        end

        % Button down function: phantom_axes
        function phantom_axesButtonDown(app, event)
            
        end

        % Callback function
        function GetBackProjectoinButtonPushed(app, event)
            
        end

        % Button pushed function: GetBackProjectoinButton
        function GetBackProjectoinButtonPushed2(app, event)
            % get the sheep logan phantom and display it
            sheppPantom = phantom('Modified Shepp-Logan',256);
            imshow(sheppPantom , 'Parent',app.SheppPhantom_axes);
            % set the 5 angels in an array to use
            angels1 =  [0, 20, 40, 60, 160];
            % get the projections at the 5 angels and inverse radon to
            % get the image from projections and display it
            lamino5projections = radon(sheppPantom , angels1);
            lamino5 = iradon(lamino5projections,angels1);
            imshow(lamino5,'Parent',app.Lamino5_axes);
            % set 180 angle with 1 steps 
            angels2 = 0:1:179;
            [lamino180projections,x_axis] = radon(sheppPantom,angels2);
            % flip the projections to be rows in the image instead of cols
            lamino5projectionsFliped= flipud(lamino180projections');
            % display the sinogram
            imshow(lamino5projectionsFliped,[], 'XData',x_axis(:),'YData', angels2,'Parent',app.sinogram_axes );
            % get inverse radon to getthe image but with no filters and
            % display it
            lamino180 = iradon(lamino180projections,angels2,'linear','none');
            imshow(lamino180,[],'Parent',app.lamino180_axes);
            % use ram-lack filter property
            ram_lack = iradon(lamino180projections,angels2,'linear','Ram-Lak');
            imshow(ram_lack,[],'Parent',app.ramLack_axes);
            % use hamming filter property
            hamming = iradon(lamino180projections,angels2,'linear','Hamming');
            imshow(hamming,[],'Parent',app.hamming_axes);
        end

        % Button down function: CTImg_axes
        function CTImg_axesButtonDown(app, event)
          
        end

        % Button pushed function: displaytheimageButton
        function displaytheimageButtonPushed(app, event)
            % read image and display it
            CT_RGB = imread("US Image.jpeg");
            imageHandle = imshow(CT_RGB, 'Parent',app.CTImg_axes);
            roiPos = [];
            disp("batata");
            % pad the image to get variance
            kernelSize = 25;
            [h,w,~]=size(CT_RGB);
            paddedCT = zeros(h+24,w+24);
            paddedCT(floor(kernelSize/2)+1:end-ceil(kernelSize/2)+1,floor(kernelSize/2)+1:end-ceil(kernelSize/2)+1)= CT_RGB(:,:,1);
            
            % calculate local variance
            Variance = zeros(h,w);
            for r = 1:h
                for c = 1:w
                    Histo = zeros(1,256);
                    for i=r:r+24
                        for j=c:c+24
                            Histo(round(paddedCT(i,j))+1) = Histo(round(paddedCT(i,j))+1) + 1;
                        end
                    end
                    Histo = Histo ./ (25*25);
                    intensities = 0:255;
                    mean = sum(intensities .* Histo); 
                    Variance(r,c) = sum(((intensities - mean) .^ 2) .* Histo);
                end
            end
            %Variance = uint8(255 * mat2gray(Variance));
           % range = max(Variance(:))-min(Variance(:));
            % convert gray to RGB and display it
%             CT_RGB = zeros(h,w,3);
%             CT_RGB(:,:,1) = CT_gray;
%             CT_RGB(:,:,2) = CT_gray;
%             CT_RGB(:,:,3) = CT_gray;
%             imshow(CT_RGB, 'Parent',app.CTImg_axes);
            
            % select ROI
            roiPos = [];
            %imageHandle = imshow(uint8(CT_RGB) , 'parent', app.CTImg_axes );
            set(imageHandle,'ButtonDownFcn',@CTImgClicked);
            function CTImgClicked ( ~ , eventData )
                disp("in func");
                 coords = eventData.IntersectionPoint;
                   if size(roiPos,2) < 4
                        roiPos(size(roiPos,2)+1:size(roiPos,2)+2) = [round(coords(2)),round(coords(1))];
                    else
                        roiPos =[];
                        roiPos(size(roiPos,2)+1:size(roiPos,2)+2) = [round(coords(2)),round(coords(1))];
                   end
                   disp(roiPos);
                   disp("inside bs out of if")
                   if size(roiPos,2) == 4
                       startr = min(roiPos(1),roiPos(3));
                        endr = max(roiPos(1),roiPos(3));
                        startc = min(roiPos(2),roiPos(4));
                        endc = max(roiPos(2),roiPos(4));
                        h= endr-startr;
                        w= endc-startc;
                        %disp(range);
                        colors = jet();
                        colors = uint8(255 * mat2gray(colors));
                        red = colors(:,1);
                        green = colors(:,2);
                        blue = colors(:,3);
                        CT_colored = CT_RGB;
                        Var = uint8(255 * mat2gray(Variance(startr:endr,startc:endc)));
                        CT_colored(startr:endr,startc:endc,1) = red(Var+1);
                        CT_colored(startr:endr,startc:endc,2) = green(Var+1);
                        CT_colored(startr:endr,startc:endc,3) = blue(Var+1);
                        imageHandle = imshow(uint8(CT_colored), 'Parent',app.CTImg_axes);
                        set(imageHandle,'ButtonDownFcn',@CTImgClicked);
                        recPos = [startc ,startr, w  , h];
                        delete(app.rec1); 
                        disp("inside");
                        app.rec1 = rectangle(app.CTImg_axes, 'Position', recPos, "EdgeColor",'#0072BD',"LineWidth",1);
                   end 
                   % "FaceColor",[0.3010 ,0.7450 ,0.9330,0.2]
            end

        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [0 50 1107 768];
            app.UIFigure.Name = 'MATLAB App';

            % Create TabGroup
            app.TabGroup = uitabgroup(app.UIFigure);
            app.TabGroup.Position = [1 0 1108 769];

            % Create ImageDisplayerTab
            app.ImageDisplayerTab = uitab(app.TabGroup);
            app.ImageDisplayerTab.Title = 'Image Displayer';
            app.ImageDisplayerTab.BackgroundColor = [0.102 0.4196 0.6314];
            app.ImageDisplayerTab.ForegroundColor = [0 0.4471 0.7412];

            % Create BrowseButton
            app.BrowseButton = uibutton(app.ImageDisplayerTab, 'push');
            app.BrowseButton.ButtonPushedFcn = createCallbackFcn(app, @BrowseButtonPushed, true);
            app.BrowseButton.FontName = 'Arial';
            app.BrowseButton.FontSize = 36;
            app.BrowseButton.FontColor = [0 0.4471 0.7412];
            app.BrowseButton.Position = [48 619 227 72];
            app.BrowseButton.Text = 'Browse';

            % Create Label
            app.Label = uilabel(app.ImageDisplayerTab);
            app.Label.WordWrap = 'on';
            app.Label.FontSize = 20;
            app.Label.FontColor = [0.8 0.8 0.8];
            app.Label.Position = [48 105 308 403];
            app.Label.Text = '';

            % Create ConvertToGrayscaleButton
            app.ConvertToGrayscaleButton = uibutton(app.ImageDisplayerTab, 'push');
            app.ConvertToGrayscaleButton.ButtonPushedFcn = createCallbackFcn(app, @ConvertToGrayscaleButtonPushed, true);
            app.ConvertToGrayscaleButton.FontSize = 20;
            app.ConvertToGrayscaleButton.FontColor = [0 0.4471 0.7412];
            app.ConvertToGrayscaleButton.Position = [48 525 227 79];
            app.ConvertToGrayscaleButton.Text = 'Convert To Grayscale';

            % Create UIAxes
            app.UIAxes = uiaxes(app.ImageDisplayerTab);
            app.UIAxes.PlotBoxAspectRatio = [1 1.25 1];
            app.UIAxes.XColor = 'none';
            app.UIAxes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.UIAxes.YColor = 'none';
            app.UIAxes.ZColor = 'none';
            app.UIAxes.GridColor = 'none';
            app.UIAxes.MinorGridColor = 'none';
            app.UIAxes.Position = [426 242 350 501];

            % Create PixelscoloringTab
            app.PixelscoloringTab = uitab(app.TabGroup);
            app.PixelscoloringTab.Title = 'Pixels coloring';
            app.PixelscoloringTab.BackgroundColor = [0.102 0.4196 0.6314];
            app.PixelscoloringTab.ForegroundColor = [0 0.4471 0.7412];

            % Create CreateimageButton
            app.CreateimageButton = uibutton(app.PixelscoloringTab, 'push');
            app.CreateimageButton.ButtonPushedFcn = createCallbackFcn(app, @createImagePushed, true);
            app.CreateimageButton.FontName = 'Arial';
            app.CreateimageButton.FontSize = 20;
            app.CreateimageButton.FontColor = [0 0.4471 0.7412];
            app.CreateimageButton.Position = [48 607 227 78];
            app.CreateimageButton.Text = 'Create image';

            % Create colorpixelsButton
            app.colorpixelsButton = uibutton(app.PixelscoloringTab, 'push');
            app.colorpixelsButton.ButtonPushedFcn = createCallbackFcn(app, @colorPixelsPushed, true);
            app.colorpixelsButton.FontName = 'Arial';
            app.colorpixelsButton.FontSize = 20;
            app.colorpixelsButton.FontColor = [0 0.4471 0.7412];
            app.colorpixelsButton.Position = [48 436 227 90];
            app.colorpixelsButton.Text = 'color pixels';

            % Create Image
            app.Image = uiimage(app.PixelscoloringTab);
            app.Image.Position = [659 370 282 373];

            % Create UIAxes_2
            app.UIAxes_2 = uiaxes(app.PixelscoloringTab);
            app.UIAxes_2.PlotBoxAspectRatio = [1 1.59810126582278 1];
            app.UIAxes_2.XColor = 'none';
            app.UIAxes_2.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.UIAxes_2.YColor = 'none';
            app.UIAxes_2.ZColor = 'none';
            app.UIAxes_2.GridColor = [0.15 0.15 0.15];
            app.UIAxes_2.MinorGridColor = 'none';
            app.UIAxes_2.ButtonDownFcn = createCallbackFcn(app, @createImagePushed, true);
            app.UIAxes_2.Position = [367 286 282 442];

            % Create ZoomTab
            app.ZoomTab = uitab(app.TabGroup);
            app.ZoomTab.Title = 'Zoom';
            app.ZoomTab.BackgroundColor = [0.8 0.8 0.8];
            app.ZoomTab.ForegroundColor = [0 0.4471 0.7412];

            % Create NearestNeighborButton
            app.NearestNeighborButton = uibutton(app.ZoomTab, 'push');
            app.NearestNeighborButton.ButtonPushedFcn = createCallbackFcn(app, @NearestNeighborButtonPushed, true);
            app.NearestNeighborButton.FontName = 'Arial';
            app.NearestNeighborButton.FontSize = 20;
            app.NearestNeighborButton.FontColor = [0 0.4471 0.7412];
            app.NearestNeighborButton.Position = [48 573 186 62];
            app.NearestNeighborButton.Text = 'Nearest Neighbor';

            % Create BilinearButton
            app.BilinearButton = uibutton(app.ZoomTab, 'push');
            app.BilinearButton.ButtonPushedFcn = createCallbackFcn(app, @BilinearButtonPushed, true);
            app.BilinearButton.FontSize = 20;
            app.BilinearButton.FontColor = [0 0.4471 0.7412];
            app.BilinearButton.Position = [270 573 190 62];
            app.BilinearButton.Text = 'Bilinear ';

            % Create InputZoomingfactorEditField_2Label
            app.InputZoomingfactorEditField_2Label = uilabel(app.ZoomTab);
            app.InputZoomingfactorEditField_2Label.HorizontalAlignment = 'right';
            app.InputZoomingfactorEditField_2Label.FontName = 'Arial';
            app.InputZoomingfactorEditField_2Label.FontSize = 20;
            app.InputZoomingfactorEditField_2Label.FontColor = [0 0.4471 0.7412];
            app.InputZoomingfactorEditField_2Label.Position = [48 681 190 25];
            app.InputZoomingfactorEditField_2Label.Text = 'Input Zooming factor';

            % Create InputZoomingfactorEditField_2
            app.InputZoomingfactorEditField_2 = uieditfield(app.ZoomTab, 'numeric');
            app.InputZoomingfactorEditField_2.FontName = 'Arial';
            app.InputZoomingfactorEditField_2.FontSize = 20;
            app.InputZoomingfactorEditField_2.FontColor = [0 0.4471 0.7412];
            app.InputZoomingfactorEditField_2.Position = [253 681 207 25];
            app.InputZoomingfactorEditField_2.Value = 1;

            % Create NearestNeighborPanel
            app.NearestNeighborPanel = uipanel(app.ZoomTab);
            app.NearestNeighborPanel.ForegroundColor = [0 0.4471 0.7412];
            app.NearestNeighborPanel.TitlePosition = 'centertop';
            app.NearestNeighborPanel.Title = 'Nearest Neighbor';
            app.NearestNeighborPanel.FontWeight = 'bold';
            app.NearestNeighborPanel.Scrollable = 'on';
            app.NearestNeighborPanel.FontSize = 20;
            app.NearestNeighborPanel.Position = [0 -1 551 467];

            % Create Nearest_UIAxes
            app.Nearest_UIAxes = uiaxes(app.NearestNeighborPanel);
            app.Nearest_UIAxes.PlotBoxAspectRatio = [1 1.25 1];
            app.Nearest_UIAxes.XColor = 'none';
            app.Nearest_UIAxes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.Nearest_UIAxes.YColor = 'none';
            app.Nearest_UIAxes.ZColor = 'none';
            app.Nearest_UIAxes.GridColor = [0.15 0.15 0.15];
            app.Nearest_UIAxes.MinorGridColor = 'none';
            app.Nearest_UIAxes.Position = [0 1 548 435];

            % Create BilinearPanel
            app.BilinearPanel = uipanel(app.ZoomTab);
            app.BilinearPanel.ForegroundColor = [0 0.4471 0.7412];
            app.BilinearPanel.TitlePosition = 'centertop';
            app.BilinearPanel.Title = 'Bilinear';
            app.BilinearPanel.FontWeight = 'bold';
            app.BilinearPanel.Scrollable = 'on';
            app.BilinearPanel.FontSize = 20;
            app.BilinearPanel.Position = [551 -1 551 467];

            % Create Bilinear_UIAxes
            app.Bilinear_UIAxes = uiaxes(app.BilinearPanel);
            app.Bilinear_UIAxes.PlotBoxAspectRatio = [1 1.25 1];
            app.Bilinear_UIAxes.XColor = 'none';
            app.Bilinear_UIAxes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.Bilinear_UIAxes.YColor = 'none';
            app.Bilinear_UIAxes.ZColor = 'none';
            app.Bilinear_UIAxes.GridColor = [0.15 0.15 0.15];
            app.Bilinear_UIAxes.MinorGridColor = 'none';
            app.Bilinear_UIAxes.Position = [0 0 551 436];

            % Create BeforeZoomingPanel
            app.BeforeZoomingPanel = uipanel(app.ZoomTab);
            app.BeforeZoomingPanel.ForegroundColor = [0 0.4471 0.7412];
            app.BeforeZoomingPanel.TitlePosition = 'centertop';
            app.BeforeZoomingPanel.Title = 'Before Zooming';
            app.BeforeZoomingPanel.FontWeight = 'bold';
            app.BeforeZoomingPanel.Scrollable = 'on';
            app.BeforeZoomingPanel.FontSize = 20;
            app.BeforeZoomingPanel.Position = [551 465 552 278];

            % Create NoZoom_UIAxes
            app.NoZoom_UIAxes = uiaxes(app.BeforeZoomingPanel);
            app.NoZoom_UIAxes.PlotBoxAspectRatio = [1 1.25 1];
            app.NoZoom_UIAxes.XColor = 'none';
            app.NoZoom_UIAxes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.NoZoom_UIAxes.YColor = 'none';
            app.NoZoom_UIAxes.ZColor = 'none';
            app.NoZoom_UIAxes.GridColor = [0.15 0.15 0.15];
            app.NoZoom_UIAxes.MinorGridColor = 'none';
            app.NoZoom_UIAxes.Position = [11 9 532 237];

            % Create HistogramTab
            app.HistogramTab = uitab(app.TabGroup);
            app.HistogramTab.Title = 'Histogram';
            app.HistogramTab.BackgroundColor = [0.102 0.4196 0.6314];
            app.HistogramTab.ForegroundColor = [0 0.4471 0.7412];

            % Create equalizedimagePanel
            app.equalizedimagePanel = uipanel(app.HistogramTab);
            app.equalizedimagePanel.ForegroundColor = [0 0.4471 0.7412];
            app.equalizedimagePanel.TitlePosition = 'centertop';
            app.equalizedimagePanel.Title = 'equalized image';
            app.equalizedimagePanel.FontWeight = 'bold';
            app.equalizedimagePanel.Scrollable = 'on';
            app.equalizedimagePanel.FontSize = 20;
            app.equalizedimagePanel.Position = [573 286 513 349];

            % Create equalized_image_axes
            app.equalized_image_axes = uiaxes(app.equalizedimagePanel);
            app.equalized_image_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.equalized_image_axes.XColor = 'none';
            app.equalized_image_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.equalized_image_axes.YColor = 'none';
            app.equalized_image_axes.ZColor = 'none';
            app.equalized_image_axes.GridColor = [0.15 0.15 0.15];
            app.equalized_image_axes.MinorGridColor = 'none';
            app.equalized_image_axes.Position = [4 0 503 319];

            % Create originalimagePanel
            app.originalimagePanel = uipanel(app.HistogramTab);
            app.originalimagePanel.ForegroundColor = [0 0.4471 0.7412];
            app.originalimagePanel.TitlePosition = 'centertop';
            app.originalimagePanel.Title = 'original image';
            app.originalimagePanel.FontWeight = 'bold';
            app.originalimagePanel.Scrollable = 'on';
            app.originalimagePanel.FontSize = 20;
            app.originalimagePanel.Position = [3 286 550 349];

            % Create original_image_axes
            app.original_image_axes = uiaxes(app.originalimagePanel);
            app.original_image_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.original_image_axes.XColor = 'none';
            app.original_image_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.original_image_axes.YColor = 'none';
            app.original_image_axes.YTick = [0 0.2 0.4 0.6 0.8 1];
            app.original_image_axes.ZColor = 'none';
            app.original_image_axes.GridColor = [0.15 0.15 0.15];
            app.original_image_axes.MinorGridColor = 'none';
            app.original_image_axes.Position = [2 0 548 318];

            % Create originalhistogramPanel
            app.originalhistogramPanel = uipanel(app.HistogramTab);
            app.originalhistogramPanel.ForegroundColor = [0 0.4471 0.7412];
            app.originalhistogramPanel.TitlePosition = 'centertop';
            app.originalhistogramPanel.Title = 'original histogram';
            app.originalhistogramPanel.FontWeight = 'bold';
            app.originalhistogramPanel.Scrollable = 'on';
            app.originalhistogramPanel.FontSize = 20;
            app.originalhistogramPanel.Position = [2 1 550 277];

            % Create original_histogram_axes
            app.original_histogram_axes = uiaxes(app.originalhistogramPanel);
            xlabel(app.original_histogram_axes, 'intensity')
            ylabel(app.original_histogram_axes, 'frequency')
            zlabel(app.original_histogram_axes, 'Z')
            app.original_histogram_axes.PlotBoxAspectRatio = [2.67094017094017 1 1];
            app.original_histogram_axes.XLim = [0 255];
            app.original_histogram_axes.XTick = [0 50 100 150 200 250];
            app.original_histogram_axes.XTickLabel = {'0'; '50'; '100'; '150'; '200'; '250'};
            app.original_histogram_axes.Position = [5 3 544 243];

            % Create equalizedhistogramPanel
            app.equalizedhistogramPanel = uipanel(app.HistogramTab);
            app.equalizedhistogramPanel.ForegroundColor = [0 0.4471 0.7412];
            app.equalizedhistogramPanel.TitlePosition = 'centertop';
            app.equalizedhistogramPanel.Title = 'equalized histogram';
            app.equalizedhistogramPanel.FontWeight = 'bold';
            app.equalizedhistogramPanel.Scrollable = 'on';
            app.equalizedhistogramPanel.FontSize = 20;
            app.equalizedhistogramPanel.Position = [574 1 512 277];

            % Create equalized_histogram_axes
            app.equalized_histogram_axes = uiaxes(app.equalizedhistogramPanel);
            xlabel(app.equalized_histogram_axes, 'intensity')
            ylabel(app.equalized_histogram_axes, 'frequency')
            zlabel(app.equalized_histogram_axes, 'Z')
            app.equalized_histogram_axes.PlotBoxAspectRatio = [2.54666666666667 1 1];
            app.equalized_histogram_axes.XLim = [0 255];
            app.equalized_histogram_axes.Position = [3 10 503 236];

            % Create ShowHistogramButton
            app.ShowHistogramButton = uibutton(app.HistogramTab, 'push');
            app.ShowHistogramButton.ButtonPushedFcn = createCallbackFcn(app, @ShowHistogramButtonPushed, true);
            app.ShowHistogramButton.FontSize = 20;
            app.ShowHistogramButton.FontColor = [0 0.4471 0.7412];
            app.ShowHistogramButton.Position = [111 658 328 70];
            app.ShowHistogramButton.Text = 'Show Histogram';

            % Create EqualizeButton
            app.EqualizeButton = uibutton(app.HistogramTab, 'push');
            app.EqualizeButton.ButtonPushedFcn = createCallbackFcn(app, @EqualizeButtonPushed, true);
            app.EqualizeButton.FontSize = 20;
            app.EqualizeButton.FontColor = [0 0.4471 0.7412];
            app.EqualizeButton.Position = [636 658 328 70];
            app.EqualizeButton.Text = 'Equalize';

            % Create SpatialFilteringTab
            app.SpatialFilteringTab = uitab(app.TabGroup);
            app.SpatialFilteringTab.Title = 'Spatial Filtering';
            app.SpatialFilteringTab.BackgroundColor = [0 0.4471 0.7412];
            app.SpatialFilteringTab.ForegroundColor = [0 0.4471 0.7412];

            % Create BeforFilterPanel
            app.BeforFilterPanel = uipanel(app.SpatialFilteringTab);
            app.BeforFilterPanel.ForegroundColor = [0 0.4471 0.7412];
            app.BeforFilterPanel.TitlePosition = 'centertop';
            app.BeforFilterPanel.Title = 'original image';
            app.BeforFilterPanel.FontWeight = 'bold';
            app.BeforFilterPanel.Scrollable = 'on';
            app.BeforFilterPanel.FontSize = 20;
            app.BeforFilterPanel.Position = [367 379 352 349];

            % Create BeforeFilter_axes
            app.BeforeFilter_axes = uiaxes(app.BeforFilterPanel);
            app.BeforeFilter_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.BeforeFilter_axes.XColor = 'none';
            app.BeforeFilter_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.BeforeFilter_axes.YColor = 'none';
            app.BeforeFilter_axes.YTick = [0 0.2 0.4 0.6 0.8 1];
            app.BeforeFilter_axes.ZColor = 'none';
            app.BeforeFilter_axes.GridColor = [0.15 0.15 0.15];
            app.BeforeFilter_axes.MinorGridColor = 'none';
            app.BeforeFilter_axes.Position = [1 3 339 313];

            % Create AfterFilteringPanel
            app.AfterFilteringPanel = uipanel(app.SpatialFilteringTab);
            app.AfterFilteringPanel.ForegroundColor = [0 0.4471 0.7412];
            app.AfterFilteringPanel.TitlePosition = 'centertop';
            app.AfterFilteringPanel.Title = 'After Filtering';
            app.AfterFilteringPanel.FontWeight = 'bold';
            app.AfterFilteringPanel.Scrollable = 'on';
            app.AfterFilteringPanel.FontSize = 20;
            app.AfterFilteringPanel.Position = [736 12 355 349];

            % Create AfterFiltering_axes
            app.AfterFiltering_axes = uiaxes(app.AfterFilteringPanel);
            app.AfterFiltering_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.AfterFiltering_axes.XColor = 'none';
            app.AfterFiltering_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.AfterFiltering_axes.YColor = 'none';
            app.AfterFiltering_axes.ZColor = 'none';
            app.AfterFiltering_axes.GridColor = [0.15 0.15 0.15];
            app.AfterFiltering_axes.MinorGridColor = 'none';
            app.AfterFiltering_axes.Position = [1 1 348 318];

            % Create ApplyFilterButton
            app.ApplyFilterButton = uibutton(app.SpatialFilteringTab, 'push');
            app.ApplyFilterButton.ButtonPushedFcn = createCallbackFcn(app, @ApplyFilterButtonPushed, true);
            app.ApplyFilterButton.FontName = 'Arial';
            app.ApplyFilterButton.FontSize = 20;
            app.ApplyFilterButton.FontColor = [0 0.4471 0.7412];
            app.ApplyFilterButton.Position = [60 330 227 78];
            app.ApplyFilterButton.Text = 'Apply Filter';

            % Create KernelsizeEditField_2Label
            app.KernelsizeEditField_2Label = uilabel(app.SpatialFilteringTab);
            app.KernelsizeEditField_2Label.BackgroundColor = [0.9412 0.9412 0.9412];
            app.KernelsizeEditField_2Label.HorizontalAlignment = 'right';
            app.KernelsizeEditField_2Label.FontSize = 20;
            app.KernelsizeEditField_2Label.FontColor = [0 0.4471 0.7412];
            app.KernelsizeEditField_2Label.Position = [32 690 104 25];
            app.KernelsizeEditField_2Label.Text = 'Kernel size';

            % Create KernelsizeEditField_2
            app.KernelsizeEditField_2 = uieditfield(app.SpatialFilteringTab, 'numeric');
            app.KernelsizeEditField_2.Position = [151 693 163 22];

            % Create factorKEditFieldLabel
            app.factorKEditFieldLabel = uilabel(app.SpatialFilteringTab);
            app.factorKEditFieldLabel.BackgroundColor = [0.9412 0.9412 0.9412];
            app.factorKEditFieldLabel.HorizontalAlignment = 'right';
            app.factorKEditFieldLabel.FontSize = 20;
            app.factorKEditFieldLabel.FontColor = [0 0.4471 0.7412];
            app.factorKEditFieldLabel.Position = [33 610 75 25];
            app.factorKEditFieldLabel.Text = 'factor K';

            % Create factorKEditField
            app.factorKEditField = uieditfield(app.SpatialFilteringTab, 'numeric');
            app.factorKEditField.Position = [123 613 192 22];

            % Create BoxFilterPanel
            app.BoxFilterPanel = uipanel(app.SpatialFilteringTab);
            app.BoxFilterPanel.ForegroundColor = [0 0.4471 0.7412];
            app.BoxFilterPanel.TitlePosition = 'centertop';
            app.BoxFilterPanel.Title = 'Box Filter';
            app.BoxFilterPanel.FontWeight = 'bold';
            app.BoxFilterPanel.Scrollable = 'on';
            app.BoxFilterPanel.FontSize = 20;
            app.BoxFilterPanel.Position = [739 378 355 349];

            % Create BoxFilter_axes
            app.BoxFilter_axes = uiaxes(app.BoxFilterPanel);
            app.BoxFilter_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.BoxFilter_axes.XColor = 'none';
            app.BoxFilter_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.BoxFilter_axes.YColor = 'none';
            app.BoxFilter_axes.ZColor = 'none';
            app.BoxFilter_axes.GridColor = [0.15 0.15 0.15];
            app.BoxFilter_axes.MinorGridColor = 'none';
            app.BoxFilter_axes.Position = [1 1 348 318];

            % Create edgesPanel
            app.edgesPanel = uipanel(app.SpatialFilteringTab);
            app.edgesPanel.ForegroundColor = [0 0.4471 0.7412];
            app.edgesPanel.TitlePosition = 'centertop';
            app.edgesPanel.Title = 'Edges';
            app.edgesPanel.FontWeight = 'bold';
            app.edgesPanel.Scrollable = 'on';
            app.edgesPanel.FontSize = 20;
            app.edgesPanel.Position = [364 13 355 348];

            % Create edges_axes
            app.edges_axes = uiaxes(app.edgesPanel);
            app.edges_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.edges_axes.XColor = 'none';
            app.edges_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.edges_axes.YColor = 'none';
            app.edges_axes.ZColor = 'none';
            app.edges_axes.GridColor = [0.15 0.15 0.15];
            app.edges_axes.MinorGridColor = 'none';
            app.edges_axes.Position = [1 8 348 310];

            % Create ScalingmethodButtonGroup
            app.ScalingmethodButtonGroup = uibuttongroup(app.SpatialFilteringTab);
            app.ScalingmethodButtonGroup.ForegroundColor = [0 0.4471 0.7412];
            app.ScalingmethodButtonGroup.Title = 'Scaling method';
            app.ScalingmethodButtonGroup.FontSize = 20;
            app.ScalingmethodButtonGroup.Position = [60 474 227 95];

            % Create thresholdmethodButton
            app.thresholdmethodButton = uiradiobutton(app.ScalingmethodButtonGroup);
            app.thresholdmethodButton.Text = 'threshold method';
            app.thresholdmethodButton.FontSize = 16;
            app.thresholdmethodButton.FontColor = [0 0.4471 0.7412];
            app.thresholdmethodButton.Position = [11 40 146 22];

            % Create scalethenewrangeButton
            app.scalethenewrangeButton = uiradiobutton(app.ScalingmethodButtonGroup);
            app.scalethenewrangeButton.Text = 'scale the new range';
            app.scalethenewrangeButton.FontSize = 16;
            app.scalethenewrangeButton.FontColor = [0 0.4471 0.7412];
            app.scalethenewrangeButton.Position = [11 18 165 22];
            app.scalethenewrangeButton.Value = true;

            % Create Fourier1Tab
            app.Fourier1Tab = uitab(app.TabGroup);
            app.Fourier1Tab.Title = 'Fourier 1';
            app.Fourier1Tab.BackgroundColor = [0 0.4471 0.7412];
            app.Fourier1Tab.ForegroundColor = [0 0.4471 0.7412];

            % Create TimeDomainPanel
            app.TimeDomainPanel = uipanel(app.Fourier1Tab);
            app.TimeDomainPanel.ForegroundColor = [0 0.4471 0.7412];
            app.TimeDomainPanel.TitlePosition = 'centertop';
            app.TimeDomainPanel.Title = 'Image in Time Domian';
            app.TimeDomainPanel.FontWeight = 'bold';
            app.TimeDomainPanel.Scrollable = 'on';
            app.TimeDomainPanel.FontSize = 20;
            app.TimeDomainPanel.Position = [14 211 352 349];

            % Create TimeDomain_axes
            app.TimeDomain_axes = uiaxes(app.TimeDomainPanel);
            app.TimeDomain_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.TimeDomain_axes.XColor = 'none';
            app.TimeDomain_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.TimeDomain_axes.YColor = 'none';
            app.TimeDomain_axes.YTick = [0 0.2 0.4 0.6 0.8 1];
            app.TimeDomain_axes.ZColor = 'none';
            app.TimeDomain_axes.GridColor = [0.15 0.15 0.15];
            app.TimeDomain_axes.MinorGridColor = 'none';
            app.TimeDomain_axes.Position = [1 3 339 313];

            % Create PhasePanel
            app.PhasePanel = uipanel(app.Fourier1Tab);
            app.PhasePanel.ForegroundColor = [0 0.4471 0.7412];
            app.PhasePanel.TitlePosition = 'centertop';
            app.PhasePanel.Title = 'Phase';
            app.PhasePanel.FontWeight = 'bold';
            app.PhasePanel.Scrollable = 'on';
            app.PhasePanel.FontSize = 20;
            app.PhasePanel.Position = [395 14 352 349];

            % Create Phase_axes
            app.Phase_axes = uiaxes(app.PhasePanel);
            app.Phase_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.Phase_axes.XColor = 'none';
            app.Phase_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.Phase_axes.YColor = 'none';
            app.Phase_axes.YTick = [0 0.2 0.4 0.6 0.8 1];
            app.Phase_axes.ZColor = 'none';
            app.Phase_axes.GridColor = [0.15 0.15 0.15];
            app.Phase_axes.MinorGridColor = 'none';
            app.Phase_axes.Position = [1 3 339 313];

            % Create MagnitudePanel
            app.MagnitudePanel = uipanel(app.Fourier1Tab);
            app.MagnitudePanel.ForegroundColor = [0 0.4471 0.7412];
            app.MagnitudePanel.TitlePosition = 'centertop';
            app.MagnitudePanel.Title = 'Magnitude';
            app.MagnitudePanel.FontWeight = 'bold';
            app.MagnitudePanel.Scrollable = 'on';
            app.MagnitudePanel.FontSize = 20;
            app.MagnitudePanel.Position = [394 379 352 349];

            % Create Magnitude_axes
            app.Magnitude_axes = uiaxes(app.MagnitudePanel);
            app.Magnitude_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.Magnitude_axes.XColor = 'none';
            app.Magnitude_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.Magnitude_axes.YColor = 'none';
            app.Magnitude_axes.ZColor = 'none';
            app.Magnitude_axes.GridColor = [0.15 0.15 0.15];
            app.Magnitude_axes.MinorGridColor = 'none';
            app.Magnitude_axes.Position = [1 3 339 313];

            % Create DiplayFouriertransformButton
            app.DiplayFouriertransformButton = uibutton(app.Fourier1Tab, 'push');
            app.DiplayFouriertransformButton.ButtonPushedFcn = createCallbackFcn(app, @DiplayFouriertransformButtonPushed, true);
            app.DiplayFouriertransformButton.FontSize = 18;
            app.DiplayFouriertransformButton.FontColor = [0 0.4471 0.7412];
            app.DiplayFouriertransformButton.Position = [60 622 260 76];
            app.DiplayFouriertransformButton.Text = 'Diplay Fourier transform';

            % Create removefrequenciesPanel
            app.removefrequenciesPanel = uipanel(app.Fourier1Tab);
            app.removefrequenciesPanel.ForegroundColor = [0 0.4471 0.7412];
            app.removefrequenciesPanel.TitlePosition = 'centertop';
            app.removefrequenciesPanel.Title = 'remove frequencies';
            app.removefrequenciesPanel.FontWeight = 'bold';
            app.removefrequenciesPanel.FontSize = 20;
            app.removefrequenciesPanel.Position = [775 21 305 706];

            % Create selectfreqinXstartSpinnerLabel
            app.selectfreqinXstartSpinnerLabel = uilabel(app.removefrequenciesPanel);
            app.selectfreqinXstartSpinnerLabel.HorizontalAlignment = 'right';
            app.selectfreqinXstartSpinnerLabel.FontSize = 16;
            app.selectfreqinXstartSpinnerLabel.FontColor = [0 0.4471 0.7412];
            app.selectfreqinXstartSpinnerLabel.Position = [21 619 147 22];
            app.selectfreqinXstartSpinnerLabel.Text = 'select freq in X start';

            % Create selectfreqinXstartSpinner
            app.selectfreqinXstartSpinner = uispinner(app.removefrequenciesPanel);
            app.selectfreqinXstartSpinner.Position = [183 619 100 22];

            % Create removefrequenciesButton
            app.removefrequenciesButton = uibutton(app.removefrequenciesPanel, 'push');
            app.removefrequenciesButton.ButtonPushedFcn = createCallbackFcn(app, @removefrequenciesButtonPushed, true);
            app.removefrequenciesButton.BackgroundColor = [0 0.4471 0.7412];
            app.removefrequenciesButton.FontSize = 20;
            app.removefrequenciesButton.FontColor = [0.902 0.902 0.902];
            app.removefrequenciesButton.Position = [35 375 215 62];
            app.removefrequenciesButton.Text = 'remove frequencies';

            % Create selectfreqinYstartSpinnerLabel
            app.selectfreqinYstartSpinnerLabel = uilabel(app.removefrequenciesPanel);
            app.selectfreqinYstartSpinnerLabel.HorizontalAlignment = 'right';
            app.selectfreqinYstartSpinnerLabel.FontSize = 16;
            app.selectfreqinYstartSpinnerLabel.FontColor = [0 0.4471 0.7412];
            app.selectfreqinYstartSpinnerLabel.Position = [22 531 147 22];
            app.selectfreqinYstartSpinnerLabel.Text = 'select freq in Y start';

            % Create selectfreqinYstartSpinner
            app.selectfreqinYstartSpinner = uispinner(app.removefrequenciesPanel);
            app.selectfreqinYstartSpinner.Position = [184 531 100 22];

            % Create selectfreqinXendSpinnerLabel
            app.selectfreqinXendSpinnerLabel = uilabel(app.removefrequenciesPanel);
            app.selectfreqinXendSpinnerLabel.HorizontalAlignment = 'right';
            app.selectfreqinXendSpinnerLabel.FontSize = 16;
            app.selectfreqinXendSpinnerLabel.FontColor = [0 0.4471 0.7412];
            app.selectfreqinXendSpinnerLabel.Position = [25 577 143 22];
            app.selectfreqinXendSpinnerLabel.Text = 'select freq in X end';

            % Create selectfreqinXendSpinner
            app.selectfreqinXendSpinner = uispinner(app.removefrequenciesPanel);
            app.selectfreqinXendSpinner.Position = [183 577 100 22];

            % Create selectfreqinYendSpinnerLabel
            app.selectfreqinYendSpinnerLabel = uilabel(app.removefrequenciesPanel);
            app.selectfreqinYendSpinnerLabel.HorizontalAlignment = 'right';
            app.selectfreqinYendSpinnerLabel.FontSize = 16;
            app.selectfreqinYendSpinnerLabel.FontColor = [0 0.4471 0.7412];
            app.selectfreqinYendSpinnerLabel.Position = [27 483 143 22];
            app.selectfreqinYendSpinnerLabel.Text = 'select freq in Y end';

            % Create selectfreqinYendSpinner
            app.selectfreqinYendSpinner = uispinner(app.removefrequenciesPanel);
            app.selectfreqinYendSpinner.Position = [185 483 100 22];

            % Create Phase_axes_2
            app.Phase_axes_2 = uiaxes(app.removefrequenciesPanel);
            app.Phase_axes_2.PlotBoxAspectRatio = [1 1.25 1];
            app.Phase_axes_2.XColor = 'none';
            app.Phase_axes_2.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.Phase_axes_2.YColor = 'none';
            app.Phase_axes_2.ZColor = 'none';
            app.Phase_axes_2.GridColor = [0.15 0.15 0.15];
            app.Phase_axes_2.MinorGridColor = 'none';
            app.Phase_axes_2.Position = [-3 29 305 313];

            % Create Fourier2Tab
            app.Fourier2Tab = uitab(app.TabGroup);
            app.Fourier2Tab.Title = 'Fourier 2';
            app.Fourier2Tab.BackgroundColor = [0 0.4471 0.7412];
            app.Fourier2Tab.ForegroundColor = [0 0.4471 0.7412];

            % Create TimeDomainBeforePanel
            app.TimeDomainBeforePanel = uipanel(app.Fourier2Tab);
            app.TimeDomainBeforePanel.ForegroundColor = [0 0.4471 0.7412];
            app.TimeDomainBeforePanel.TitlePosition = 'centertop';
            app.TimeDomainBeforePanel.Title = 'Image in Time Domian Before filter';
            app.TimeDomainBeforePanel.FontWeight = 'bold';
            app.TimeDomainBeforePanel.Scrollable = 'on';
            app.TimeDomainBeforePanel.FontSize = 20;
            app.TimeDomainBeforePanel.Position = [518 382 352 349];

            % Create TimeDomainBefore_axes
            app.TimeDomainBefore_axes = uiaxes(app.TimeDomainBeforePanel);
            app.TimeDomainBefore_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.TimeDomainBefore_axes.XColor = 'none';
            app.TimeDomainBefore_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.TimeDomainBefore_axes.YColor = 'none';
            app.TimeDomainBefore_axes.YTick = [0 0.2 0.4 0.6 0.8 1];
            app.TimeDomainBefore_axes.ZColor = 'none';
            app.TimeDomainBefore_axes.GridColor = [0.15 0.15 0.15];
            app.TimeDomainBefore_axes.MinorGridColor = 'none';
            app.TimeDomainBefore_axes.Position = [1 3 339 313];

            % Create AfterFilterFreqPanel
            app.AfterFilterFreqPanel = uipanel(app.Fourier2Tab);
            app.AfterFilterFreqPanel.ForegroundColor = [0 0.4471 0.7412];
            app.AfterFilterFreqPanel.TitlePosition = 'centertop';
            app.AfterFilterFreqPanel.Title = 'After Frequency domain Filter';
            app.AfterFilterFreqPanel.FontWeight = 'bold';
            app.AfterFilterFreqPanel.Scrollable = 'on';
            app.AfterFilterFreqPanel.FontSize = 20;
            app.AfterFilterFreqPanel.Position = [43 11 352 349];

            % Create AfterFilterFreq_axes
            app.AfterFilterFreq_axes = uiaxes(app.AfterFilterFreqPanel);
            app.AfterFilterFreq_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.AfterFilterFreq_axes.XColor = 'none';
            app.AfterFilterFreq_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.AfterFilterFreq_axes.YColor = 'none';
            app.AfterFilterFreq_axes.ZColor = 'none';
            app.AfterFilterFreq_axes.GridColor = [0.15 0.15 0.15];
            app.AfterFilterFreq_axes.MinorGridColor = 'none';
            app.AfterFilterFreq_axes.Position = [1 3 339 313];

            % Create DifferencePanel
            app.DifferencePanel = uipanel(app.Fourier2Tab);
            app.DifferencePanel.ForegroundColor = [0 0.4471 0.7412];
            app.DifferencePanel.TitlePosition = 'centertop';
            app.DifferencePanel.Title = 'Diffrence between filter in T & F Domains';
            app.DifferencePanel.FontWeight = 'bold';
            app.DifferencePanel.Scrollable = 'on';
            app.DifferencePanel.FontSize = 17;
            app.DifferencePanel.Position = [519 11 352 349];

            % Create Difference_axes
            app.Difference_axes = uiaxes(app.DifferencePanel);
            app.Difference_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.Difference_axes.XColor = 'none';
            app.Difference_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.Difference_axes.YColor = 'none';
            app.Difference_axes.ZColor = 'none';
            app.Difference_axes.GridColor = [0.15 0.15 0.15];
            app.Difference_axes.MinorGridColor = 'none';
            app.Difference_axes.Position = [1 7 339 313];

            % Create KernelsizeEditField_3Label
            app.KernelsizeEditField_3Label = uilabel(app.Fourier2Tab);
            app.KernelsizeEditField_3Label.BackgroundColor = [0 0.4471 0.7412];
            app.KernelsizeEditField_3Label.HorizontalAlignment = 'right';
            app.KernelsizeEditField_3Label.FontSize = 20;
            app.KernelsizeEditField_3Label.FontColor = [0.9412 0.9412 0.9412];
            app.KernelsizeEditField_3Label.Position = [48 658 104 25];
            app.KernelsizeEditField_3Label.Text = 'Kernel size';

            % Create Kernelsize2
            app.Kernelsize2 = uieditfield(app.Fourier2Tab, 'numeric');
            app.Kernelsize2.Position = [167 661 217 22];

            % Create ApplyfilterinFrequencyDomainButton
            app.ApplyfilterinFrequencyDomainButton = uibutton(app.Fourier2Tab, 'push');
            app.ApplyfilterinFrequencyDomainButton.ButtonPushedFcn = createCallbackFcn(app, @ApplyfilterinFrequencyDomainButtonPushed, true);
            app.ApplyfilterinFrequencyDomainButton.FontSize = 18;
            app.ApplyfilterinFrequencyDomainButton.FontColor = [0 0.4471 0.7412];
            app.ApplyfilterinFrequencyDomainButton.Position = [91 532 280 76];
            app.ApplyfilterinFrequencyDomainButton.Text = 'Apply filter in Frequency Domain';

            % Create commentLabel
            app.commentLabel = uilabel(app.Fourier2Tab);
            app.commentLabel.WordWrap = 'on';
            app.commentLabel.FontSize = 18;
            app.commentLabel.FontColor = [0.902 0.902 0.902];
            app.commentLabel.Position = [91 407 278 101];
            app.commentLabel.Text = '';

            % Create NoiseTab
            app.NoiseTab = uitab(app.TabGroup);
            app.NoiseTab.Title = 'Noise';
            app.NoiseTab.BackgroundColor = [0 0.4471 0.7412];
            app.NoiseTab.ForegroundColor = [0 0.4471 0.7412];

            % Create PhantomPanel
            app.PhantomPanel = uipanel(app.NoiseTab);
            app.PhantomPanel.ForegroundColor = [0 0.4471 0.7412];
            app.PhantomPanel.TitlePosition = 'centertop';
            app.PhantomPanel.Title = 'Phantom';
            app.PhantomPanel.FontWeight = 'bold';
            app.PhantomPanel.Scrollable = 'on';
            app.PhantomPanel.FontSize = 20;
            app.PhantomPanel.Position = [14 362 352 371];

            % Create DisplayPhantomButton
            app.DisplayPhantomButton = uibutton(app.PhantomPanel, 'push');
            app.DisplayPhantomButton.ButtonPushedFcn = createCallbackFcn(app, @DisplayPhantomButtonPushed, true);
            app.DisplayPhantomButton.FontSize = 20;
            app.DisplayPhantomButton.FontColor = [0 0.4471 0.7412];
            app.DisplayPhantomButton.Position = [57 8 224 51];
            app.DisplayPhantomButton.Text = 'Display Phantom';

            % Create phantom_axes
            app.phantom_axes = uiaxes(app.PhantomPanel);
            app.phantom_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.phantom_axes.XColor = 'none';
            app.phantom_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.phantom_axes.YColor = 'none';
            app.phantom_axes.ZColor = 'none';
            app.phantom_axes.GridColor = [0.15 0.15 0.15];
            app.phantom_axes.MinorGridColor = 'none';
            app.phantom_axes.ButtonDownFcn = createCallbackFcn(app, @phantom_axesButtonDown, true);
            app.phantom_axes.Position = [1 64 339 274];

            % Create NoisyImgPanel
            app.NoisyImgPanel = uipanel(app.NoiseTab);
            app.NoisyImgPanel.ForegroundColor = [0 0.4471 0.7412];
            app.NoisyImgPanel.TitlePosition = 'centertop';
            app.NoisyImgPanel.Title = 'Noisy Image';
            app.NoisyImgPanel.FontWeight = 'bold';
            app.NoisyImgPanel.Scrollable = 'on';
            app.NoisyImgPanel.FontSize = 20;
            app.NoisyImgPanel.Position = [673 379 352 349];

            % Create NoisyImg_axes
            app.NoisyImg_axes = uiaxes(app.NoisyImgPanel);
            app.NoisyImg_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.NoisyImg_axes.XColor = 'none';
            app.NoisyImg_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.NoisyImg_axes.YColor = 'none';
            app.NoisyImg_axes.ZColor = 'none';
            app.NoisyImg_axes.GridColor = [0.15 0.15 0.15];
            app.NoisyImg_axes.MinorGridColor = 'none';
            app.NoisyImg_axes.ButtonDownFcn = createCallbackFcn(app, @SelectNoiseButtonGroupSelectionChanged, true);
            app.NoisyImg_axes.Position = [1 3 339 313];

            % Create HistogramPanel
            app.HistogramPanel = uipanel(app.NoiseTab);
            app.HistogramPanel.ForegroundColor = [0 0.4471 0.7412];
            app.HistogramPanel.TitlePosition = 'centertop';
            app.HistogramPanel.Title = 'Histogram';
            app.HistogramPanel.FontWeight = 'bold';
            app.HistogramPanel.Scrollable = 'on';
            app.HistogramPanel.FontSize = 20;
            app.HistogramPanel.Position = [420 21 605 324];

            % Create histogram_axes
            app.histogram_axes = uiaxes(app.HistogramPanel);
            xlabel(app.histogram_axes, 'intensity')
            ylabel(app.histogram_axes, 'frequency')
            zlabel(app.histogram_axes, 'Z')
            app.histogram_axes.PlotBoxAspectRatio = [2.67094017094017 1 1];
            app.histogram_axes.XLim = [0 280];
            app.histogram_axes.XTick = [0 50 100 150 200 250];
            app.histogram_axes.XTickLabel = {'0'; '50'; '100'; '150'; '200'; '250'};
            app.histogram_axes.Position = [5 9 544 284];

            % Create SelectROIButton
            app.SelectROIButton = uibutton(app.NoiseTab, 'push');
            app.SelectROIButton.ButtonPushedFcn = createCallbackFcn(app, @SelectROIButtonPushed, true);
            app.SelectROIButton.FontSize = 20;
            app.SelectROIButton.FontColor = [0 0.4471 0.7412];
            app.SelectROIButton.Position = [411 396 238 80];
            app.SelectROIButton.Text = 'Select ROI';

            % Create ROIPanel
            app.ROIPanel = uipanel(app.NoiseTab);
            app.ROIPanel.ForegroundColor = [0 0.4471 0.7412];
            app.ROIPanel.TitlePosition = 'centertop';
            app.ROIPanel.Title = 'ROI';
            app.ROIPanel.FontWeight = 'bold';
            app.ROIPanel.Scrollable = 'on';
            app.ROIPanel.FontSize = 20;
            app.ROIPanel.Position = [13 17 352 334];

            % Create ROI_axes
            app.ROI_axes = uiaxes(app.ROIPanel);
            app.ROI_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.ROI_axes.XColor = 'none';
            app.ROI_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.ROI_axes.YColor = 'none';
            app.ROI_axes.ZColor = 'none';
            app.ROI_axes.GridColor = [0.15 0.15 0.15];
            app.ROI_axes.MinorGridColor = 'none';
            app.ROI_axes.Position = [1 13 339 288];

            % Create SelectNoiseButtonGroup
            app.SelectNoiseButtonGroup = uibuttongroup(app.NoiseTab);
            app.SelectNoiseButtonGroup.SelectionChangedFcn = createCallbackFcn(app, @SelectNoiseButtonGroupSelectionChanged, true);
            app.SelectNoiseButtonGroup.ForegroundColor = [0 0.4471 0.7412];
            app.SelectNoiseButtonGroup.Title = 'Select Noise';
            app.SelectNoiseButtonGroup.FontSize = 20;
            app.SelectNoiseButtonGroup.Position = [420 585 229 142];

            % Create GaussiannoiseButton_2
            app.GaussiannoiseButton_2 = uiradiobutton(app.SelectNoiseButtonGroup);
            app.GaussiannoiseButton_2.Text = 'Gaussian noise';
            app.GaussiannoiseButton_2.FontSize = 16;
            app.GaussiannoiseButton_2.FontColor = [0 0.4471 0.7412];
            app.GaussiannoiseButton_2.Position = [11 86 132 22];

            % Create UniformnoiseButton_2
            app.UniformnoiseButton_2 = uiradiobutton(app.SelectNoiseButtonGroup);
            app.UniformnoiseButton_2.Text = 'Uniform noise';
            app.UniformnoiseButton_2.FontSize = 16;
            app.UniformnoiseButton_2.FontColor = [0 0.4471 0.7412];
            app.UniformnoiseButton_2.Position = [11 64 121 22];

            % Create NoneButton_2
            app.NoneButton_2 = uiradiobutton(app.SelectNoiseButtonGroup);
            app.NoneButton_2.Text = 'None';
            app.NoneButton_2.FontSize = 16;
            app.NoneButton_2.FontColor = [0 0.4471 0.7412];
            app.NoneButton_2.Position = [11 16 65 22];
            app.NoneButton_2.Value = true;

            % Create SaltpepperButton
            app.SaltpepperButton = uiradiobutton(app.SelectNoiseButtonGroup);
            app.SaltpepperButton.Text = 'Salt & pepper';
            app.SaltpepperButton.FontSize = 16;
            app.SaltpepperButton.FontColor = [0 0.4471 0.7412];
            app.SaltpepperButton.Position = [11 37 119 22];

            % Create SaltLabel
            app.SaltLabel = uilabel(app.NoiseTab);
            app.SaltLabel.BackgroundColor = [0 0.4471 0.7412];
            app.SaltLabel.HorizontalAlignment = 'right';
            app.SaltLabel.FontSize = 20;
            app.SaltLabel.FontColor = [0.9412 0.9412 0.9412];
            app.SaltLabel.Position = [431 551 63 25];
            app.SaltLabel.Text = 'Salt %';

            % Create saltPercent
            app.saltPercent = uieditfield(app.NoiseTab, 'numeric');
            app.saltPercent.Position = [539 554 110 22];

            % Create pepperLabel
            app.pepperLabel = uilabel(app.NoiseTab);
            app.pepperLabel.BackgroundColor = [0 0.4471 0.7412];
            app.pepperLabel.HorizontalAlignment = 'right';
            app.pepperLabel.FontSize = 20;
            app.pepperLabel.FontColor = [0.9412 0.9412 0.9412];
            app.pepperLabel.Position = [433 501 91 25];
            app.pepperLabel.Text = 'pepper %';

            % Create pepperPercent
            app.pepperPercent = uieditfield(app.NoiseTab, 'numeric');
            app.pepperPercent.Position = [539 504 110 22];

            % Create BackProjectionTab
            app.BackProjectionTab = uitab(app.TabGroup);
            app.BackProjectionTab.Title = 'Back Projection';
            app.BackProjectionTab.BackgroundColor = [0 0.4471 0.7412];
            app.BackProjectionTab.ForegroundColor = [0 0.4471 0.7412];

            % Create lamino5Panel
            app.lamino5Panel = uipanel(app.BackProjectionTab);
            app.lamino5Panel.ForegroundColor = [0 0.4471 0.7412];
            app.lamino5Panel.TitlePosition = 'centertop';
            app.lamino5Panel.Title = 'Laminogram 5 projections';
            app.lamino5Panel.FontWeight = 'bold';
            app.lamino5Panel.Scrollable = 'on';
            app.lamino5Panel.FontSize = 20;
            app.lamino5Panel.Position = [576 381 352 349];

            % Create Lamino5_axes
            app.Lamino5_axes = uiaxes(app.lamino5Panel);
            app.Lamino5_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.Lamino5_axes.XColor = 'none';
            app.Lamino5_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.Lamino5_axes.YColor = 'none';
            app.Lamino5_axes.YTick = [0 0.2 0.4 0.6 0.8 1];
            app.Lamino5_axes.ZColor = 'none';
            app.Lamino5_axes.GridColor = [0.15 0.15 0.15];
            app.Lamino5_axes.MinorGridColor = 'none';
            app.Lamino5_axes.Position = [1 3 339 313];

            % Create phantom2panel
            app.phantom2panel = uipanel(app.BackProjectionTab);
            app.phantom2panel.ForegroundColor = [0 0.4471 0.7412];
            app.phantom2panel.TitlePosition = 'centertop';
            app.phantom2panel.Title = 'Shepp-Logan phantom';
            app.phantom2panel.FontWeight = 'bold';
            app.phantom2panel.Scrollable = 'on';
            app.phantom2panel.FontSize = 20;
            app.phantom2panel.Position = [168 378 352 349];

            % Create SheppPhantom_axes
            app.SheppPhantom_axes = uiaxes(app.phantom2panel);
            app.SheppPhantom_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.SheppPhantom_axes.XColor = 'none';
            app.SheppPhantom_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.SheppPhantom_axes.YColor = 'none';
            app.SheppPhantom_axes.YTick = [0 0.2 0.4 0.6 0.8 1];
            app.SheppPhantom_axes.ZColor = 'none';
            app.SheppPhantom_axes.GridColor = [0.15 0.15 0.15];
            app.SheppPhantom_axes.MinorGridColor = 'none';
            app.SheppPhantom_axes.Position = [6 7 339 270];

            % Create lamino180Panel
            app.lamino180Panel = uipanel(app.BackProjectionTab);
            app.lamino180Panel.ForegroundColor = [0 0.4471 0.7412];
            app.lamino180Panel.TitlePosition = 'centertop';
            app.lamino180Panel.Title = 'Laminogram 180 projection';
            app.lamino180Panel.FontWeight = 'bold';
            app.lamino180Panel.Scrollable = 'on';
            app.lamino180Panel.FontSize = 20;
            app.lamino180Panel.Position = [12 10 352 349];

            % Create lamino180_axes
            app.lamino180_axes = uiaxes(app.lamino180Panel);
            app.lamino180_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.lamino180_axes.XColor = 'none';
            app.lamino180_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.lamino180_axes.YColor = 'none';
            app.lamino180_axes.YTick = [0 0.2 0.4 0.6 0.8 1];
            app.lamino180_axes.ZColor = 'none';
            app.lamino180_axes.GridColor = [0.15 0.15 0.15];
            app.lamino180_axes.MinorGridColor = 'none';
            app.lamino180_axes.Position = [1 3 339 313];

            % Create ramLackPanel
            app.ramLackPanel = uipanel(app.BackProjectionTab);
            app.ramLackPanel.ForegroundColor = [0 0.4471 0.7412];
            app.ramLackPanel.TitlePosition = 'centertop';
            app.ramLackPanel.Title = 'Ram-lack filter';
            app.ramLackPanel.FontWeight = 'bold';
            app.ramLackPanel.Scrollable = 'on';
            app.ramLackPanel.FontSize = 20;
            app.ramLackPanel.Position = [382 11 352 349];

            % Create ramLack_axes
            app.ramLack_axes = uiaxes(app.ramLackPanel);
            app.ramLack_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.ramLack_axes.XColor = 'none';
            app.ramLack_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.ramLack_axes.YColor = 'none';
            app.ramLack_axes.YTick = [0 0.2 0.4 0.6 0.8 1];
            app.ramLack_axes.ZColor = 'none';
            app.ramLack_axes.GridColor = [0.15 0.15 0.15];
            app.ramLack_axes.MinorGridColor = 'none';
            app.ramLack_axes.Position = [1 3 339 313];

            % Create hammingPanel
            app.hammingPanel = uipanel(app.BackProjectionTab);
            app.hammingPanel.ForegroundColor = [0 0.4471 0.7412];
            app.hammingPanel.TitlePosition = 'centertop';
            app.hammingPanel.Title = 'Hamming filter';
            app.hammingPanel.FontWeight = 'bold';
            app.hammingPanel.Scrollable = 'on';
            app.hammingPanel.FontSize = 20;
            app.hammingPanel.Position = [750 10 352 349];

            % Create hamming_axes
            app.hamming_axes = uiaxes(app.hammingPanel);
            app.hamming_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.hamming_axes.XColor = 'none';
            app.hamming_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.hamming_axes.YColor = 'none';
            app.hamming_axes.YTick = [0 0.2 0.4 0.6 0.8 1];
            app.hamming_axes.ZColor = 'none';
            app.hamming_axes.GridColor = [0.15 0.15 0.15];
            app.hamming_axes.MinorGridColor = 'none';
            app.hamming_axes.Position = [1 3 339 313];

            % Create GetBackProjectoinButton
            app.GetBackProjectoinButton = uibutton(app.BackProjectionTab, 'push');
            app.GetBackProjectoinButton.ButtonPushedFcn = createCallbackFcn(app, @GetBackProjectoinButtonPushed2, true);
            app.GetBackProjectoinButton.FontSize = 20;
            app.GetBackProjectoinButton.FontColor = [0 0.4471 0.7412];
            app.GetBackProjectoinButton.Position = [66 692 238 41];
            app.GetBackProjectoinButton.Text = 'Get Back Projectoin';

            % Create sinogramPanel
            app.sinogramPanel = uipanel(app.BackProjectionTab);
            app.sinogramPanel.ForegroundColor = [0 0.4471 0.7412];
            app.sinogramPanel.TitlePosition = 'centertop';
            app.sinogramPanel.Title = 'phantom sinogram';
            app.sinogramPanel.FontWeight = 'bold';
            app.sinogramPanel.Scrollable = 'on';
            app.sinogramPanel.FontSize = 20;
            app.sinogramPanel.Position = [377 379 352 349];

            % Create sinogram_axes
            app.sinogram_axes = uiaxes(app.sinogramPanel);
            app.sinogram_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.sinogram_axes.XColor = 'none';
            app.sinogram_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.sinogram_axes.YColor = 'none';
            app.sinogram_axes.YTick = [0 0.2 0.4 0.6 0.8 1];
            app.sinogram_axes.ZColor = 'none';
            app.sinogram_axes.GridColor = [0.15 0.15 0.15];
            app.sinogram_axes.MinorGridColor = 'none';
            app.sinogram_axes.Position = [1 3 339 313];

            % Create ColormapTab
            app.ColormapTab = uitab(app.TabGroup);
            app.ColormapTab.Title = 'Color map';
            app.ColormapTab.ForegroundColor = [0 0.4471 0.7412];

            % Create CTImgPanel
            app.CTImgPanel = uipanel(app.ColormapTab);
            app.CTImgPanel.ForegroundColor = [0 0.4471 0.7412];
            app.CTImgPanel.TitlePosition = 'centertop';
            app.CTImgPanel.Title = ' CT Image';
            app.CTImgPanel.FontWeight = 'bold';
            app.CTImgPanel.Scrollable = 'on';
            app.CTImgPanel.FontSize = 20;
            app.CTImgPanel.Position = [253 246 454 454];

            % Create CTImg_axes
            app.CTImg_axes = uiaxes(app.CTImgPanel);
            app.CTImg_axes.PlotBoxAspectRatio = [1 1.25 1];
            app.CTImg_axes.XColor = 'none';
            app.CTImg_axes.XTick = [0 0.2 0.4 0.6 0.8 1];
            app.CTImg_axes.YColor = 'none';
            app.CTImg_axes.ZColor = 'none';
            app.CTImg_axes.GridColor = [0.15 0.15 0.15];
            app.CTImg_axes.MinorGridColor = 'none';
            app.CTImg_axes.ButtonDownFcn = createCallbackFcn(app, @CTImg_axesButtonDown, true);
            app.CTImg_axes.Position = [1 0 453 421];

            % Create displaytheimageButton
            app.displaytheimageButton = uibutton(app.ColormapTab, 'push');
            app.displaytheimageButton.ButtonPushedFcn = createCallbackFcn(app, @displaytheimageButtonPushed, true);
            app.displaytheimageButton.Position = [319 94 355 90];
            app.displaytheimageButton.Text = 'display the image';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = FullImageFilteringTool

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end