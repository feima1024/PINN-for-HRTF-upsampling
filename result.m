load('40.mat');                        % load the data into work place 
load('40_L3.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot the result over 7 frequencies
figure('position',[0,0,1800,1200]);
sgtitle('Interpolation experiment @ Magnitude of left-ear HRTF of subject 40 from the HUTUBS database');
for ii=1:7
    freq = freq_bins(ii); 
    freq = round(freq/100)/10; % in kHz                          % the frequency of interest
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(7,4,ii*4-3);                                         % plot the ground truth 
    temp = squeeze(total_hrtf(ii,:,:));
    total_hrtf_mag = abs( temp(1,:) + 1i*temp(2,:));   
    total_hrtf_mag   = abs(total_hrtf_mag);
    T                  = delaunay(total_coor(:,8),total_coor(:,7));
    trisurf(T,total_coor(:,8),total_coor(:,7),total_hrtf_mag); 
    figure_conf();
    var=strcat('ground truth @',num2str(freq),' kHz');
    title(var);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(7,4,ii*4-2);                                         % plot the training data 
    temp = squeeze(train_hrtf(ii,:,:));
    ground_truth_mag = abs( temp(1,:) + 1i*temp(2,:));   
    ground_truth_mag   = abs(ground_truth_mag);
    T                  = delaunay(train_coor(:,8),train_coor(:,7));
    trisurf(T,train_coor(:,8),train_coor(:,7),ground_truth_mag); 
    figure_conf();
    var=strcat('train hrtf @',num2str(freq),' kHz');
    title(var);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(7,4,ii*4-1);                                         % plot the ground truth estimation 
    temp = squeeze(total_est(ii,:,:));
    total_est_mag = abs( temp(1,:) + 1i*temp(2,:));   
    total_est_mag = abs(total_est_mag);
    T             = delaunay(total_coor(:,8),total_coor(:,7));
    trisurf(T,total_coor(:,8),total_coor(:,7),total_est_mag); 
    figure_conf();
    var=strcat('interpolation  @',num2str(freq),' kHz');
    title(var);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(7,4,ii*4);                                         % plot the difference between the ground truth and the estimation
    est_err_mag   = abs(total_hrtf_mag - total_est_mag); 
    T             = delaunay(total_coor(:,8),total_coor(:,7));
    trisurf(T,total_coor(:,8),total_coor(:,7),est_err_mag); 
    figure_conf();
    var=strcat('abs( ground truth - interpolation)  @',num2str(freq),' kHz');
    title(var);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


