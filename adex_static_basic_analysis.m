
% load data. This is the datafile you specify when calling adex_network2column.py
% sample load command to execute before running this script:
%load D:/vbox/test_adex.mat
%
% Included in this .mat file are outputs of all neurons. Variable names L3/4/5 refer to layer 3/4/5. 
% APs = action potentials, lfp = local field potential (sum of all E/IPSPs). final digit 1/2 refers to column 1/2.
% FS = fast spiking inhibitory neuron, RS = regular spiking excitatory neuron, BU = bursting excitatory neuron
% spiking data were not used in this paper, but are exported for completeness.

% sampling rate
srate = 1000/mean(diff(lfptimes));

%% plot LFP power

% average all layers together
lfpdata = ( mean(L3_RS_lfp1,1)+mean(L4_BU_lfp1,1)+mean(L4_RS_lfp1,1)+mean(L5_BU_lfp1,1)+mean(L5_RS_lfp1,1) ) ./ 5;

% compute FFT
lfp_power = abs(fft(lfpdata))*2;

% frequencies for power spectra
hz = linspace(0,srate,size(lfpdata,2)/2+1);

% plot
figure(1), clf
plot(hz,lfp_power(1:length(hz)))
set(gca,'xlim',[1 100])
xlabel('Frequency (Hz)'), ylabel('Amplitude')
title('LFP power spectrum for one trial')

% Note that the power spectrum here is noisier than in the figure in the paper. 
% This is because the paper averages over 100 trials (where each trial has the same 
% parameters but different random seeds for noise, connectivity, synapse strengths, etc.)

%% Plotting spiking data

% *APs variables contain neuron number and spike time. For example:
figure
plot(L3_RS_APs2(:,2),L3_RS_APs2(:,1),'k.','markersize',2)
xlabel('Time in seconds'), ylabel('Neuron number')
title('Action potential timing in L3, column 2')
